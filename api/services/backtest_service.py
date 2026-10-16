"""
Backtest service v2 — 完整回测闭环。

改进点 (相对 v1)：
  1. 用 cn_trading_dates_between() 真实 A 股交易日历（覆盖春节/国庆），
     替代 weekday() < 5 的粗糙逻辑
  2. 回测 submit 时一次性预加载全区间 K 线进磁盘缓存 → 主循环所有价格
     查询走本地 CSV（毫秒级），不再每次在线爬
  3. 回测结果持久化 SQLite（WAL 模式），进程重启不丢
  4. 完整绩效：累计收益、夏普比率、最大回撤、胜率、持仓曲线
  5. 保留 v1 的 daemon thread + get_job 轮询 + records 结构（向后兼容）

Data flow:
  submit() → _preload_cache() [一次在线拉全区间 + 存盘]
           → daemon thread:
               for trade_date in cn_trading_dates_between(...)
                   propagate(symbol, trade_date)  # Agent 决策
                   _get_entry_and_exit()          # 走磁盘缓存
               写 SQLite
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
import time as _time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np
import pandas as pd

from tradingagents.dataflows.trade_calendar import cn_trading_dates_between


# ──────────────────────────────────────────────────────────────────────────────
# 存储：SQLite（WAL 模式，daemon-safe）
# ──────────────────────────────────────────────────────────────────────────────

def _db_path() -> str:
    from tradingagents.dataflows.config import get_config
    cache_dir = get_config().get("data_cache_dir", "data")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, "backtest_jobs.db")


_DB_LOCK = threading.Lock()


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(), check_same_thread=False, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=10000;")
    return conn


# 建表（幂等）
with _DB_LOCK:
    _conn = _get_db()
    _conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            symbol TEXT,
            start_date TEXT,
            end_date TEXT,
            selected_analysts TEXT,   -- JSON
            hold_days INTEGER,
            sample_interval INTEGER,
            config TEXT,              -- JSON
            status TEXT,              -- pending/running/completed/failed
            error TEXT,
            total_dates INTEGER DEFAULT 0,
            completed_dates INTEGER DEFAULT 0,
            started_at TEXT,
            finished_at TEXT,
            created_at TEXT
        )
    """)
    _conn.execute("""
        CREATE TABLE IF NOT EXISTS job_records (
            job_id TEXT NOT NULL,
            seq INTEGER NOT NULL,
            date TEXT,
            action TEXT,
            entry_price REAL,
            exit_price REAL,
            return_pct REAL,
            decision_summary TEXT,
            error TEXT,
            PRIMARY KEY (job_id, seq)
        )
    """)
    _conn.execute("""
        CREATE TABLE IF NOT EXISTS job_stats (
            job_id TEXT PRIMARY KEY,
            stats TEXT     -- JSON blob: 胜率/夏普/回撤/累计曲线
        )
    """)
    _conn.commit()
    _conn.close()


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ──────────────────────────────────────────────────────────────────────────────
# Job 状态 API（线程安全）
# ──────────────────────────────────────────────────────────────────────────────

def _set_job_field(job_id: str, **kwargs: Any) -> None:
    with _DB_LOCK:
        conn = _get_db()
        sets = ", ".join(f"{k}=?" for k in kwargs)
        values = list(kwargs.values()) + [job_id]
        conn.execute(f"UPDATE jobs SET {sets} WHERE job_id=?", values)
        conn.commit()
        conn.close()


def _insert_job(job_id: str, initial: Dict[str, Any]) -> None:
    with _DB_LOCK:
        conn = _get_db()
        cols = ", ".join(initial.keys())
        placeholders = ", ".join(["?"] * len(initial))
        conn.execute(f"INSERT INTO jobs ({cols}) VALUES ({placeholders})", list(initial.values()))
        conn.commit()
        conn.close()


def _insert_records(job_id: str, records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    with _DB_LOCK:
        conn = _get_db()
        cur_row = conn.execute(
            "SELECT COALESCE(MAX(seq), -1) FROM job_records WHERE job_id=?", (job_id,)
        ).fetchone()
        base_seq = (cur_row[0] if cur_row else -1) + 1
        conn.executemany(
            "INSERT INTO job_records (job_id, seq, date, action, entry_price, exit_price, return_pct, decision_summary, error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (job_id, base_seq + i, r.get("date"), r.get("action"),
                 r.get("entry_price"), r.get("exit_price"),
                 r.get("return_pct"), r.get("decision_summary"), r.get("error"))
                for i, r in enumerate(records)
            ],
        )
        conn.commit()
        conn.close()


def _upsert_stats(job_id: str, stats: Dict[str, Any]) -> None:
    with _DB_LOCK:
        conn = _get_db()
        conn.execute(
            "INSERT INTO job_stats (job_id, stats) VALUES (?, ?) "
            "ON CONFLICT(job_id) DO UPDATE SET stats=excluded.stats",
            (job_id, json.dumps(stats, ensure_ascii=False)),
        )
        conn.commit()
        conn.close()


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _DB_LOCK:
        conn = _get_db()
        row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            conn.close()
            return None
        cols = [d[0] for d in conn.execute("SELECT * FROM jobs LIMIT 0").description]
        job = dict(zip(cols, row))

        # records
        recs = conn.execute(
            "SELECT date, action, entry_price, exit_price, return_pct, decision_summary, error "
            "FROM job_records WHERE job_id=? ORDER BY seq", (job_id,)
        ).fetchall()
        job["records"] = [
            {
                "date": r[0], "action": r[1],
                "entry_price": r[2], "exit_price": r[3], "return_pct": r[4],
                "decision_summary": r[5], "error": r[6],
            }
            for r in recs
        ]

        # stats
        srow = conn.execute("SELECT stats FROM job_stats WHERE job_id=?", (job_id,)).fetchone()
        job["stats"] = json.loads(srow[0]) if srow else None

        conn.close()
        return job


def list_jobs() -> List[Dict[str, Any]]:
    with _DB_LOCK:
        conn = _get_db()
        rows = conn.execute(
            "SELECT job_id, symbol, start_date, end_date, status, created_at, total_dates, completed_dates "
            "FROM jobs ORDER BY created_at DESC"
        ).fetchall()
        conn.close()
    cols = ["job_id", "symbol", "start_date", "end_date", "status", "created_at", "total_dates", "completed_dates"]
    return [dict(zip(cols, r)) for r in rows]


def delete_job(job_id: str) -> bool:
    with _DB_LOCK:
        conn = _get_db()
        cur = conn.execute("DELETE FROM jobs WHERE job_id=?", (job_id,))
        conn.execute("DELETE FROM job_records WHERE job_id=?", (job_id,))
        conn.execute("DELETE FROM job_stats WHERE job_id=?", (job_id,))
        conn.commit()
        conn.close()
        return cur.rowcount > 0


# ──────────────────────────────────────────────────────────────────────────────
# 缓存预热：submit 时一次性把回测区间 K 线拉进磁盘缓存
# ──────────────────────────────────────────────────────────────────────────────

def _preload_cache(symbol: str, start_date: str, end_date: str, lookback_days: int = 250) -> None:
    """一次性把 [start_date - lookback_days, end_date] 全量 K 线拉进 Provider 磁盘缓存。

    通过 route_to_vendor 走正式链路 → cn_akshare provider._fetch_hist_df 自动写盘。
    """
    from tradingagents.dataflows.interface import route_to_vendor

    fmt = "%Y-%m-%d"
    pre_start = (datetime.strptime(start_date, fmt) - timedelta(days=lookback_days)).strftime(fmt)

    # 走正式路由 → 自动触发 cn_akshare 的磁盘缓存层（有就秒回，没有就在线拉并存盘）
    route_to_vendor("get_stock_data", symbol, pre_start, end_date)


def _load_price_df(symbol: str) -> Optional[pd.DataFrame]:
    """从 Provider 磁盘缓存读全历史 OHLCV，找不到触发一次 route_to_vendor 预加载。"""
    from tradingagents.dataflows.interface import route_to_vendor
    from tradingagents.dataflows.providers.cn_akshare_provider import CnAkshareProvider

    # 优先 cn_akshare provider 的磁盘缓存（A 股默认第一源）
    provider = CnAkshareProvider()
    cached = provider._try_load_from_cache(symbol)
    if cached is not None and not cached.empty:
        return cached

    # 缓存没有 → 触发一次 route_to_vendor，自动在线拉并存盘
    route_to_vendor("get_stock_data", symbol, "2000-01-01", datetime.now().strftime("%Y-%m-%d"))
    return provider._try_load_from_cache(symbol)


def _get_price_on(symbol: str, trade_date: str) -> Optional[float]:
    """返回 trade_date 或之前最近 A 股交易日的收盘价。"""
    df = _load_price_df(symbol)
    if df is None or df.empty:
        return None
    df = df[df["Date"] <= pd.to_datetime(trade_date)]
    if df.empty:
        return None
    return float(df.iloc[-1]["Close"])


def _get_exit_price(symbol: str, trade_date: str, hold_days: int) -> Optional[float]:
    """返回 trade_date 之后第 hold_days 个 A 股交易日的收盘价。"""
    df = _load_price_df(symbol)
    if df is None or df.empty:
        return None
    future = df[df["Date"] > pd.to_datetime(trade_date)]
    if len(future) < hold_days:
        return None
    return float(future.iloc[hold_days - 1]["Close"])


# ──────────────────────────────────────────────────────────────────────────────
# Agent 决策 + 分类
# ──────────────────────────────────────────────────────────────────────────────

def _run_single_analysis(symbol: str, trade_date: str, selected_analysts: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
    """Run one full analysis without SSE. Returns final state dict."""
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.dataflows.config import set_config

    set_config(config)
    graph = TradingAgentsGraph(
        selected_analysts=selected_analysts,
        debug=False,
        config=config,
    )
    final_state, _ = graph.propagate(symbol, trade_date)
    decision_raw = final_state.get("final_trade_decision", "")
    decision = graph.process_signal(decision_raw)
    return {
        "final_trade_decision": decision_raw,
        "decision": decision,
    }


def _classify_decision(decision: str) -> str:
    """Classify decision as BUY / SELL / HOLD."""
    d = decision.upper()
    if any(k in d for k in ["BUY", "增持", "买入", "BULLISH", "LONG", "看好"]):
        return "BUY"
    if any(k in d for k in ["SELL", "减持", "卖出", "BEARISH", "SHORT", "看空"]):
        return "SELL"
    return "HOLD"


# ──────────────────────────────────────────────────────────────────────────────
# 完整绩效统计
# ──────────────────────────────────────────────────────────────────────────────

def _compute_stats(records: List[Dict[str, Any]], risk_free_rate: float = 0.02) -> Dict[str, Any]:
    """计算完整绩效指标。

    risk_free_rate: 年化无风险利率（默认 2%），用于夏普比率。
    """
    trades = [r for r in records if r.get("action") in ("BUY", "SELL") and r.get("return_pct") is not None]
    if not trades:
        return {
            "total_signals": 0,
            "trade_dates_count": len(records),
            "win_rate": None,
            "avg_return_pct": None,
            "best_return_pct": None,
            "worst_return_pct": None,
            "sharpe_ratio": None,
            "max_drawdown_pct": None,
            "cumulative_return_pct": None,
        }

    wins = 0
    returns = []   # 已按 BUY/SELL 方向修正后的收益（SELL 空头对收益应是 -raw）
    for t in trades:
        ret = t["return_pct"]
        returns.append(ret)
        # Win = 对我们押的方向收益为正
        # return_pct 的语义：BUY 正=赚，SELL 正=空头赚（这里我们的 return_pct 已经是方向修正过的）
        if ret > 0:
            wins += 1

    rets_arr = np.array(returns) / 100.0   # 转成小数

    # 累计收益曲线 & 最大回撤（等权假设每次 1 单位）
    cumulative = np.cumprod(1.0 + rets_arr)
    cummax = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - cummax) / cummax
    max_dd_pct = float(np.min(drawdowns)) * 100.0

    # 夏普比率（简化：按 252 交易日年化）
    excess = rets_arr - risk_free_rate / 252.0
    if len(excess) > 1 and np.std(rets_arr) > 0:
        sharpe = float(np.mean(excess) / np.std(rets_arr) * math.sqrt(252))
    else:
        sharpe = None

    cum_ret = (cumulative[-1] - 1.0) * 100.0
    win_rate = wins / len(trades) * 100.0

    # 累计曲线采样（给前端画图用，最多 50 点）
    curve_step = max(1, len(trades) // 50)
    curve = []
    for i in range(0, len(trades), curve_step):
        curve.append({
            "date": trades[i]["date"],
            "cumulative_return_pct": float((cumulative[i] - 1.0) * 100.0),
        })
    # 最后一个点务必精确
    if curve[-1]["date"] != trades[-1]["date"]:
        curve.append({
            "date": trades[-1]["date"],
            "cumulative_return_pct": cum_ret,
        })

    return {
        "total_signals": len(trades),
        "trade_dates_count": len(records),
        "win_rate": round(win_rate, 1),
        "avg_return_pct": round(float(np.mean(rets_arr)) * 100.0, 2),
        "best_return_pct": round(float(np.max(rets_arr)) * 100.0, 2),
        "worst_return_pct": round(float(np.min(rets_arr)) * 100.0, 2),
        "sharpe_ratio": round(sharpe, 2) if sharpe is not None else None,
        "max_drawdown_pct": round(max_dd_pct, 2),
        "cumulative_return_pct": round(cum_ret, 2),
        "cumulative_curve": curve,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 后台回测主循环
# ──────────────────────────────────────────────────────────────────────────────

def _run_backtest(job_id: str, symbol: str, start_date: str, end_date: str,
                  selected_analysts: List[str], hold_days: int, sample_interval: int,
                  config: Dict[str, Any]) -> None:
    """Background thread: run backtest, 走完整闭环。"""
    _set_job_field(job_id, status="running", started_at=_utcnow_iso())

    # 1) 缓存预热（一次性拉全 K 线进磁盘缓存）
    try:
        _preload_cache(symbol, start_date, end_date, lookback_days=250)
    except Exception as exc:
        _set_job_field(job_id, status="failed", error=f"cache preload failed: {str(exc)[:200]}")
        return

    # 2) 真实 A 股交易日列表
    dates = cn_trading_dates_between(start_date, end_date, step=sample_interval)
    total = len(dates)
    _set_job_field(job_id, total_dates=total, completed_dates=0)

    records: List[Dict[str, Any]] = []

    for i, trade_date in enumerate(dates):
        record: Dict[str, Any] = {"date": trade_date, "action": "HOLD", "return_pct": None, "error": None}
        try:
            analysis = _run_single_analysis(symbol, trade_date, selected_analysts, config)
            action = _classify_decision(analysis["decision"])
            record["action"] = action
            summary = (analysis.get("final_trade_decision") or "")
            record["decision_summary"] = summary[:200]

            if action in ("BUY", "SELL"):
                entry_price = _get_price_on(symbol, trade_date)
                exit_price = _get_exit_price(symbol, trade_date, hold_days)
                if entry_price and exit_price and entry_price > 0:
                    raw_return = (exit_price - entry_price) / entry_price * 100
                    # 对我们押的方向：BUY 涨了=赚，SELL 跌了=赚
                    record["entry_price"] = round(entry_price, 2)
                    record["exit_price"] = round(exit_price, 2)
                    record["return_pct"] = round(raw_return if action == "BUY" else -raw_return, 2)
        except Exception as exc:
            record["error"] = str(exc)[:300]

        records.append(record)

        # 增量写 records 表 + 更新进度（每 5 个交易日 flush 一次）
        _insert_records(job_id, [record])
        if (i + 1) % 5 == 0 or (i + 1) == total:
            _set_job_field(job_id, completed_dates=i + 1)

    # 3) 最终统计 + 持久化
    stats = _compute_stats(records)
    _upsert_stats(job_id, stats)
    _set_job_field(job_id, status="completed", finished_at=_utcnow_iso(), completed_dates=total)


# ──────────────────────────────────────────────────────────────────────────────
# 提交入口
# ──────────────────────────────────────────────────────────────────────────────

def submit(
    symbol: str,
    start_date: str,
    end_date: str,
    selected_analysts: List[str],
    hold_days: int,
    sample_interval: int,
    config: Dict[str, Any],
) -> str:
    """Submit a backtest job. Returns job_id.

    参数:
      sample_interval: 采样间隔（交易日数）。
        1   = 逐日（Agent 每天都分析一次，最细）
        5   = 约每周一次
        10  = 双周
        20  = 约每月一次
    """
    job_id = uuid4().hex

    initial = {
        "job_id": job_id,
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "selected_analysts": json.dumps(selected_analysts, ensure_ascii=False),
        "hold_days": hold_days,
        "sample_interval": sample_interval,
        "config": json.dumps(config, ensure_ascii=False),
        "status": "pending",
        "error": None,
        "total_dates": 0,
        "completed_dates": 0,
        "started_at": None,
        "finished_at": None,
        "created_at": _utcnow_iso(),
    }
    _insert_job(job_id, initial)

    thread = threading.Thread(
        target=_run_backtest,
        args=(job_id, symbol, start_date, end_date, selected_analysts, hold_days, sample_interval, config),
        daemon=True,
    )
    thread.start()
    return job_id
