from __future__ import annotations

import re
from datetime import date, datetime, time
from functools import lru_cache
from zoneinfo import ZoneInfo

import pandas as pd

CN_TZ = ZoneInfo("Asia/Shanghai")


def now_cn() -> datetime:
    return datetime.now(CN_TZ)


def cn_today_str() -> str:
    return now_cn().date().strftime("%Y-%m-%d")


def _parse_date(date_str: str) -> date:
    return datetime.strptime(date_str, "%Y-%m-%d").date()


@lru_cache(maxsize=1)
def _load_cn_trade_dates() -> tuple[list[date], set[date]]:
    try:
        import akshare as ak  # type: ignore

        df = ak.tool_trade_date_hist_sina()
        if df is None or df.empty or "trade_date" not in df.columns:
            raise ValueError("empty trade date table")
        dates = sorted(
            pd_dt.date()
            for pd_dt in pd.to_datetime(df["trade_date"], errors="coerce")
            if str(pd_dt) != "NaT"
        )
        return dates, set(dates)
    except Exception:
        # Fallback: no holiday calendar, only weekend rule.
        return [], set()


def is_cn_symbol(symbol: str) -> bool:
    s = symbol.strip().upper()
    return bool(re.match(r"^\d{6}(\.(SH|SZ|SS))?$", s))


def is_cn_trading_day(date_str: str) -> bool:
    d = _parse_date(date_str)
    dates, dates_set = _load_cn_trade_dates()
    if dates:
        return d in dates_set
    return d.weekday() < 5


def cn_trading_dates_between(start_str: str, end_str: str, step: int = 1) -> list[str]:
    """返回 [start_str, end_str] 区间内的 A 股真实交易日列表。

    step=1 逐日采样（默认），step=5 逐周（每周第一个交易日）。
    优先用 akshare 官方交易日历（覆盖春节/国庆/清明等法定节假日），
    拉不到时降级为周一至周五。
    """
    lo = _parse_date(start_str)
    hi = _parse_date(end_str)
    dates, _ = _load_cn_trade_dates()
    if dates:
        result = [d.strftime("%Y-%m-%d") for d in dates if lo <= d <= hi]
    else:
        # fallback: 只有周末规则
        result = []
        cur = lo
        while cur <= hi:
            if cur.weekday() < 5:
                result.append(cur.strftime("%Y-%m-%d"))
            cur = cur.fromordinal(cur.toordinal() + 1)
    if step > 1:
        result = result[::step]
    return result


def previous_cn_trading_day(date_str: str) -> str:
    d = _parse_date(date_str)
    dates, _ = _load_cn_trade_dates()
    if dates:
        idx = 0
        # 找到首个 >= d 的位置
        lo, hi = 0, len(dates)
        while lo < hi:
            mid = (lo + hi) // 2
            if dates[mid] < d:
                lo = mid + 1
            else:
                hi = mid
        idx = lo - 1
        if idx >= 0:
            return dates[idx].strftime("%Y-%m-%d")
    # Fallback to weekend-only rollback
    cur = d
    while True:
        cur = cur.fromordinal(cur.toordinal() - 1)
        if cur.weekday() < 5:
            return cur.strftime("%Y-%m-%d")


def cn_market_phase(now: datetime | None = None) -> str:
    now_dt = now or now_cn()
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=CN_TZ)
    else:
        now_dt = now_dt.astimezone(CN_TZ)

    today = now_dt.date().strftime("%Y-%m-%d")
    if not is_cn_trading_day(today):
        return "closed"

    t = now_dt.time()
    if t < time(9, 30):
        return "pre_open"
    if time(9, 30) <= t < time(11, 30):
        return "in_session"
    if time(11, 30) <= t < time(13, 0):
        return "lunch_break"
    if time(13, 0) <= t < time(15, 0):
        return "in_session"
    return "post_close"


def cn_no_data_reason(date_str: str) -> str:
    if not is_cn_trading_day(date_str):
        return "N/A：非交易日（A股休市）"

    today = cn_today_str()
    if date_str == today:
        phase = cn_market_phase()
        if phase == "pre_open":
            return "N/A：今日尚未开盘"
        if phase in ("in_session", "lunch_break"):
            return "N/A：今日盘中，日线未收盘（可参考实时价）"
        if phase == "post_close":
            return "N/A：今日已收盘，数据源尚未更新"

    return "N/A：该交易日暂无数据（可能停牌或数据延迟）"
