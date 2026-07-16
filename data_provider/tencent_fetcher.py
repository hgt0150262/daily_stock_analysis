# -*- coding: utf-8 -*-
"""Tencent direct daily K-line fetcher for A-share fallback routing."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd
import requests

try:
    import exchange_calendars as xcals
except ImportError:  # pragma: no cover - dependency is present in supported installs
    xcals = None

from .base import BaseFetcher, DataFetchError, STANDARD_COLUMNS, normalize_stock_code, is_bse_code

logger = logging.getLogger(__name__)

_MAX_KLINE_BARS = 800


class TencentFetcher(BaseFetcher):
    """Fetch qfq daily K-line data from Tencent's direct quote endpoint."""

    name = "TencentFetcher"
    priority = 0
    allow_empty_daily_data = True

    _KLINE_ENDPOINT = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    _QUOTE_ENDPOINT = "https://qt.gtimg.cn/q="
    _HTTP_TIMEOUT_SECONDS = 8

    # 港/美股主要指数 -> 腾讯行情符号（qt.gtimg.cn），字段含昨收/涨跌幅，
    # 作为 Yahoo Finance 受限环境（如 GitHub Actions runner）下的大盘复盘兜底
    _INDEX_SYMBOLS = {
        "hk": [
            ("HSI", "hkHSI", "恒生指数"),
            ("HSTECH", "hkHSTECH", "恒生科技指数"),
            ("HSCEI", "hkHSCEI", "国企指数"),
        ],
        "us": [
            ("SPX", "usINX", "标普500指数"),
            ("IXIC", "usIXIC", "纳斯达克综合指数"),
            ("DJI", "usDJI", "道琼斯工业指数"),
            ("VIX", "usVIX", "VIX恐慌指数"),
        ],
    }
    _INDEX_MAX_STALE_DAYS = 5

    def get_main_indices(self, region: str = "cn") -> Optional[list[dict[str, Any]]]:
        """获取主要指数实时行情（腾讯 qt.gtimg.cn），支持港股与美股，A 股返回 None 走其他数据源。"""
        symbols = self._INDEX_SYMBOLS.get(region)
        if not symbols:
            return None
        query = ",".join(sym for _, sym, _ in symbols)
        response = requests.get(
            f"{self._QUOTE_ENDPOINT}{query}",
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://gu.qq.com/"},
            timeout=self._HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        response.encoding = "gbk"
        raw = {}
        for line in response.text.splitlines():
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            raw[key.strip().removeprefix("v_")] = value.strip().strip(";").strip('"')

        results = []
        for code, symbol, name in symbols:
            fields = raw.get(symbol, "").split("~")
            item = _parse_tencent_index_quote(
                fields,
                code=code,
                name=name,
                max_stale_days=self._INDEX_MAX_STALE_DAYS,
            )
            if item:
                results.append(item)
            else:
                logger.warning("[Tencent] 获取指数 %s(%s) 失败或数据过期", name, symbol)
        if results:
            logger.info("[Tencent] 成功获取 %d 个%s指数行情", len(results), "港股" if region == "hk" else "美股")
            return results
        return None

    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        code = normalize_stock_code(stock_code)
        symbol = _to_tencent_symbol(code)
        if not symbol:
            raise DataFetchError(f"TencentFetcher unsupported stock code: {stock_code}")

        lookback = _estimate_lookback_days(start_date=start_date, end_date=end_date)
        explicit_start = _format_tencent_date(start_date)
        explicit_end = _format_tencent_date(end_date)
        explicit_window = (
            f"{explicit_start},{explicit_end}"
            if explicit_start and explicit_end
            else ","
        )
        response = requests.get(
            self._KLINE_ENDPOINT,
            params={"param": f"{symbol},day,{explicit_window},{lookback},qfq"},
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json,text/plain,*/*"},
            timeout=self._HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        rows = _extract_kline_rows(payload, symbol=symbol)
        if not rows:
            logger.info("TencentFetcher empty daily history for %s", stock_code)
            return _empty_daily_frame()

        df = pd.DataFrame(rows)
        first_returned_date = _first_returned_date(df)
        if first_returned_date and _is_capped_history_incomplete(
            first_returned_date=first_returned_date,
            start_date=start_date,
            lookback=lookback,
            returned_rows=len(rows),
        ):
            logger.info(
                "TencentFetcher incomplete capped daily history for %s: first_date=%s requested_start=%s",
                stock_code,
                first_returned_date,
                start_date,
            )
            return _empty_daily_frame()

        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        if df.empty:
            logger.info(
                "TencentFetcher daily history outside requested range for %s: %s~%s",
                stock_code,
                start_date,
                end_date,
            )
            return _empty_daily_frame()
        return df

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        normalized = df.copy()
        for column in ("open", "high", "low", "close", "volume", "amount"):
            if column in normalized.columns:
                normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        if "pct_chg" not in normalized.columns:
            normalized["pct_chg"] = normalized["close"].pct_change().fillna(0.0) * 100
        normalized = normalized[["date", "open", "high", "low", "close", "volume", "amount", "pct_chg"]]
        return normalized


def _parse_tencent_index_quote(
    fields: list[str],
    *,
    code: str,
    name: str,
    max_stale_days: int,
) -> Optional[dict[str, Any]]:
    """解析 qt.gtimg.cn 波浪号分隔行情：3=现价 4=昨收 5=开盘 30=时间 31=涨跌额 32=涨跌幅 33=最高 34=最低"""
    if len(fields) < 35:
        return None
    try:
        price = float(fields[3])
        prev_close = float(fields[4])
        open_price = float(fields[5])
        change = float(fields[31])
        change_pct = float(fields[32])
        high = float(fields[33])
        low = float(fields[34])
    except (TypeError, ValueError):
        return None
    if price <= 0 or prev_close <= 0:
        return None
    quote_time = fields[30].strip()
    for fmt in ("%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y%m%d%H%M%S"):
        try:
            parsed = datetime.strptime(quote_time, fmt)
            break
        except ValueError:
            parsed = None
    if parsed is not None and (datetime.now() - parsed) > timedelta(days=max_stale_days):
        return None
    amplitude = ((high - low) / prev_close * 100) if (high > 0 and low > 0) else 0.0
    return {
        "code": code,
        "name": name,
        "current": price,
        "change": change,
        "change_pct": change_pct,
        "open": open_price,
        "high": high,
        "low": low,
        "prev_close": prev_close,
        "volume": 0.0,
        "amount": 0.0,
        "amplitude": amplitude,
    }


def _to_tencent_symbol(stock_code: str) -> str:
    code = normalize_stock_code(stock_code)
    if not code or not code.isdigit() or len(code) != 6:
        return ""
    if is_bse_code(code):
        return f"bj{code}"
    if code.startswith(("6", "5", "9")):
        return f"sh{code}"
    return f"sz{code}"


def _estimate_lookback_days(*, start_date: str, end_date: str) -> int:
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        calendar_days = max(1, (end - start).days + 1)
    except ValueError:
        calendar_days = 90
    # Trading days are sparse over calendar days; add margin for holidays/suspensions.
    return max(30, min(_MAX_KLINE_BARS, int(calendar_days * 1.8) + 20))


def _empty_daily_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=STANDARD_COLUMNS)


def _first_returned_date(df: pd.DataFrame) -> Optional[str]:
    if "date" not in df.columns or df.empty:
        return None
    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    if dates.empty:
        return None
    return dates.min().strftime("%Y-%m-%d")


def _is_capped_history_incomplete(
    *,
    first_returned_date: str,
    start_date: str,
    lookback: int,
    returned_rows: int,
) -> bool:
    hit_cap = lookback >= _MAX_KLINE_BARS and returned_rows >= _MAX_KLINE_BARS
    if not hit_cap:
        return False
    try:
        first = datetime.strptime(first_returned_date, "%Y-%m-%d")
        requested_start = datetime.strptime(start_date, "%Y-%m-%d")
    except ValueError:
        return False
    return first > _first_trading_date_on_or_after(requested_start)


def _first_trading_date_on_or_after(start_date: datetime) -> datetime:
    if xcals is not None:
        try:
            cal = xcals.get_calendar("XSHG")
            session = cal.date_to_session(start_date.date(), direction="next")
            return datetime.combine(session.date(), datetime.min.time())
        except Exception:
            pass

    current = start_date
    while current.weekday() >= 5:
        current += timedelta(days=1)
    return current


def _format_tencent_date(date_text: str) -> Optional[str]:
    try:
        return datetime.strptime(date_text, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        return None


def _lots_to_shares(volume: Any) -> Any:
    try:
        return float(volume) * 100
    except (TypeError, ValueError):
        return volume


def _extract_kline_rows(payload: dict[str, Any], *, symbol: str) -> list[dict[str, Any]]:
    data = payload.get("data") if isinstance(payload, dict) else None
    item = data.get(symbol) if isinstance(data, dict) else None
    if not isinstance(item, dict):
        return []
    rows = item.get("qfqday") or item.get("day") or []
    result: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 6:
            continue
        amount: Optional[Any] = row[6] if len(row) > 6 else None
        result.append(
            {
                "date": str(row[0]),
                "open": row[1],
                "close": row[2],
                "high": row[3],
                "low": row[4],
                "volume": _lots_to_shares(row[5]),
                "amount": amount,
            }
        )
    return result
