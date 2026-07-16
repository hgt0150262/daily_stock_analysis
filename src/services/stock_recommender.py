"""
个股推荐服务

为「推荐驱动分析」模式（RECOMMEND_ANALYSIS_ENABLED=true）提供每日候选股票：
- 美股：默认使用 TradingView Screener（tradingview-screener 包）全市场扫描，
  按技术面综合评级（Recommend.All）排序取 Top N；失败时自动回退内置
  yfinance 动量筛选器（固定高流动性候选池）。
- A 股：可选复用 AlphaSift 选股策略（需 ALPHASIFT_ENABLED=true）。

推荐结果供 main.py 在未显式指定股票时替代 STOCK_LIST 进行个股分析；
任何一路推荐失败均不阻断主流程（返回空列表，由调用方回退 STOCK_LIST）。
"""

from __future__ import annotations

import logging
from typing import List

from src.config import Config

logger = logging.getLogger(__name__)

# 内置美股候选池（高流动性大盘股，作为 TradingView 不可用时的回退数据源；
# 可用 US_RECOMMEND_UNIVERSE 覆盖）
DEFAULT_US_UNIVERSE: List[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "BRK-B", "LLY",
    "JPM", "V", "XOM", "UNH", "MA", "PG", "COST", "HD", "JNJ", "ORCL",
    "MRK", "ABBV", "CVX", "CRM", "BAC", "KO", "AMD", "PEP", "WMT", "NFLX",
    "TMO", "ADBE", "LIN", "MCD", "CSCO", "ACN", "ABT", "QCOM", "INTU", "IBM",
    "GE", "CAT", "TXN", "AMAT", "VZ", "DIS", "PFE", "CMCSA", "UBER", "NOW",
    "GS", "MS", "AXP", "RTX", "HON", "BKNG", "ISRG", "NEE", "SPGI", "LOW",
    "T", "UNP", "SCHW", "PGR", "ETN", "SYK", "MU", "BLK", "LRCX", "PLTR",
    "PANW", "CRWD", "ANET", "KLAC", "SNPS", "CDNS", "MRVL", "ABNB", "SBUX", "GILD",
]

# TradingView 扫描过滤阈值
_TV_MIN_MARKET_CAP_USD = 10_000_000_000  # 市值 > 100 亿美元
_TV_MIN_VOLUME = 1_000_000               # 日成交量 > 100 万股


def get_recommended_stocks(config: Config) -> List[str]:
    """返回今日推荐的个股代码列表（美股 Top N + 可选 A 股 Top N）。

    失败时返回已成功部分（可能为空列表），不抛异常。
    """
    codes: List[str] = []

    us_top_n = max(0, int(getattr(config, "us_recommend_top_n", 0) or 0))
    if us_top_n > 0:
        codes.extend(recommend_us_stocks(config, us_top_n))

    cn_top_n = max(0, int(getattr(config, "cn_recommend_top_n", 0) or 0))
    if cn_top_n > 0:
        codes.extend(recommend_cn_stocks(config, cn_top_n))

    # 去重并保序
    seen = set()
    unique_codes = []
    for code in codes:
        key = code.upper()
        if key not in seen:
            seen.add(key)
            unique_codes.append(code)
    return unique_codes


# ---------------------------------------------------------------------------
# 美股推荐
# ---------------------------------------------------------------------------

def recommend_us_stocks(config: Config, top_n: int) -> List[str]:
    """美股推荐入口：默认 TradingView Screener，失败回退内置筛选器。"""
    source = (getattr(config, "us_recommend_source", "tradingview") or "tradingview").lower()
    if source != "builtin":
        try:
            result = _recommend_us_tradingview(config, top_n)
            if result:
                return result
            logger.warning("[推荐] TradingView 扫描返回空结果，回退内置筛选器")
        except Exception as exc:
            logger.warning("[推荐] TradingView 扫描失败（%s），回退内置筛选器", exc)
    try:
        return _recommend_us_builtin(config, top_n)
    except Exception as exc:
        logger.warning("[推荐] 内置美股筛选器失败: %s", exc)
        return []


def _recommend_us_tradingview(config: Config, top_n: int) -> List[str]:
    """使用 TradingView Screener 全市场扫描美股。

    过滤：NASDAQ/NYSE 主板普通股 + 市值 > 100 亿美元 + 成交量 > 100 万股，
    按 TradingView 技术面综合评级 Recommend.All 降序取 Top N。
    """
    from tradingview_screener import Query, col

    query = (
        Query()
        .set_markets("america")
        .select("name", "close", "volume", "market_cap_basic", "Recommend.All", "exchange")
        .where(
            col("exchange").isin(["NASDAQ", "NYSE"]),
            col("is_primary") == True,  # noqa: E712 - tradingview-screener 表达式语法
            col("typespecs").has("common"),
            col("market_cap_basic") > _TV_MIN_MARKET_CAP_USD,
            col("volume") > _TV_MIN_VOLUME,
        )
        .order_by("Recommend.All", ascending=False)
        .limit(max(top_n, 1))
    )
    total_count, df = query.get_scanner_data(timeout=60)
    if df is None or df.empty:
        return []

    picks: List[str] = []
    for _, row in df.iterrows():
        symbol = str(row.get("name") or "").strip().upper()
        if not symbol:
            continue
        rating = row.get("Recommend.All")
        logger.info(
            "[推荐] TradingView 美股候选: %s (评级 %.2f, 收盘 %s)",
            symbol, float(rating) if rating is not None else float("nan"), row.get("close"),
        )
        picks.append(symbol)
        if len(picks) >= top_n:
            break
    logger.info("[推荐] TradingView 全市场命中 %s 只，选出 Top %s: %s", total_count, top_n, picks)
    return picks


def _get_us_universe(config: Config) -> List[str]:
    universe = getattr(config, "us_recommend_universe", None) or []
    if universe:
        return [str(c).strip().upper() for c in universe if str(c).strip()]
    return list(DEFAULT_US_UNIVERSE)


def _recommend_us_builtin(config: Config, top_n: int) -> List[str]:
    """内置美股筛选器（回退方案）。

    对固定候选池逐票计算复合得分：20 日动量 + 均线趋势（收盘 > MA20 > MA60）
    + 量能放大比，取得分最高的 Top N。
    """
    import yfinance as yf

    universe = _get_us_universe(config)
    scored = []
    for symbol in universe:
        try:
            hist = yf.Ticker(symbol).history(period="4mo", interval="1d", auto_adjust=True)
            if hist is None or len(hist) < 60:
                continue
            close = hist["Close"]
            volume = hist["Volume"]
            momentum_20d = float(close.iloc[-1] / close.iloc[-21] - 1.0)
            ma20 = float(close.rolling(20).mean().iloc[-1])
            ma60 = float(close.rolling(60).mean().iloc[-1])
            last = float(close.iloc[-1])
            trend_score = (1.0 if last > ma20 else 0.0) + (1.0 if ma20 > ma60 else 0.0)
            vol_ratio = float(volume.iloc[-5:].mean() / max(volume.iloc[-60:].mean(), 1.0))
            score = momentum_20d * 100.0 + trend_score * 5.0 + min(vol_ratio, 3.0) * 2.0
            scored.append((score, symbol))
        except Exception as exc:
            logger.debug("[推荐] 内置筛选器跳过 %s: %s", symbol, exc)
    scored.sort(reverse=True)
    picks = [symbol for _, symbol in scored[:top_n]]
    logger.info("[推荐] 内置筛选器候选池 %s 只有效 %s 只，选出 Top %s: %s",
                len(universe), len(scored), top_n, picks)
    return picks


# ---------------------------------------------------------------------------
# A 股推荐（复用 AlphaSift）
# ---------------------------------------------------------------------------

def recommend_cn_stocks(config: Config, top_n: int) -> List[str]:
    """A 股推荐：复用 AlphaSift 选股策略（需 ALPHASIFT_ENABLED=true）。"""
    if not getattr(config, "alphasift_enabled", False):
        logger.info("[推荐] 未启用 AlphaSift（ALPHASIFT_ENABLED=false），跳过 A 股推荐")
        return []
    try:
        from src.services.alphasift_service import AlphaSiftService

        strategy = (getattr(config, "cn_recommend_strategy", "") or "balanced_alpha").strip()
        service = AlphaSiftService(config=config)
        result = service.screen(strategy=strategy, market="cn", max_results=top_n)
        candidates = result.get("candidates") or result.get("results") or []
        picks: List[str] = []
        for item in candidates:
            code = ""
            if isinstance(item, dict):
                code = str(item.get("code") or item.get("symbol") or item.get("ts_code") or "").strip()
            else:
                code = str(item).strip()
            code = code.split(".")[0]
            if code:
                picks.append(code)
            if len(picks) >= top_n:
                break
        logger.info("[推荐] AlphaSift(%s) A 股 Top %s: %s", strategy, top_n, picks)
        return picks
    except Exception as exc:
        logger.warning("[推荐] AlphaSift A 股推荐失败: %s", exc)
        return []
