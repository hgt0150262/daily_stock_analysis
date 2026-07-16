# -*- coding: utf-8 -*-
"""Regression tests for recommendation-driven stock selection."""

from types import SimpleNamespace
from unittest.mock import patch

from src.services.stock_recommender import get_recommended_stocks, recommend_us_stocks


def test_get_recommended_stocks_combines_and_deduplicates_markets() -> None:
    config = SimpleNamespace(us_recommend_top_n=3, cn_recommend_top_n=2)

    with patch(
        "src.services.stock_recommender.recommend_us_stocks",
        return_value=["AAPL", "msft", "AAPL"],
    ) as recommend_us, patch(
        "src.services.stock_recommender.recommend_cn_stocks",
        return_value=["600519", "MSFT"],
    ) as recommend_cn:
        result = get_recommended_stocks(config)

    assert result == ["AAPL", "msft", "600519"]
    recommend_us.assert_called_once_with(config, 3)
    recommend_cn.assert_called_once_with(config, 2)


def test_recommend_us_stocks_falls_back_when_tradingview_fails() -> None:
    config = SimpleNamespace(us_recommend_source="tradingview")

    with patch(
        "src.services.stock_recommender._recommend_us_tradingview",
        side_effect=RuntimeError("provider unavailable"),
    ), patch(
        "src.services.stock_recommender._recommend_us_builtin",
        return_value=["MSFT", "AAPL"],
    ) as builtin:
        result = recommend_us_stocks(config, 2)

    assert result == ["MSFT", "AAPL"]
    builtin.assert_called_once_with(config, 2)


def test_recommend_us_stocks_falls_back_when_tradingview_is_empty() -> None:
    config = SimpleNamespace(us_recommend_source="tradingview")

    with patch(
        "src.services.stock_recommender._recommend_us_tradingview",
        return_value=[],
    ), patch(
        "src.services.stock_recommender._recommend_us_builtin",
        return_value=["NVDA"],
    ):
        result = recommend_us_stocks(config, 1)

    assert result == ["NVDA"]


def test_recommend_us_stocks_returns_empty_when_both_sources_fail() -> None:
    config = SimpleNamespace(us_recommend_source="tradingview")

    with patch(
        "src.services.stock_recommender._recommend_us_tradingview",
        side_effect=RuntimeError("provider unavailable"),
    ), patch(
        "src.services.stock_recommender._recommend_us_builtin",
        side_effect=RuntimeError("fallback unavailable"),
    ):
        result = recommend_us_stocks(config, 1)

    assert result == []
