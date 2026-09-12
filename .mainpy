from typing import List, Optional
from fastapi import Body, FastAPI, HTTPException, Query
from pydantic import BaseModel
import numpy as np
import yfinance as yf

app = FastAPI()

class PortfolioRiskRequest(BaseModel):
    tickers: List[str]

class PortfolioRiskBody(BaseModel):
    tickers: List[str]

def calculate_risk(ticker: str) -> dict:
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1y")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not fetch data for ticker: {ticker}") from exc

    if data.empty:
        raise HTTPException(status_code=404, detail=f"No historical data found for ticker: {ticker}")

    closing_prices = data["Close"]
    daily_returns = closing_prices.pct_change().dropna()

    if daily_returns.empty:
        raise HTTPException(status_code=400, detail=f"Not enough historical data for ticker: {ticker}")

    annualized_volatility = float(daily_returns.std() * np.sqrt(252))
    var_95 = float(np.percentile(daily_returns, 5))
    worst_returns = daily_returns[daily_returns <= var_95]
    expected_shortfall = float(worst_returns.mean()) if not worst_returns.empty else var_95
    latest_price = float(closing_prices.iloc[-1])

    return {
        "ticker": ticker,
        "latest_price": latest_price,
        "annualized_volatility": annualized_volatility,
        "var_95": var_95,
        "expected_shortfall_95": expected_shortfall,
    }
    
def calculate_portfolio_risk(tickers: list) -> dict:
    try:
        data = yf.download(tickers, period="1y")["Close"]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not fetch data for tickers: {tickers}") from exc

    if data.empty:
        raise HTTPException(status_code=404, detail=f"No historical data found for tickers: {tickers}")

    daily_returns = data.pct_change().dropna()

    if daily_returns.empty:
        raise HTTPException(status_code=400, detail=f"Not enough historical data for tickers: {tickers}")

    portfolio_daily_returns = daily_returns.mean(axis=1)
    annualized_volatility = float(portfolio_daily_returns.std() * np.sqrt(252))
    var_95 = float(np.percentile(portfolio_daily_returns, 5))
    worst_returns = portfolio_daily_returns[portfolio_daily_returns <= var_95]
    expected_shortfall = float(worst_returns.mean()) if not worst_returns.empty else var_95
    latest_prices = data.iloc[-1].to_dict()

    return {
        "tickers": tickers,
        "latest_prices": latest_prices,
        "annualized_volatility": annualized_volatility,
        "var_95": var_95,
        "expected_shortfall_95": expected_shortfall,
    }   


@app.get("/")
def read_root():
    return {"message": "Financial Risk API is running successfully!"}


@app.api_route("/risk/{ticker}", methods=["GET", "POST"])
def get_risk(ticker: str):
    return calculate_risk(ticker)

@app.api_route("/portfolio/risk", methods=["GET", "POST"])
@app.api_route("/portfolio/risk.", methods=["GET", "POST"])
def get_portfolio_risk(
    tickers: Optional[List[str]] = Query(default=None),
    payload: Optional[PortfolioRiskRequest] = Body(default=None),
):
    if payload is not None:
        tickers = payload.tickers
    elif not tickers:
        raise HTTPException(
            status_code=422,
            detail="Provide tickers as query parameters or in a JSON body.",
        )

    return calculate_portfolio_risk(tickers)
