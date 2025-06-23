import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests # For making HTTP requests to other APIs

FMP_API_KEY = st.secrets["FMP-API-KEY"]
EODHD_API_KEY = st.secrets["EODHD-API-KEY"]
# --- Scoring Logic Functions (copied from previous response) ---

def score_revenue_growth(growth):
    """Scores Revenue Growth (3Y CAGR) on a 0-10 scale."""
    if growth > 0.15:
        return 10
    elif 0.10 <= growth <= 0.15:
        return 7
    elif 0.05 <= growth < 0.10:
        return 5
    else:
        return 2

def score_eps_growth(growth):
    """Scores EPS Growth (3Y) on a 0-10 scale."""
    if growth > 0.15:
        return 10
    elif 0.10 <= growth <= 0.15:
        return 7
    elif 0.05 <= growth < 0.10:
        return 5
    else:
        return 2

def score_roe(roe):
    """Scores Return on Equity (ROE) on a 0-10 scale."""
    if roe > 0.20:
        return 10
    elif 0.15 <= roe <= 0.20:
        return 7
    elif 0.10 <= roe < 0.15:
        return 5
    else:
        return 2

def score_debt_equity(debt_equity):
    """Scores Debt/Equity Ratio on a 0-10 scale."""
    if debt_equity < 0.5:
        return 10
    elif 0.5 <= debt_equity <= 1.0:
        return 7
    elif 1.0 < debt_equity <= 2.0:
        return 4
    else:
        return 1

def score_interest_coverage(icr):
    """Scores Interest Coverage Ratio on a 0-10 scale."""
    if icr > 5:
        return 10
    elif 3 <= icr <= 5:
        return 7
    elif 1 <= icr < 3:
        return 4
    else:
        return 0

def score_operating_margin(margin):
    """Scores Operating Margin on a 0-10 scale."""
    if margin > 0.20:
        return 10
    elif 0.15 <= margin <= 0.20:
        return 7
    elif 0.10 <= margin < 0.15:
        return 4
    else:
        return 2

def score_pe_ratio(pe_vs_industry):
    """Scores P/E Ratio (vs Industry) on a 0-10 scale."""
    if pe_vs_industry == 'Less than Industry Avg':
        return 10
    elif pe_vs_industry == '~ Industry Avg':
        return 5
    else: # '> Industry Avg'
        return 2

def score_peg_ratio(peg):
    """Scores PEG Ratio on a 0-10 scale."""
    if peg < 1:
        return 10
    elif 1 <= peg <= 2:
        return 5
    else:
        return 2

def score_price_book(pb):
    """Scores Price/Book Ratio on a 0-10 scale."""
    if pb < 3:
        return 10
    elif 3 <= pb <= 5:
        return 5
    else:
        return 2

def score_dividend_yield(dy):
    """Scores Dividend Yield on a 0-10 scale."""
    if dy > 0.03:
        return 10
    elif 0.01 <= dy <= 0.03:
        return 6
    else:
        return 2

def score_insider_trading(trend):
    """Scores Insider Buying/Selling trend on a 0-10 scale."""
    if trend == 'More buying':
        return 10
    elif trend == 'Net neutral':
        return 5
    else: # 'More selling'
        return 0

def score_news_sentiment(sentiment):
    """Scores News Sentiment Score on a 0-10 scale."""
    if sentiment == 'Positive':
        return 10
    elif sentiment == 'Neutral':
        return 5
    else: # 'Negative'
        return 0

def score_analyst_recommendations(rec):
    """Scores Analyst Recommendations on a 0-10 scale."""
    if rec == 'Strong Buy':
        return 10
    elif rec == 'Buy':
        return 7
    elif rec == 'Hold':
        return 5
    else: # 'Sell'
        return 0

def score_rsi(rsi):
    """Scores RSI (14-day) on a 0-10 scale."""
    if 30 <= rsi <= 70:
        return 10
    elif rsi < 30:
        return 7
    else: # rsi > 70
        return 3

def score_ma_cross(cross):
    """Scores 50D/200D MA Cross on a 0-10 scale."""
    if cross == 'Golden Cross':
        return 10
    elif cross == 'Flat':
        return 5
    else: # 'Death Cross'
        return 0

def score_macd_signal(signal):
    """Scores MACD Signal on a 0-10 scale."""
    if signal == 'Bullish crossover':
        return 10
    elif signal == 'Neutral':
        return 5
    else: # 'Bearish crossover'
        return 0

def score_volume_trend(trend):
    """Scores Volume Trend on a 0-10 scale."""
    if trend == 'Rising with price':
        return 10
    elif trend == 'Flat':
        return 5
    else: # 'Falling with price'
        return 2

def score_governance_rating(rating):
    """Scores Governance Rating on a 0-10 scale."""
    if rating == 'High':
        return 10
    elif rating == 'Average':
        return 5
    else: # 'Poor'
        return 0

def score_promoter_holding(trend):
    """Scores Promoter Holding Trend on a 0-10 scale."""
    if trend == 'Increasing':
        return 10
    elif trend == 'Stable':
        return 5
    else: # 'Decreasing'
        return 0

def score_brand_value(value):
    """Scores Brand Value / Moat on a 0-10 scale."""
    if value == 'Strong':
        return 10
    elif value == 'Moderate':
        return 5
    else: # 'Weak'
        return 0

def score_market_share(share):
    """Scores Market Share Leadership on a 0-10 scale."""
    if share == 'Leader':
        return 10
    elif share == 'Top 3':
        return 7
    else: # 'Minor player'
        return 3

def score_beta(beta):
    """Scores Beta (Volatility) on a 0-10 scale."""
    if beta < 1:
        return 10
    elif beta == 1:
        return 5
    else: # beta > 1.5
        return 2

def score_litigation_risk(risk):
    """Scores Litigation/Political Risk on a 0-10 scale."""
    if risk == 'None':
        return 10
    elif risk == 'Minor':
        return 5
    else: # 'High'
        return 0

# --- Main Calculation Function ---
def calculate_stock_score(data):
    """
    Calculates the total stock score and category scores based on input data.
    Input data dictionary should contain all parameters specified in the UI.
    """
    # Fundamental Metrics (30% weight)
    fundamentals_parameter_scores = {
        'revenue_growth': score_revenue_growth(data['revenue_growth']),
        'eps_growth': score_eps_growth(data['eps_growth']),
        'roe': score_roe(data['roe']),
        'debt_equity': score_debt_equity(data['debt_equity']),
        'interest_coverage_ratio': score_interest_coverage(data['interest_coverage_ratio']),
        'operating_margin': score_operating_margin(data['operating_margin'])
    }
    fundamentals_score = (
        fundamentals_parameter_scores['revenue_growth'] * 5 +
        fundamentals_parameter_scores['eps_growth'] * 5 +
        fundamentals_parameter_scores['roe'] * 5 +
        fundamentals_parameter_scores['debt_equity'] * 5 +
        fundamentals_parameter_scores['interest_coverage_ratio'] * 5 +
        fundamentals_parameter_scores['operating_margin'] * 5
    ) / 30 # Normalize to 0-10 scale as sum of individual weights is 30

    # Valuation (20% weight)
    valuation_parameter_scores = {
        'pe_vs_industry': score_pe_ratio(data['pe_vs_industry']),
        'peg_ratio': score_peg_ratio(data['peg_ratio']),
        'price_book_ratio': score_price_book(data['price_book_ratio']),
        'dividend_yield': score_dividend_yield(data['dividend_yield'])
    }
    valuation_score = (
        valuation_parameter_scores['pe_vs_industry'] * 7 +
        valuation_parameter_scores['peg_ratio'] * 5 +
        valuation_parameter_scores['price_book_ratio'] * 4 +
        valuation_parameter_scores['dividend_yield'] * 4
    ) / 20 # Normalize to 0-10 scale

    # Market Sentiment (10% weight)
    market_sentiment_parameter_scores = {
        'insider_trading': score_insider_trading(data['insider_trading']),
        'news_sentiment': score_news_sentiment(data['news_sentiment']),
        'analyst_recommendations': score_analyst_recommendations(data['analyst_recommendations'])
    }
    market_sentiment_score = (
        market_sentiment_parameter_scores['insider_trading'] * 4 +
        market_sentiment_parameter_scores['news_sentiment'] * 3 +
        market_sentiment_parameter_scores['analyst_recommendations'] * 3
    ) / 10 # Normalize to 0-10 scale

    # Technical Indicators (10% weight)
    technical_parameter_scores = {
        'rsi': score_rsi(data['rsi']),
        'ma_cross': score_ma_cross(data['ma_cross']),
        'macd_signal': score_macd_signal(data['macd_signal']),
        'volume_trend': score_volume_trend(data['volume_trend'])
    }
    technical_score = (
        technical_parameter_scores['rsi'] * 3 +
        technical_parameter_scores['ma_cross'] * 3 +
        technical_parameter_scores['macd_signal'] * 2 +
        technical_parameter_scores['volume_trend'] * 2
    ) / 10 # Normalize to 0-10 scale

    # Management Quality (10% weight)
    management_parameter_scores = {
        'governance_rating': score_governance_rating(data['governance_rating']),
        'promoter_holding_trend': score_promoter_holding(data['promoter_holding_trend'])
    }
    management_score = (
        management_parameter_scores['governance_rating'] * 5 +
        management_parameter_scores['promoter_holding_trend'] * 5
    ) / 10 # Normalize to 0-10 scale

    # Moat & Competitiveness (10% weight)
    moat_parameter_scores = {
        'brand_value': score_brand_value(data['brand_value']),
        'market_share': score_market_share(data['market_share'])
    }
    moat_score = (
        moat_parameter_scores['brand_value'] * 5 +
        moat_parameter_scores['market_share'] * 5
    ) / 10 # Normalize to 0-10 scale

    # Risk Metrics (10% weight)
    risk_parameter_scores = {
        'beta': score_beta(data['beta']),
        'litigation_risk': score_litigation_risk(data['litigation_risk'])
    }
    risk_score = (
        risk_parameter_scores['beta'] * 5 +
        risk_parameter_scores['litigation_risk'] * 5
    ) / 10 # Normalize to 0-10 scale

    # Calculate total score (sum of weighted normalized category scores, scaled to 0-100)
    total_score = (
        fundamentals_score * 0.30 +
        valuation_score * 0.20 +
        market_sentiment_score * 0.10 +
        technical_score * 0.10 +
        management_score * 0.10 +
        moat_score * 0.10 +
        risk_score * 0.10
    )

    # Determine recommendation
    if total_score > 75:
        recommendation = '✅ BUY'
    elif 50 <= total_score <= 75:
        recommendation = '🟡 HOLD'
    else:
        recommendation = '❌ SELL'

    return {
        'total_score': total_score,
        'recommendation': recommendation,
        'category_scores': {
            'Fundamentals': fundamentals_score * 10, # Display 0-100 for categories as well
            'Valuation': valuation_score * 10,
            'Market Sentiment': market_sentiment_score * 10,
            'Technical Trends': technical_score * 10,
            'Management Quality': management_score * 10,
            'Moat & Competitiveness': moat_score * 10,
            'Risk Factors': risk_score * 10
        }
    }

# --- Data Fetching and Calculation Functions ---

def calculate_cagr(series):
    """Calculates Compound Annual Growth Rate for a series over multiple periods."""
    if len(series) < 2:
        return 0.0
    series = series.sort_index()
    try:
        start_value = series.iloc[0]
        end_value = series.iloc[-1]
        num_periods = len(series) - 1
        if start_value <= 0:
            return 0.0
        return (end_value / start_value)**(1/num_periods) - 1
    except (ZeroDivisionError, ValueError):
        return 0.0

def calculate_rsi(data, window=14):
    """Calculates the Relative Strength Index (RSI)."""
    if 'Close' not in data.columns or len(data) < window:
        return 50
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    # Handle cases where loss is zero to avoid division by zero
    if loss.iloc[-1] == 0:
        rs = np.inf if gain.iloc[-1] > 0 else 0
    else:
        rs = gain.iloc[-1] / loss.iloc[-1]

    rsi = 100 - (100 / (1 + rs))
    return rsi if not pd.isna(rsi) else 50

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """Calculates MACD and MACD Signal."""
    if 'Close' not in data.columns or len(data) < max(fast_period, slow_period, signal_period):
        return 'Neutral'

    ema_fast = data['Close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal_period, adjust=False).mean()

    if len(macd) < 2 or len(signal_line) < 2 or pd.isna(macd.iloc[-1]) or pd.isna(signal_line.iloc[-1]):
        return 'Neutral'

    if macd.iloc[-2] < signal_line.iloc[-2] and macd.iloc[-1] > signal_line.iloc[-1]:
        return 'Bullish crossover'
    elif macd.iloc[-2] > signal_line.iloc[-2] and macd.iloc[-1] < signal_line.iloc[-1]:
        return 'Bearish crossover'
    else:
        return 'Neutral'

def get_ma_cross_status(data, short_window=50, long_window=200):
    """Determines 50D/200D MA cross status."""
    if 'Close' not in data.columns or len(data) < long_window:
        return 'Flat'

    data[f'SMA_{short_window}'] = data['Close'].rolling(window=short_window).mean()
    data[f'SMA_{long_window}'] = data['Close'].rolling(window=long_window).mean()

    if pd.isna(data[f'SMA_{long_window}'].iloc[-1]) or pd.isna(data[f'SMA_{long_window}'].iloc[-2]):
        return 'Flat'

    short_ma = data[f'SMA_{short_window}'].iloc[-1]
    long_ma = data[f'SMA_{long_window}'].iloc[-1]
    short_ma_prev = data[f'SMA_{short_window}'].iloc[-2]
    long_ma_prev = data[f'SMA_{long_window}'].iloc[-2]

    if short_ma > long_ma and short_ma_prev <= long_ma_prev:
        return 'Golden Cross'
    elif short_ma < long_ma and short_ma_prev >= long_ma_prev:
        return 'Death Cross'
    else:
        return 'Flat'

def get_volume_trend(data, short_period=30, long_period=90):
    """Determines volume trend based on recent and historical average volume."""
    if 'Volume' not in data.columns or 'Close' not in data.columns or len(data) < long_period:
        return 'Flat'

    recent_avg_volume = data['Volume'].iloc[-short_period:].mean()
    long_term_avg_volume = data['Volume'].iloc[-long_period:].mean()

    recent_avg_price = data['Close'].iloc[-short_period:].mean()
    long_term_avg_price = data['Close'].iloc[-long_period:].mean()

    if recent_avg_volume > 1.2 * long_term_avg_volume and recent_avg_price > long_term_avg_price:
        return 'Rising with price'
    elif recent_avg_volume < 0.8 * long_term_avg_volume and recent_avg_price < long_term_avg_price:
        return 'Falling with price'
    else:
        return 'Flat'

def fetch_fmp_industry_pe(industry_name):
    """Fetches industry average PE from FMP API."""
    if not FMP_API_KEY:
        return None
    # FMP's industry name might need specific formatting or a lookup table
    # This is a simplification; actual FMP API for industry PE usually requires a date and exchange.
    # A more robust solution would involve fetching all industries and finding a match.
    # For now, we'll use a generic industry PE if we can't map it directly.
    # Example FMP Industry PE endpoint: https://financialmodelingprep.com/api/v4/industry_price_earning_ratio?date=2023-10-10&exchange=NYSE&apikey=YOUR_API_KEY
    # This requires looking up the latest date and industry string.
    # Given the complexity for a free tier and specific string matching, we'll return a default.
    return None # Cannot reliably fetch with simple string match and free tier limitations

def fetch_fmp_analyst_recommendations(ticker):
    """Fetches analyst recommendations from FMP API."""
    if not FMP_API_KEY:
        return 'Hold'
    try:
        url = f"https://financialmodelingprep.com/api/v3/analyst-stock-recommendations/{ticker}?limit=1&apikey={FMP_API_KEY}"
        response = requests.get(url)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        if data:
            # FMP returns a list, usually most recent first.
            # Look for "rating" or "recommendation" field.
            # Example response structure might be {'rating': 'Buy', 'date': '...'}
            recommendation_score = data[0].get('rating', 'Hold') # Get the latest rating
            # Map FMP ratings to our categories
            if 'buy' in recommendation_score.lower() and 'strong' in recommendation_score.lower():
                return 'Strong Buy'
            elif 'buy' in recommendation_score.lower():
                return 'Buy'
            elif 'hold' in recommendation_score.lower():
                return 'Hold'
            elif 'sell' in recommendation_score.lower():
                return 'Sell'
            return 'Hold' # Default if not matched
    except requests.exceptions.RequestException as e:
        st.warning(f"FMP Analyst Recommendations API Error: {e}. Defaulting to 'Hold'.")
    except (KeyError, IndexError) as e:
        st.warning(f"FMP Analyst Recommendations data parsing error: {e}. Defaulting to 'Hold'.")
    return 'Hold'

def fetch_fmp_insider_trading(ticker):
    """Fetches insider trading data from FMP API and determines trend."""
    if not FMP_API_KEY:
        return 'Net neutral'
    try:
        # Fetch recent insider transactions
        url = f"https://financialmodelingprep.com/api/v3/insider-trading?symbol={ticker}&limit=10&apikey={FMP_API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if not data:
            return 'Net neutral'

        buy_value = 0
        sell_value = 0
        for transaction in data:
            if 'transactionType' in transaction and 'value' in transaction:
                if 'buy' in transaction['transactionType'].lower():
                    buy_value += abs(transaction['value'])
                elif 'sell' in transaction['transactionType'].lower():
                    sell_value += abs(transaction['value'])

        if buy_value > sell_value * 1.5: # Significant buying
            return 'More buying'
        elif sell_value > buy_value * 1.5: # Significant selling
            return 'More selling'
        else:
            return 'Net neutral'
    except requests.exceptions.RequestException as e:
        st.warning(f"FMP Insider Trading API Error: {e}. Defaulting to 'Net neutral'.")
    except (KeyError, IndexError) as e:
        st.warning(f"FMP Insider Trading data parsing error: {e}. Defaulting to 'Net neutral'.")
    return 'Net neutral'


def fetch_eodhd_news_sentiment(ticker):
    """Fetches news sentiment score from EODHD API."""
    if not EODHD_API_KEY:
        return 'Neutral'
    try:
        # EODHD sentiment API often requires date ranges and ticker format (e.g., AAPL.US)
        # For simplicity, we'll try to fetch recent sentiment.
        # Date for past 30 days
        today = pd.to_datetime('today').strftime('%Y-%m-%d')
        thirty_days_ago = (pd.to_datetime('today') - pd.DateOffset(days=30)).strftime('%Y-%m-%d')

        url = f"https://eodhd.com/api/sentiments?s={ticker}&from={thirty_days_ago}&to={today}&api_token={EODHD_API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if data and ticker in data and 'general' in data[ticker] and 'sentiment' in data[ticker]['general']:
            sentiment_score = data[ticker]['general']['sentiment'] # Score from -1 to 1
            if sentiment_score > 0.3:
                return 'Positive'
            elif sentiment_score < -0.3:
                return 'Negative'
            else:
                return 'Neutral'
    except requests.exceptions.RequestException as e:
        st.warning(f"EODHD News Sentiment API Error: {e}. Defaulting to 'Neutral'.")
    except (KeyError, IndexError) as e:
        st.warning(f"EODHD News Sentiment data parsing error: {e}. Defaulting to 'Neutral'.")
    return 'Neutral'


def fetch_and_process_stock_data(ticker_symbol):
    """Fetches and processes stock data using yfinance and other external APIs."""
    stock = yf.Ticker(ticker_symbol)
    data = {}
    errors = []

    # Initialize all data points with defaults (neutral/cannot determine)
    # These will be overwritten if data is successfully fetched
    data = {
        'revenue_growth': 0.0,
        'eps_growth': 0.0,
        'roe': 0.0,
        'debt_equity': 0.0,
        'interest_coverage_ratio': 0.0,
        'operating_margin': 0.0,
        'pe_vs_industry': '~ Industry Avg',
        'peg_ratio': 2.0,
        'price_book_ratio': 3.0,
        'dividend_yield': 0.0,
        'insider_trading': 'Net neutral',
        'news_sentiment': 'Neutral',
        'analyst_recommendations': 'Hold',
        'rsi': 50,
        'ma_cross': 'Flat',
        'macd_signal': 'Neutral',
        'volume_trend': 'Flat',
        'governance_rating': 'Average', # Qualitative, unlikely from free APIs
        'promoter_holding_trend': 'Stable', # Qualitative, unlikely from free APIs
        'brand_value': 'Moderate', # Qualitative, unlikely from free APIs
        'market_share': 'Minor player', # Qualitative, unlikely from free APIs
        'beta': 1.0,
        'litigation_risk': 'None' # Qualitative, unlikely from free APIs
    }

    # --- Fetching from yfinance ---
    try:
        info = stock.info
        if not info:
            errors.append(f"No info found for {ticker_symbol}. It might be an invalid ticker.")
            return data, errors # Return defaults immediately if basic info fails

        data['beta'] = info.get('beta', data['beta'])
        data['dividend_yield'] = info.get('dividendYield', data['dividend_yield'])
        data['peg_ratio'] = info.get('pegRatio', data['peg_ratio'])
        data['price_book_ratio'] = info.get('priceToBook', data['price_book_ratio'])
        current_pe = info.get('trailingPE', None) # Get current PE for comparison later
        
        # Get industry for FMP comparison
        stock_industry = info.get('industry', None)
        stock_sector = info.get('sector', None)

    except Exception as e:
        errors.append(f"Could not fetch basic info from yfinance for {ticker_symbol}: {e}. Defaults used.")


    try:
        annual_financials = stock.financials
        if not annual_financials.empty:
            if 'Total Revenue' in annual_financials.index and len(annual_financials.columns) >= 4:
                revenue = annual_financials.loc['Total Revenue'].sort_index(ascending=True)
                data['revenue_growth'] = calculate_cagr(revenue.iloc[-4:])
            else: errors.append("Insufficient revenue data from yfinance.")

            annual_earnings = stock.earnings
            if not annual_earnings.empty and 'Basic EPS' in annual_earnings.index and len(annual_earnings.columns) >= 4:
                eps = annual_earnings.loc['Basic EPS'].sort_index(ascending=True)
                data['eps_growth'] = calculate_cagr(eps.iloc[-4:])
            else: errors.append("Insufficient EPS data from yfinance.")

            op_income_idx = 'Operating Income' if 'Operating Income' in annual_financials.index else ('Gross Profit' if 'Gross Profit' in annual_financials.index else None)
            if op_income_idx and 'Total Revenue' in annual_financials.index and not annual_financials.columns.empty:
                op_income = annual_financials.loc[op_income_idx, annual_financials.columns[0]]
                total_revenue = annual_financials.loc['Total Revenue', annual_financials.columns[0]]
                if total_revenue != 0: data['operating_margin'] = op_income / total_revenue
                else: errors.append("Total Revenue is zero for Operating Margin (yfinance).")
            else: errors.append("Missing operating income/total revenue for Operating Margin (yfinance).")

            if 'Operating Income' in annual_financials.index and 'Interest Expense' in annual_financials.index and not annual_financials.columns.empty:
                operating_income = annual_financials.loc['Operating Income', annual_financials.columns[0]]
                interest_expense = annual_financials.loc['Interest Expense', annual_financials.columns[0]]
                if interest_expense != 0: data['interest_coverage_ratio'] = operating_income / interest_expense
                else: data['interest_coverage_ratio'] = 1000.0 # Very high if no interest expense
            else: errors.append("Missing operating income/interest expense for Interest Coverage Ratio (yfinance).")
        else: errors.append("Could not fetch annual financials from yfinance.")

        annual_balance_sheet = stock.balance_sheet
        if not annual_balance_sheet.empty and not annual_balance_sheet.columns.empty:
            if 'Net Income' in annual_financials.index and 'Total Stockholder Equity' in annual_balance_sheet.index:
                net_income_latest = annual_financials.loc['Net Income', annual_financials.columns[0]]
                total_equity_latest = annual_balance_sheet.loc['Total Stockholder Equity', annual_balance_sheet.columns[0]]
                if total_equity_latest != 0: data['roe'] = net_income_latest / total_equity_latest
                else: errors.append("Total Stockholder Equity is zero for ROE (yfinance).")
            else: errors.append("Missing Net Income/Total Stockholder Equity for ROE (yfinance).")

            if 'Total Debt' in annual_balance_sheet.index and 'Total Stockholder Equity' in annual_balance_sheet.index:
                total_debt_latest = annual_balance_sheet.loc['Total Debt', annual_balance_sheet.columns[0]]
                total_equity_latest = annual_balance_sheet.loc['Total Stockholder Equity', annual_balance_sheet.columns[0]]
                if total_equity_latest != 0: data['debt_equity'] = total_debt_latest / total_equity_latest
                else: errors.append("Total Stockholder Equity is zero for Debt/Equity (yfinance).")
            else: errors.append("Missing Total Debt/Total Stockholder Equity for Debt/Equity (yfinance).")
        else: errors.append("Could not fetch balance sheet from yfinance.")

    except Exception as e:
        errors.append(f"An error occurred while processing yfinance fundamentals for {ticker_symbol}: {e}. Defaults used.")

    try:
        hist = stock.history(period="1y", interval="1d")
        if not hist.empty:
            data['rsi'] = calculate_rsi(hist)
            data['ma_cross'] = get_ma_cross_status(hist)
            data['macd_signal'] = calculate_macd(hist)
            data['volume_trend'] = get_volume_trend(hist)
        else: errors.append("Could not fetch historical data from yfinance for technicals.")
    except Exception as e:
        errors.append(f"An error occurred while processing yfinance historical data for {ticker_symbol}: {e}. Defaults used for technicals.")

    # --- Supplementing with other APIs (FMP, EODHD) ---

    # FMP: P/E Ratio (vs Industry)
    # This is complex as FMP's free industry PE API requires specific date and exact industry string,
    # which can be inconsistent with yfinance's industry names.
    # For now, we'll keep this as a default based on current_pe if available, otherwise '~ Industry Avg'.
    if current_pe is not None:
        # A hypothetical mapping or API call for industry average PE would go here.
        # For a truly automated comparison, you'd need a robust industry mapping or
        # an API that directly returns industry average PE for a given ticker's industry.
        # As it's highly variable by industry and harder to auto-fetch accurately with free APIs,
        # we'll keep it simple: assume if PE is low/high, it's relative to average.
        if current_pe < 15: # Arbitrary threshold, requires real industry comparison
            data['pe_vs_industry'] = 'Less than Industry Avg'
        elif current_pe > 30: # Arbitrary threshold
            data['pe_vs_industry'] = '> Industry Avg'
        else:
            data['pe_vs_industry'] = '~ Industry Avg'

    # FMP: Analyst Recommendations
    if FMP_API_KEY:
        data['analyst_recommendations'] = fetch_fmp_analyst_recommendations(ticker_symbol)
    else:
        errors.append("FMP API Key not provided. Analyst Recommendations defaulted.")

    # FMP: Insider Trading
    if FMP_API_KEY:
        data['insider_trading'] = fetch_fmp_insider_trading(ticker_symbol)
    else:
        errors.append("FMP API Key not provided. Insider Trading defaulted.")

    # EODHD: News Sentiment
    if EODHD_API_KEY:
        data['news_sentiment'] = fetch_eodhd_news_sentiment(ticker_symbol)
    else:
        errors.append("EODHD API Key not provided. News Sentiment defaulted.")


    return data, errors

# --- Streamlit UI ---

st.set_page_config(layout="centered", page_title="Stock Scoring Model (Auto-Fetch)")

st.title("📈 Stock Scoring Model (Auto-Fetch)")
st.markdown("Enter a stock ticker symbol, and the model will automatically fetch relevant data from Yahoo Finance and other sources to score the stock.")

st.markdown("""
<div style="padding: 10px; border-radius: 5px; background-color: #f0f2f6; margin-bottom: 20px;">
    <strong>Interpretation:</strong><br>
    Score > 75: <span style="color: green; font-weight: bold;">✅ BUY</span><br>
    Score 50–75: <span style="color: orange; font-weight: bold;">🟡 HOLD</span><br>
    Score < 50: <span style="color: red; font-weight: bold;">❌ SELL</span>
</div>
""", unsafe_allow_html=True)

ticker_symbol = st.text_input("Enter Stock Ticker").upper()
calculate_button = st.button("Calculate Stock Score")

# --- Display Results ---
if calculate_button and ticker_symbol:
    with st.spinner(f"Fetching and calculating score for {ticker_symbol}..."):
        fetched_data, errors = fetch_and_process_stock_data(ticker_symbol)

    if errors:
        st.error(f"⚠️ Issues encountered while fetching data for {ticker_symbol}:")
        for error_msg in errors:
            st.info(f"- {error_msg}")
        st.info("Parameters not fetched were defaulted. Consider reviewing them for accuracy.")
        st.write("---")

    # Use the fetched_data directly as the input for calculation,
    # as it already contains defaults for un-fetchable/missing data.
    result = calculate_stock_score(fetched_data)

    st.markdown("---")
    st.header(f"📊 Stock Score & Recommendation for {ticker_symbol}")

    col_score, col_rec = st.columns(2)
    with col_score:
        st.metric(label="Total Stock Score", value=f"{result['total_score']:.2f}")
    with col_rec:
        if result['recommendation'] == '✅ BUY':
            st.success(f"Recommendation: {result['recommendation']}")
        elif result['recommendation'] == '🟡 HOLD':
            st.warning(f"Recommendation: {result['recommendation']}")
        else:
            st.error(f"Recommendation: {result['recommendation']}")

    st.subheader("Category-wise Scores (out of 100)")
    # Display category scores
    for category, score in result['category_scores'].items():
        st.info(f"**{category}:** {score:.2f}")

    st.markdown("---")
    st.markdown("### Parameters Used for Calculation (Fetched / Defaulted):")
    st.json(fetched_data) # Show the raw data used

    st.markdown("---")
    st.markdown("### Model Structure:")
    st.markdown("""
    **Total Score = $\sum$ (Weight $\times$ Normalized Parameter Score)**
    * **Fundamentals**: 30%
    * **Valuation**: 20%
    * **Market Sentiment**: 10%
    * **Technical Trends**: 10%
    * **Quality of Management**: 10%
    * **Moat/Competitiveness**: 10%
    * **Risk Factors**: 10%
    """)

elif calculate_button and not ticker_symbol:
    st.error("Please enter a stock ticker symbol to calculate the score.")

st.info("💡 **Tip:** This app provides a powerful automated starting point. For truly robust investment decisions, always combine automated scores with your own in-depth qualitative analysis and cross-reference data from multiple reputable sources.")
