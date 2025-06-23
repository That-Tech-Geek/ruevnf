import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests # For making HTTP requests to other APIs

# --- API Keys (Accessed via st.secrets) ---
# Ensure you have .streamlit/secrets.toml configured with your API keys:
# FMP-API-KEY = "your_actual_fmp_api_key_here"
# EODHD-API-KEY = "your_actual_eodhd_api_key_here"

# Accessing secrets using square bracket notation
FMP_API_KEY = st.secrets["FMP-API-KEY"] if "FMP-API-KEY" in st.secrets else ""
EODHD_API_KEY = st.secrets["EODHD-API-KEY"] if "EODHD-API-KEY" in st.secrets else ""


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
    """Scores 50D/200D MA cross status."""
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
    if total_score > 7.5:
        recommendation = '✅ BUY'
    elif 5.0 <= total_score <= 7.5:
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

# FMP API Calls
def fetch_fmp_industry_pe(industry_name, current_pe_value):
    """Attempts to determine PE vs Industry based on provided current_pe and a hypothetical industry average.
    Requires FMP Industry PE API, which is not directly simple for free tier without a date/exchange lookup.
    This function will simply use an arbitrary threshold based on current PE as a substitute for actual comparison."""
    
    # In a real application, you'd fetch industry average PE from FMP
    # Example: https://financialmodelingprep.com/api/v4/industry_price_earning_ratio?date=YYYY-MM-DD&exchange=NYSE&apikey=YOUR_API_KEY
    # This requires dynamic date and accurate industry name mapping.
    # For now, we'll make a simplified assumption.
    # User requested to add this line for reference:
    # https://financialmodelingprep.com/api/v4/industry_price_earning_ratio?date=YYYY-MM-DD&exchange=NYSE&apikey=YOUR_API_KEY
    
    if current_pe_value is None:
        return '~ Industry Avg' # Cannot determine without current PE

    # These thresholds are arbitrary and for demonstration only.
    # A proper implementation would compare against an actual fetched industry average.
    if current_pe_value < 15:
        return 'Less than Industry Avg'
    elif current_pe_value > 30:
        return '> Industry Avg'
    else:
        return '~ Industry Avg'

def fetch_fmp_analyst_recommendations(ticker):
    """Fetches analyst recommendations from FMP API."""
    if not FMP_API_KEY:
        return 'Hold'
    try:
        url = f"https://financialmodelingprep.com/api/v3/analyst-stock-recommendations/{ticker}?limit=1&apikey={FMP_API_KEY}"
        response = requests.get(url, timeout=5) # Add timeout
        response.raise_for_status() # This will raise an HTTPError for 4xx/5xx responses
        data = response.json()
        if data and isinstance(data, list) and len(data) > 0:
            recommendation_score = data[0].get('rating', 'Hold')
            if 'buy' in recommendation_score.lower() and 'strong' in recommendation_score.lower():
                return 'Strong Buy'
            elif 'buy' in recommendation_score.lower():
                return 'Buy'
            elif 'hold' in recommendation_score.lower():
                return 'Hold'
            elif 'sell' in recommendation_score.lower():
                return 'Sell'
        return 'Hold'
    except requests.exceptions.RequestException as e:
        # st.warning(f"FMP Analyst Recommendations API Error: {e}. Defaulting to 'Hold'.")
        # Suppress in-app warnings here, errors will be collected and shown at the end
        pass
    except (KeyError, IndexError, TypeError) as e:
        # st.warning(f"FMP Analyst Recommendations data parsing error: {e}. Defaulting to 'Hold'.")
        pass
    return 'Hold'

def fetch_fmp_insider_trading(ticker):
    """Fetches insider trading data from FMP API and determines trend."""
    if not FMP_API_KEY:
        return 'Net neutral'
    try:
        url = f"https://financialmodelingprep.com/api/v3/insider-trading?symbol={ticker}&limit=20&apikey={FMP_API_KEY}" # Get more data for better trend
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()

        if not data or not isinstance(data, list):
            return 'Net neutral'

        buy_value = 0
        sell_value = 0
        for transaction in data:
            if 'transactionType' in transaction and 'value' in transaction:
                # Value can be negative for sales, take absolute for summation
                if 'buy' in transaction['transactionType'].lower():
                    buy_value += abs(float(transaction['value']))
                elif 'sell' in transaction['transactionType'].lower():
                    sell_value += abs(float(transaction['value']))

        if buy_value > sell_value * 1.5:
            return 'More buying'
        elif sell_value > buy_value * 1.5:
            return 'More selling'
        else:
            return 'Net neutral'
    except requests.exceptions.RequestException as e:
        # st.warning(f"FMP Insider Trading API Error: {e}. Defaulting to 'Net neutral'.")
        pass
    except (KeyError, IndexError, ValueError, TypeError) as e:
        # st.warning(f"FMP Insider Trading data parsing error: {e}. Defaulting to 'Net neutral'.")
        pass
    return 'Net neutral'

# EODHD API Call
def fetch_eodhd_news_sentiment(ticker):
    """Fetches news sentiment score from EODHD API."""
    if not EODHD_API_KEY:
        return 'Neutral'
    try:
        today = pd.to_datetime('today').strftime('%Y-%m-%d')
        ninety_days_ago = (pd.to_datetime('today') - pd.DateOffset(days=90)).strftime('%Y-%m-%d') # More days for better sentiment
        
        # EODHD often requires ticker.EX for US stocks, e.g., AAPL.US
        # Try to append .US if not already present and not explicitly an international ticker
        if '.' not in ticker and not any(char.islower() for char in ticker): # Simple check for common US tickers
            eodhd_ticker = f"{ticker}.US"
        else:
            eodhd_ticker = ticker


        url = f"https://eodhd.com/api/sentiments?s={eodhd_ticker}&from={ninety_days_ago}&to={today}&api_token={EODHD_API_KEY}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()

        if data and eodhd_ticker in data and 'general' in data[eodhd_ticker] and 'sentiment' in data[eodhd_ticker]['general']:
            sentiment_score = data[eodhd_ticker]['general']['sentiment'] # Score from -1 to 1
            if sentiment_score > 0.3:
                return 'Positive'
            elif sentiment_score < -0.3:
                return 'Negative'
            else:
                return 'Neutral'
    except requests.exceptions.RequestException as e:
        # st.warning(f"EODHD News Sentiment API Error: {e}. Defaulting to 'Neutral'.")
        pass
    except (KeyError, IndexError, TypeError) as e:
        # st.warning(f"EODHD News Sentiment data parsing error: {e}. Defaulting to 'Neutral'.")
        pass
    return 'Neutral'


@st.cache_data(ttl=3600) # Cache data for 1 hour to reduce API calls
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
    info = None
    try:
        info = stock.info
        if not info or not info.get('regularMarketPrice'):
            errors.append(f"No valid basic info found for {ticker_symbol} from yfinance. It might be an invalid ticker or delisted.")
            # Do not return immediately, allow other API calls to proceed if possible
        else:
            data['beta'] = info.get('beta', data['beta'])
            data['dividend_yield'] = info.get('dividendYield', data['dividend_yield'])
            data['peg_ratio'] = info.get('pegRatio', data['peg_ratio'])
            data['price_book_ratio'] = info.get('priceToBook', data['price_book_ratio'])
            current_pe = info.get('trailingPE', None)
            data['pe_vs_industry'] = fetch_fmp_industry_pe(info.get('industry'), current_pe)

    except Exception as e:
        errors.append(f"Could not fetch basic info from yfinance for {ticker_symbol}: {e}. Defaults used.")


    annual_financials = pd.DataFrame() # Initialize as empty DataFrame
    try:
        # Check if info was successfully fetched to get currency and period
        if info and info.get('financialCurrency'): # Check for info and if it contains expected key
            annual_financials = stock.financials
            if annual_financials is None or annual_financials.empty:
                errors.append("Could not fetch annual financials from yfinance or it's empty.")
            else:
                # Ensure column index is date-like for sorting
                annual_financials.columns = pd.to_datetime(annual_financials.columns)

                if 'Total Revenue' in annual_financials.index and len(annual_financials.columns) >= 4: # Needs 4 years for 3Y CAGR
                    revenue = annual_financials.loc['Total Revenue'].sort_index(ascending=True) # Sort by date index
                    data['revenue_growth'] = calculate_cagr(revenue.iloc[-4:])
                else: errors.append("Insufficient 'Total Revenue' data from yfinance for 3Y CAGR.")

                # Fetch and process EPS growth
                annual_earnings = stock.earnings
                if annual_earnings is None or annual_earnings.empty:
                    errors.append("Could not fetch annual earnings from yfinance or it's empty.")
                else:
                    annual_earnings.columns = pd.to_datetime(annual_earnings.columns)
                    if 'Basic EPS' in annual_earnings.index and len(annual_earnings.columns) >= 4:
                        eps = annual_earnings.loc['Basic EPS'].sort_index(ascending=True)
                        data['eps_growth'] = calculate_cagr(eps.iloc[-4:])
                    else: errors.append("Insufficient 'Basic EPS' data from yfinance for 3Y CAGR.")

                op_income_idx = 'Operating Income' if 'Operating Income' in annual_financials.index else ('Gross Profit' if 'Gross Profit' in annual_financials.index else None)
                if op_income_idx and 'Total Revenue' in annual_financials.index and not annual_financials.columns.empty:
                    op_income = annual_financials.loc[op_income_idx, annual_financials.columns[0]]
                    total_revenue = annual_financials.loc['Total Revenue', annual_financials.columns[0]]
                    if pd.notna(total_revenue) and total_revenue != 0: data['operating_margin'] = op_income / total_revenue
                    else: errors.append("Total Revenue is zero or NaN for Operating Margin (yfinance).")
                else: errors.append("Missing operating income/total revenue for Operating Margin (yfinance).")

                if 'Operating Income' in annual_financials.index and 'Interest Expense' in annual_financials.index and not annual_financials.columns.empty:
                    operating_income = annual_financials.loc['Operating Income', annual_financials.columns[0]]
                    interest_expense = annual_financials.loc['Interest Expense', annual_financials.columns[0]]
                    if pd.notna(interest_expense) and interest_expense != 0: data['interest_coverage_ratio'] = operating_income / interest_expense
                    else: data['interest_coverage_ratio'] = 1000.0 # Very high if no interest expense
                else: errors.append("Missing operating income/interest expense for Interest Coverage Ratio (yfinance).")

        else:
            errors.append("Basic info not fetched, or financial currency missing, so financials cannot be fetched accurately from yfinance.")

        annual_balance_sheet = stock.balance_sheet # This might also return None
        if annual_balance_sheet is None or annual_balance_sheet.empty:
            errors.append("Could not fetch balance sheet from yfinance or it's empty.")
        else:
            annual_balance_sheet.columns = pd.to_datetime(annual_balance_sheet.columns) # Ensure column index is date-like

            if 'Net Income' in annual_financials.index and 'Total Stockholder Equity' in annual_balance_sheet.index and not annual_financials.columns.empty and not annual_balance_sheet.columns.empty:
                net_income_latest = annual_financials.loc['Net Income', annual_financials.columns[0]]
                total_equity_latest = annual_balance_sheet.loc['Total Stockholder Equity', annual_balance_sheet.columns[0]]
                if pd.notna(total_equity_latest) and total_equity_latest != 0: data['roe'] = net_income_latest / total_equity_latest
                else: errors.append("Total Stockholder Equity is zero or NaN for ROE (yfinance).")
            else: errors.append("Missing Net Income/Total Stockholder Equity for ROE (yfinance).")

            if 'Total Debt' in annual_balance_sheet.index and 'Total Stockholder Equity' in annual_balance_sheet.index and not annual_balance_sheet.columns.empty:
                total_debt_latest = annual_balance_sheet.loc['Total Debt', annual_balance_sheet.columns[0]]
                total_equity_latest = annual_balance_sheet.loc['Total Stockholder Equity', annual_balance_sheet.columns[0]]
                if pd.notna(total_equity_latest) and total_equity_latest != 0: data['debt_equity'] = total_debt_latest / total_equity_latest
                else: errors.append("Total Stockholder Equity is zero or NaN for Debt/Equity (yfinance).")
            else: errors.append("Missing Total Debt/Total Stockholder Equity for Debt/Equity (yfinance).")
        
    except Exception as e:
        errors.append(f"An error occurred while processing yfinance financials/balance sheet for {ticker_symbol}: {e}. Defaults used.")

    try:
        hist = stock.history(period="1y", interval="1d")
        if hist is None or hist.empty:
            errors.append("Could not fetch historical data from yfinance for technicals or it's empty.")
        else:
            data['rsi'] = calculate_rsi(hist)
            data['ma_cross'] = get_ma_cross_status(hist)
            data['macd_signal'] = calculate_macd(hist)
            data['volume_trend'] = get_volume_trend(hist)
    except Exception as e:
        errors.append(f"An error occurred while processing yfinance historical data for {ticker_symbol}: {e}. Defaults used for technicals.")

    # --- Supplementing with other APIs (FMP, EODHD) ---

    # FMP: Analyst Recommendations
    if FMP_API_KEY:
        analyst_rec = fetch_fmp_analyst_recommendations(ticker_symbol)
        data['analyst_recommendations'] = analyst_rec
        if analyst_rec == 'Hold':
            errors.append("FMP Analyst Recommendations might be unavailable or defaulted (check API key/rate limits or data for ticker).")
    else:
        errors.append("FMP API Key not provided. Analyst Recommendations defaulted.")

    # FMP: Insider Trading
    if FMP_API_KEY:
        insider_trend = fetch_fmp_insider_trading(ticker_symbol)
        data['insider_trading'] = insider_trend
        if insider_trend == 'Net neutral':
            errors.append("FMP Insider Trading might be unavailable or defaulted (check API key/rate limits or data for ticker).")
    else:
        errors.append("FMP API Key not provided. Insider Trading defaulted.")

    # EODHD: News Sentiment
    if EODHD_API_KEY:
        news_sent = fetch_eodhd_news_sentiment(ticker_symbol)
        data['news_sentiment'] = news_sent
        if news_sent == 'Neutral':
            errors.append("EODHD News Sentiment might be unavailable or defaulted (check API key/rate limits or data for ticker).")
    else:
        errors.append("EODHD API Key not provided. News Sentiment defaulted.")

    return data, errors

# --- Streamlit UI ---

st.set_page_config(layout="centered", page_title="Stock Scoring Model (Auto-Fetch)")

st.title("📈 Stock Scoring Model (Auto-Fetch)")
st.markdown("Enter a stock ticker symbol, and the model will automatically fetch relevant data from Yahoo Finance and other sources to score the stock.")

st.warning("⚠️ **Important Limitations & Manual Considerations:**")
st.markdown("""
-   **API Keys:** For full functionality, you need to obtain free API keys from **Financial Modeling Prep (FMP)** and **EOD Historical Data (EODHD)**.
    * **Crucially, configure your `.streamlit/secrets.toml` file with `FMP-API-KEY` and `EODHD-API-KEY`. A `403 Forbidden` error often means the key is invalid or rate limits are hit.**
-   **Qualitative Parameters (Defaulted):** The following parameters are **not** fetched by free APIs and are set to **neutral/average defaults**:
    * Management Quality: Governance Rating, Promoter Holding Trend
    * Moat & Competitiveness: Brand Value / Moat, Market Share Leadership
    * Risk Factors (Non-Beta): Litigation/Political Risk
    * *These require your qualitative judgment or more advanced (often paid) data sources.*
-   **P/E Ratio vs Industry:** The model attempts a simplified comparison. For accurate industry-relative P/E, manual research of the industry average or a dedicated API for this purpose is recommended.
-   **Data Availability:** Data from `yfinance` and other free APIs can be inconsistent or incomplete for some tickers. If a parameter cannot be fetched, its default value will be used, and a warning will be displayed.
-   **Rate Limits:** Free APIs have strict rate limits. Frequent requests may lead to temporary blocking.
""")


ticker_symbol = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, GOOG, TSLA)", value="AAPL").upper()
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
