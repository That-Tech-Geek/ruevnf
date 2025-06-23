import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

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
    if total_score > 5:
        recommendation = '✅ BUY'
    elif 3 <= total_score <= 4.999999999:
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

# --- yfinance Data Fetching and Calculation ---

def calculate_cagr(series):
    """Calculates Compound Annual Growth Rate for a series over multiple periods."""
    if len(series) < 2:
        return 0.0
    # Ensure series is sorted by index (date/year) for correct calculation
    series = series.sort_index()
    try:
        start_value = series.iloc[0]
        end_value = series.iloc[-1]
        num_periods = len(series) - 1
        if start_value <= 0: # Avoid division by zero or negative base for CAGR
            return 0.0
        return (end_value / start_value)**(1/num_periods) - 1
    except (ZeroDivisionError, ValueError):
        return 0.0

def calculate_rsi(data, window=14):
    """Calculates the Relative Strength Index (RSI)."""
    if 'Close' not in data.columns or len(data) < window:
        return 50 # Default if insufficient data
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    # Avoid division by zero for RS
    if loss.iloc[-1] == 0:
        rs = np.inf if gain.iloc[-1] > 0 else 0
    else:
        rs = gain / loss

    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty and not pd.isna(rsi.iloc[-1]) else 50 # Default to 50 if calculation fails

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """Calculates MACD and MACD Signal."""
    if 'Close' not in data.columns or len(data) < max(fast_period, slow_period, signal_period):
        return 'Neutral' # Default if insufficient data

    ema_fast = data['Close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal_period, adjust=False).mean()

    # Ensure there are enough data points for crossover logic
    if len(macd) < 2 or len(signal_line) < 2:
        return 'Neutral'

    # Check for crossover in the most recent two points
    if macd.iloc[-2] < signal_line.iloc[-2] and macd.iloc[-1] > signal_line.iloc[-1]:
        return 'Bullish crossover'
    elif macd.iloc[-2] > signal_line.iloc[-2] and macd.iloc[-1] < signal_line.iloc[-1]:
        return 'Bearish crossover'
    else:
        return 'Neutral' # No clear recent crossover or already crossed

def get_ma_cross_status(data, short_window=50, long_window=200):
    """Determines 50D/200D MA cross status."""
    if 'Close' not in data.columns or len(data) < long_window:
        return 'Flat' # Not enough data for long SMA

    data[f'SMA_{short_window}'] = data['Close'].rolling(window=short_window).mean()
    data[f'SMA_{long_window}'] = data['Close'].rolling(window=long_window).mean()

    # Ensure SMAs are not NaN at the required positions
    if pd.isna(data[f'SMA_{long_window}'].iloc[-1]) or pd.isna(data[f'SMA_{long_window}'].iloc[-2]):
        return 'Flat' # Cannot determine if MA values are missing

    # Check the most recent values
    short_ma = data[f'SMA_{short_window}'].iloc[-1]
    long_ma = data[f'SMA_{long_window}'].iloc[-1]

    # Check previous values for crossover detection
    short_ma_prev = data[f'SMA_{short_window}'].iloc[-2]
    long_ma_prev = data[f'SMA_{long_window}'].iloc[-2]

    if short_ma > long_ma and short_ma_prev <= long_ma_prev:
        return 'Golden Cross'
    elif short_ma < long_ma and short_ma_prev >= long_ma_prev:
        return 'Death Cross'
    else:
        return 'Flat' # No recent clear cross or already crossed

def get_volume_trend(data, short_period=30, long_period=90):
    """Determines volume trend based on recent and historical average volume."""
    if 'Volume' not in data.columns or 'Close' not in data.columns or len(data) < long_period:
        return 'Flat' # Not enough data

    recent_avg_volume = data['Volume'].iloc[-short_period:].mean()
    long_term_avg_volume = data['Volume'].iloc[-long_period:].mean()

    recent_avg_price = data['Close'].iloc[-short_period:].mean()
    long_term_avg_price = data['Close'].iloc[-long_period:].mean()

    # Compare recent average with a slightly older average to detect trend
    if recent_avg_volume > 1.2 * long_term_avg_volume and recent_avg_price > long_term_avg_price:
        return 'Rising with price'
    elif recent_avg_volume < 0.8 * long_term_avg_volume and recent_avg_price < long_term_avg_price:
        return 'Falling with price'
    else:
        return 'Flat'

def fetch_and_process_stock_data(ticker_symbol):
    """Fetches and processes stock data using yfinance."""
    stock = yf.Ticker(ticker_symbol)
    data = {}
    errors = []

    # Initialize all data points with defaults that represent 'neutral' or 'cannot determine'
    data = {
        'revenue_growth': 0.0,
        'eps_growth': 0.0,
        'roe': 0.0,
        'debt_equity': 0.0,
        'interest_coverage_ratio': 0.0,
        'operating_margin': 0.0,
        'pe_vs_industry': '~ Industry Avg', # yfinance doesn't provide industry PE comparison
        'peg_ratio': 2.0,
        'price_book_ratio': 3.0,
        'dividend_yield': 0.0,
        'insider_trading': 'Net neutral', # Qualitative, default
        'news_sentiment': 'Neutral', # Qualitative, default
        'analyst_recommendations': 'Hold', # Qualitative, default
        'rsi': 50, # Default neutral RSI
        'ma_cross': 'Flat', # Default neutral MA cross
        'macd_signal': 'Neutral', # Default neutral MACD
        'volume_trend': 'Flat', # Default neutral volume trend
        'governance_rating': 'Average', # Qualitative, default
        'promoter_holding_trend': 'Stable', # Qualitative, default
        'brand_value': 'Moderate', # Qualitative, default
        'market_share': 'Minor player', # Qualitative, default
        'beta': 1.0, # Default beta to market average
        'litigation_risk': 'None' # Qualitative, default
    }

    # Try fetching info
    try:
        info = stock.info
        if not info:
            errors.append(f"No info found for {ticker_symbol}. It might be an invalid ticker.")
            return data, errors # Return defaults immediately

        data['beta'] = info.get('beta', data['beta'])
        data['dividend_yield'] = info.get('dividendYield', data['dividend_yield'])
        data['peg_ratio'] = info.get('pegRatio', data['peg_ratio'])
        data['price_book_ratio'] = info.get('priceToBook', data['price_book_ratio'])
        data['current_pe'] = info.get('trailingPE', 15.0) # Store current PE for user info

    except Exception as e:
        errors.append(f"Could not fetch basic info for {ticker_symbol}: {e}. Defaults used.")


    # Try fetching financials for fundamentals
    try:
        # Get annual financials for CAGR calculation
        annual_financials = stock.financials
        if not annual_financials.empty:
            # Revenue Growth (3Y CAGR)
            if 'Total Revenue' in annual_financials.index and len(annual_financials.columns) >= 4: # Need at least 4 years for 3Y CAGR
                revenue = annual_financials.loc['Total Revenue'].sort_index(ascending=True)
                data['revenue_growth'] = calculate_cagr(revenue.iloc[-4:])
            else:
                errors.append("Insufficient revenue data for 3Y CAGR.")

            # EPS Growth (3Y)
            annual_earnings = stock.earnings
            if not annual_earnings.empty and 'Basic EPS' in annual_earnings.index and len(annual_earnings.columns) >= 4:
                eps = annual_earnings.loc['Basic EPS'].sort_index(ascending=True)
                data['eps_growth'] = calculate_cagr(eps.iloc[-4:])
            else:
                errors.append("Insufficient EPS data for 3Y CAGR.")

            # Operating Margin (latest)
            op_income_idx = 'Operating Income' if 'Operating Income' in annual_financials.index else ('Gross Profit' if 'Gross Profit' in annual_financials.index else None) # Fallback
            if op_income_idx and 'Total Revenue' in annual_financials.index:
                op_income = annual_financials.loc[op_income_idx, annual_financials.columns[0]]
                total_revenue = annual_financials.loc['Total Revenue', annual_financials.columns[0]]
                if total_revenue != 0:
                    data['operating_margin'] = op_income / total_revenue
                else: errors.append("Total Revenue is zero, cannot calculate Operating Margin.")
            else: errors.append("Missing operating income or total revenue for Operating Margin.")


            # Interest Coverage Ratio (latest)
            if 'Operating Income' in annual_financials.index and 'Interest Expense' in annual_financials.index:
                operating_income = annual_financials.loc['Operating Income', annual_financials.columns[0]]
                interest_expense = annual_financials.loc['Interest Expense', annual_financials.columns[0]]
                if interest_expense != 0:
                    data['interest_coverage_ratio'] = operating_income / interest_expense
                else:
                    data['interest_coverage_ratio'] = 1000.0 # Very high if no interest expense
            else: errors.append("Missing operating income or interest expense for Interest Coverage Ratio.")

        else:
            errors.append("Could not fetch sufficient annual financials for fundamental metrics.")

        # ROE and Debt/Equity from balance sheet
        annual_balance_sheet = stock.balance_sheet
        if not annual_balance_sheet.empty:
            # ROE (latest)
            if 'Net Income' in annual_financials.index and 'Total Stockholder Equity' in annual_balance_sheet.index:
                net_income_latest = annual_financials.loc['Net Income', annual_financials.columns[0]]
                total_equity_latest = annual_balance_sheet.loc['Total Stockholder Equity', annual_balance_sheet.columns[0]]
                if total_equity_latest != 0:
                    data['roe'] = net_income_latest / total_equity_latest
                else: errors.append("Total Stockholder Equity is zero, cannot calculate ROE.")
            else: errors.append("Missing Net Income or Total Stockholder Equity for ROE.")


            # Debt/Equity Ratio (latest)
            if 'Total Debt' in annual_balance_sheet.index and 'Total Stockholder Equity' in annual_balance_sheet.index:
                total_debt_latest = annual_balance_sheet.loc['Total Debt', annual_balance_sheet.columns[0]]
                total_equity_latest = annual_balance_sheet.loc['Total Stockholder Equity', annual_balance_sheet.columns[0]]
                if total_equity_latest != 0:
                    data['debt_equity'] = total_debt_latest / total_equity_latest
                else: errors.append("Total Stockholder Equity is zero, cannot calculate Debt/Equity.")
            else: errors.append("Missing Total Debt or Total Stockholder Equity for Debt/Equity.")

        else:
            errors.append("Could not fetch balance sheet for ROE and Debt/Equity.")

    except Exception as e:
        errors.append(f"An error occurred while fetching fundamental financials for {ticker_symbol}: {e}. Defaults used.")

    # Try fetching historical data for technicals
    try:
        hist = stock.history(period="1y", interval="1d") # Get 1 year of daily historical data
        if not hist.empty:
            data['rsi'] = calculate_rsi(hist)
            data['ma_cross'] = get_ma_cross_status(hist)
            data['macd_signal'] = calculate_macd(hist)
            data['volume_trend'] = get_volume_trend(hist)
        else:
            errors.append("Could not fetch historical data for technical indicators.")
    except Exception as e:
        errors.append(f"An error occurred while fetching historical data for {ticker_symbol}: {e}. Defaults used for technicals.")

    return data, errors

# --- Streamlit UI ---

st.set_page_config(layout="centered", page_title="Stock Scoring Model (Auto-Fetch)")

st.title("📈 Stock Scoring Model (Auto-Fetch)")
st.markdown("Enter a stock ticker symbol, and the model will automatically fetch relevant data from Yahoo Finance to score the stock.")

st.markdown("""
<div style="padding: 10px; border-radius: 5px; background-color: #f0f2f6; margin-bottom: 20px;">
    <strong>Interpretation:</strong><br>
    Score > 75: <span style="color: green; font-weight: bold;">✅ BUY</span><br>
    Score 50–75: <span style="color: orange; font-weight: bold;">🟡 HOLD</span><br>
    Score < 50: <span style="color: red; font-weight: bold;">❌ SELL</span>
</div>
""", unsafe_allow_html=True)

st.warning("⚠️ **Important Limitations:**")
st.markdown("""
- **Qualitative parameters** (e.g., Insider Trading, News Sentiment, Analyst Recommendations, Management Quality, Moat/Competitiveness, Litigation/Political Risk) cannot be reliably fetched from `yfinance` and are set to **neutral/average defaults**.
- **P/E Ratio vs Industry** is also defaulted to 'Industry Avg' as `yfinance` does not provide direct industry comparisons. You may need external research for this.
- Data availability varies for different tickers. If data is missing from `yfinance`, default values will be used for those specific parameters.
""")


ticker_symbol = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, GOOG, TSLA)", value="AAPL").upper()
calculate_button = st.button("Calculate Stock Score")

# --- Display Results ---
if calculate_button and ticker_symbol:
    with st.spinner(f"Fetching and calculating score for {ticker_symbol}..."):
        fetched_data, errors = fetch_and_process_stock_data(ticker_symbol)

    if errors:
        st.error(f"Errors occurred while fetching data for {ticker_symbol}. Default values will be used for problematic parameters.")
        for error_msg in errors:
            st.info(f"Issue: {error_msg}")
        st.write("---") # Separator if there are errors before showing results

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
    # Display the actual values used for scoring (fetched or defaulted)
    st.markdown("**Parameters Used for Calculation (Fetched / Defaulted):**")
    st.json(fetched_data) # Show the raw data used

    # Display category scores
    for category, score in result['category_scores'].items():
        st.info(f"**{category}:** {score:.2f}")

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

st.info("💡 **Tip:** While this app automates data retrieval, for a comprehensive analysis, consider cross-referencing with other financial data sources for qualitative factors and industry-specific P/E comparisons.")
