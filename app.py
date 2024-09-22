import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Title of the app
st.title("Portfolio Risk Management with Stress Testing and Monte Carlo Simulations")

# Explanation section
st.markdown("""
### Overview:
This app assesses portfolio risk by performing stress testing and running Monte Carlo simulations.
You'll be able to enter the assets in your portfolio and simulate various market conditions.
""")

# User Input: Portfolio Assets and Weights
st.sidebar.header('User Input Parameters')
tickers = st.sidebar.multiselect('Select assets in your portfolio', ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NFLX'], default=['AAPL', 'MSFT'])
weights = st.sidebar.text_input('Enter the weights of each asset (comma-separated)', '0.5, 0.5')
capital = st.sidebar.number_input('Initial Investment ($)', value=10000)

# Convert weights input to a list of floats
weights = [float(w) for w in weights.split(',')]

# Error handling for incorrect inputs
if len(weights) != len(tickers):
    st.error("Please enter the same number of weights as the number of selected assets.")
else:
    st.write(f"Portfolio: {dict(zip(tickers, weights))}")

# Function to get stock data
def get_stock_data(tickers):
    data = yf.download(tickers, period='5y')['Adj Close']
    returns = data.pct_change().dropna()
    return returns

# Load stock data
returns = get_stock_data(tickers)

# Calculate portfolio return
portfolio_return = returns.dot(weights)

# Stress Scenarios: -10%, -20%, -30% drops in asset prices
stress_factors = [-0.10, -0.20, -0.30]
stress_results = {}

st.markdown("### Stress Testing Results")
for stress in stress_factors:
    stressed_returns = returns + stress
    stressed_portfolio_return = stressed_returns.dot(weights)
    stress_results[stress] = stressed_portfolio_return.mean() * capital
    st.write(f"Stress Scenario {stress*100}% drop: Expected Portfolio Value: ${stress_results[stress]:,.2f}")

# Monte Carlo Simulations
st.markdown("### Monte Carlo Simulations")

num_simulations = 1000
num_days = 252  # 1 year of trading days
monte_carlo_simulations = []

# Simulate returns for each day
for _ in range(num_simulations):
    simulated_paths = []
    for t in tickers:
        # Assume normally distributed returns based on historical mean and volatility
        mean_return = returns[t].mean()
        vol = returns[t].std()
        simulated_path = np.random.normal(mean_return, vol, num_days)
        simulated_paths.append(simulated_path)
    
    # Calculate the portfolio returns over the simulation period
    portfolio_simulated = np.sum(np.array(simulated_paths).T.dot(weights), axis=1)
    monte_carlo_simulations.append(np.cumprod(1 + portfolio_simulated) * capital)

# Convert to DataFrame for easy plotting
monte_carlo_simulations = pd.DataFrame(monte_carlo_simulations).T
st.line_chart(monte_carlo_simulations)

# Visualize final portfolio value distribution
final_portfolio_values = monte_carlo_simulations.iloc[-1]
st.write(f"Expected Final Portfolio Value: ${final_portfolio_values.mean():,.2f}")
st.write(f"5% Value-at-Risk (VaR): ${final_portfolio_values.quantile(0.05):,.2f}")
st.write(f"5% Expected Shortfall (ES): ${final_portfolio_values[final_portfolio_values <= final_portfolio_values.quantile(0.05)].mean():,.2f}")

# Plot histogram of final portfolio values
st.markdown("### Final Portfolio Value Distribution (Monte Carlo Simulations)")
fig, ax = plt.subplots()
ax.hist(final_portfolio_values, bins=50, alpha=0.75, color='blue')
ax.axvline(final_portfolio_values.quantile(0.05), color='red', linestyle='dashed', linewidth=2)
ax.set_title("Distribution of Final Portfolio Values")
ax.set_xlabel("Portfolio Value ($)")
ax.set_ylabel("Frequency")
st.pyplot(fig)
