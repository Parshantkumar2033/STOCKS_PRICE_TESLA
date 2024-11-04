
# Technical Indicators

This document provides an overview of various technical indicators used in stock market analysis, including their calculations and significance.

## 1. Moving Averages

### MA7 (7-day Moving Average)
- **Description**: A simple moving average calculated over the past 7 days. It smooths out short-term fluctuations in stock prices, helping to identify recent trends.
- **Calculation**:  
  \[
  MA7 = \frac{P_1 + P_2 + \ldots + P_7}{7}
  \]
  where \( P \) is the price on each day.

### MA20 (20-day Moving Average)
- **Description**: A simple moving average calculated over a longer 20-day period, providing a broader view of the stock's trend over the medium term.
- **Calculation**:  
  \[
  MA20 = \frac{P_1 + P_2 + \ldots + P_{20}}{20}
  \]

## 2. MACD (Moving Average Convergence Divergence)
- **Description**: A momentum indicator that shows the relationship between two moving averages (typically 12-day and 26-day) of a stock’s price. It is used to spot changes in the strength, direction, momentum, and duration of a trend.
- **Calculation**:  
  \[
  MACD = EMA_{12} - EMA_{26}
  \]
- **Signal Line**: A 9-day EMA of MACD is often plotted to identify buy and sell signals based on MACD crossing over or under the signal line.

## 3. Volatility Measure

### 20SD (20-day Standard Deviation)
- **Description**: A measure of volatility over the past 20 days. A higher standard deviation indicates greater price variability.
- **Calculation**: The standard deviation of prices over a 20-day window, showing how far prices deviate from the average (MA20) over time.

## 4. Bollinger Bands

### Upper Band
- **Description**: The upper Bollinger Band is calculated as the MA20 plus two times the 20-day standard deviation, indicating a level where the stock price might be overbought.
- **Calculation**:  
  \[
  Upper \ Band = MA20 + (2 \times 20SD)
  \]

### Lower Band
- **Description**: The lower Bollinger Band is the MA20 minus two times the 20-day standard deviation, indicating a level where the stock might be oversold.
- **Calculation**:  
  \[
  Lower \ Band = MA20 - (2 \times 20SD)
  \]
- **Note**: Together, the upper and lower Bollinger Bands provide a price range within which the stock typically trades.

## 5. Exponential Moving Average (EMA)
- **Description**: EMA gives more weight to recent prices, making it more responsive to new information than a simple moving average (SMA). Common EMAs include the 12-day and 26-day EMAs used in MACD.
- **Calculation**:  
  \[
  EMA_{today} = (Price_{today} \times Multiplier) + (EMA_{yesterday} \times (1 - Multiplier))
  \]
  where  
  \[
  Multiplier = \frac{2}{Days + 1}
  \]

## 6. Log Momentum
- **Description**: A momentum indicator calculated by taking the natural log of the current price divided by the price at a specified previous interval, often used to capture exponential growth rates.
- **Calculation**:  
  \[
  Log \ Momentum = \log\left(\frac{Price_t}{Price_{t-n}}\right)
  \]
  where \( t \) is the current time, and \( t-n \) is the look-back period.
