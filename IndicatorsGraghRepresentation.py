import numpy as np
import matplotlib.pyplot as plt
import talib
import pandas as pd
import os
from mpl_toolkits.mplot3d import Axes3D

######   data
  
klines_1d_dataset = pd.read_csv('Data/klines/BTCUSDT-1d-2023-01-21.csv', header=None)
klines_1h_dataset = pd.read_csv('Data/klines/BTCUSDT-1h-2023-01-21.csv', header=None)
klines_1m_dataset = pd.read_csv('Data/klines/BTCUSDT-1m-2023-01-21.csv', header=None)
klines_1s_dataset = pd.read_csv('Data/klines/BTCUSDT-1s-2023-01-21.csv', header=None)
klines_2h_dataset = pd.read_csv('Data/klines/BTCUSDT-2h-2023-01-21.csv', header=None)
klines_3m_dataset = pd.read_csv('Data/klines/BTCUSDT-3m-2023-01-21.csv', header=None)
klines_4h_dataset = pd.read_csv('Data/klines/BTCUSDT-4h-2023-01-21.csv', header=None)
klines_5m_dataset = pd.read_csv('Data/klines/BTCUSDT-5m-2023-01-21.csv', header=None)
klines_6h_dataset = pd.read_csv('Data/klines/BTCUSDT-6h-2023-01-21.csv', header=None)
klines_12h_dataset = pd.read_csv('Data/klines/BTCUSDT-12h-2023-01-21.csv', header=None)
klines_15m_dataset = pd.read_csv('Data/klines/BTCUSDT-15m-2023-01-21.csv', header=None)
klines_30m_dataset = pd.read_csv('Data/klines/BTCUSDT-30m-2023-01-21.csv', header=None)

trades_dataset = pd.read_csv('Data/BTCUSDT-trades-2023-01-21.csv')
aggTrades_dataset = pd.read_csv('Data/BTCUSDT-aggTrades-2023-01-21.csv', header=None)


###### agroup

p = klines_5m_dataset.iloc[:, 1:5].to_numpy()
prices = np.empty([0])
for i in range(len(p)):
    avg = (p[i][0] + p[i][1] + p[i][2] + p[i][3]) / 4
    #print(avg)
    prices = np.append(prices, avg)
#print(prices)

volume = klines_1m_dataset.iloc[:, 5].to_numpy()
#print(volume)

close_prices = klines_5m_dataset.iloc[:, 4].to_numpy()
volumes = klines_5m_dataset.iloc[:, 5].to_numpy()

obv = np.empty([0])

# Loop through the close prices and volumes to calculate the OBV
for i in range(len(close_prices)):
    if i == 0:
        # For the first value, the OBV is just the volume
        obv = np.append(obv, volumes[i])
    else:
        if close_prices[i] > close_prices[i-1]:
            # If the close price increased, add the volume to the OBV
            obv = np.append(obv, obv[i-1] + volumes[i])
        elif close_prices[i] < close_prices[i-1]:
            # If the close price decreased, subtract the volume from the OBV
            obv = np.append(obv, obv[i-1] - volumes[i])
        else:
            # If the close price did not change, the OBV stays the same
            obv = np.append(obv, obv[i-1])
            
#Accumulation/distribution line
c = klines_5m_dataset.iloc[:, 4].to_numpy()
high = klines_5m_dataset.iloc[:, 2].to_numpy()
low = klines_5m_dataset.iloc[:, 3].to_numpy()
vol = klines_5m_dataset.iloc[:, 5].to_numpy()
ad = np.empty([0])

for i in range(len(c)):
    money_flow_multiplier = ((c[i] - low[i]) - (high[i] - c[i])) / (high[i] - low[i])
    money_flow_volume = money_flow_multiplier * vol[i]
    if i == 0:
        ad = np.append(ad, money_flow_volume)
    else:
        ad = np.append(ad, ad[i-1] + money_flow_volume)
#print(ad)

#Stochastic Oscillator indicator
def stochastic_oscillator(klines_5m_dataset):
    n = 14
    high = klines_5m_dataset.iloc[:, 2].to_numpy()
    low = klines_5m_dataset.iloc[:, 3].to_numpy()
    close = klines_5m_dataset.iloc[:, 4].to_numpy()
    stochastic_oscillator = np.empty([0])
    for i in range(len(close) - n + 1):
        highest_high = np.max(high[i:i + n])
        lowest_low = np.min(low[i:i + n])
        stochastic_oscillator = np.append(stochastic_oscillator, (close[i + n - 1] - lowest_low) / (highest_high - lowest_low) * 100)
    return stochastic_oscillator

stochastic_oscillator_indicator = stochastic_oscillator(klines_5m_dataset)

#Rate of Change indicator
# Get the close prices from the dataset
close_prices = klines_5m_dataset.iloc[:, 4].to_numpy()

# Calculate the ROC using the following formula:
# ROC = ((current close price / close price n periods ago) - 1) * 100
n = 9 # number of periods to look back
ROC = np.empty([0])
for i in range(n, len(close_prices)):
    ROC = np.append(ROC, ((close_prices[i]/close_prices[i-n]) - 1) * 100)
    
#supertrend indicator
def calc_supertrend(klines_5m_dataset, atr_period=14, multiplier=3):
    # Calculate the average true range (ATR)
    high = klines_5m_dataset.iloc[:, 2].to_numpy()
    low = klines_5m_dataset.iloc[:, 3].to_numpy()
    close = klines_5m_dataset.iloc[:, 4].to_numpy()

    tr = np.zeros(len(close))
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(close[i-1] - low[i]))
    
    atr = np.zeros(len(close))
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(close)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period

    # Calculate the Supertrend indicator
    supertrend = np.zeros(len(close))
    for i in range(atr_period-1, len(close)):
        basic_upper_band = close[i] + multiplier * atr[i]
        basic_lower_band = close[i] - multiplier * atr[i]
        if i == atr_period-1:
            supertrend[i] = basic_upper_band if close[i] <= basic_upper_band else basic_lower_band
        else:
            supertrend[i] = basic_upper_band if supertrend[i-1] <= basic_upper_band else basic_lower_band
            supertrend[i] = basic_lower_band if supertrend[i-1] >= basic_lower_band else supertrend[i]

    return supertrend

# Calculate the Supertrend indicator for the given data
supertrend = calc_supertrend(klines_5m_dataset)
    
######   functions


def moving_average(x, n, type='simple'):
	"""
	compute an n period moving average.

	type is 'simple' | 'exponential'

	"""
	x = np.asarray(x)
	if type == 'simple':
		weights = np.ones(n)
	else:
		weights = np.exp(np.linspace(-1., 0., n))

	weights /= weights.sum()

	a = np.convolve(x, weights, mode='full')[:len(x)]
	a[:n] = a[n]
	return a


def relative_strength(prices, n=14):
	"""
	compute the n period relative strength indicator
	http://stockcharts.com/school/doku.php?id=chart_school:glossary_r#relativestrengthindex
	http://www.investopedia.com/terms/r/rsi.asp
	"""

	deltas = np.diff(prices)
	seed = deltas[:n+1]
	up = seed[seed >= 0].sum()/n
	down = -seed[seed < 0].sum()/n
	rs = up/down
	rsi = np.zeros_like(prices)
	rsi[:n] = 100. - 100./(1. + rs)

	for i in range(n, len(prices)):
		delta = deltas[i - 1]  # cause the diff is 1 shorter

		if delta > 0:
			upval = delta
			downval = 0.
		else:
			upval = 0.
			downval = -delta

		up = (up*(n - 1) + upval)/n
		down = (down*(n - 1) + downval)/n

		rs = up/down
		rsi[i] = 100. - 100./(1. + rs)

	return rsi

'''
def moving_average_convergence(x, nslow=26, nfast=12):
	"""
	compute the MACD (Moving Average Convergence/Divergence) using a fast and slow exponential moving avg'
	return value is emaslow, emafast, macd which are len(x) arrays
	"""
	emaslow = moving_average(x, nslow, type='exponential')
	emafast = moving_average(x, nfast, type='exponential')
	return emaslow, emafast, emafast - emaslow
'''

######   code


#nslow = 26
#nfast = 12
nema = 9
#emaslow, emafast, macd = moving_average_convergence(prices, nslow=nslow, nfast=nfast)
macd, macdsignal, macdhist = talib.MACD(prices, fastperiod=12, slowperiod=26, signalperiod=9)
ema9 = moving_average(macd, nema, type='exponential')
rsi = relative_strength(prices)

#BOLL
closed = klines_5m_dataset.iloc[:, 4].to_numpy()
n = 21  # Number of periods for the moving average
std = np.std(closed)  # Standard deviation of closing prices

# Calculate moving average
moving_avg = np.convolve(closed, np.ones(n)/n, mode='valid')

# Calculate upper and lower Bollinger Bands
upper_band = moving_avg + 2 * std
lower_band = moving_avg - 2 * std

wins = 400  # Number of data points to display on the graph

plt.figure(1)

### prices

plt.subplot2grid((8, 1), (0, 0), rowspan = 4)
plt.plot(upper_band[-wins:], 'red', lw=1)
plt.plot(lower_band[-wins:], 'red', lw=1)
plt.plot(moving_avg[-wins:], 'yellow', lw=1)
plt.plot(prices[-wins:], 'k', lw = 1)

### rsi

plt.subplot2grid((8, 1), (5, 0))
plt.plot(rsi[-wins:], color='black', lw=1)
plt.axhline(y=30,     color='red',   linestyle='-')
plt.axhline(y=70,     color='blue',  linestyle='-')


## MACD

plt.subplot2grid((8, 1), (6, 0))

plt.plot(ema9[-wins:], 'red', lw=1)
plt.plot(macd[-wins:], 'blue', lw=1)


plt.subplot2grid((8, 1), (7, 0))

plt.plot(macd[-wins:]-ema9[-wins:], 'k', lw = 2)
plt.axhline(y=0, color='b', linestyle='-')

plt.show()

bar_graph = plt.plot(volume, color="blue")

plt.show()

plt.plot(obv)
plt.show()

plt.plot(ad)
plt.show()

plt.plot(stochastic_oscillator_indicator)
plt.xlabel("time")
plt.ylabel("Stochastic Oscillator")
plt.show()

# Plot the ROC indicator on a chart
wins = 80
plt.plot(ROC[-wins:], 'k', lw = 1)
plt.title("Rate of Change (ROC) Indicator")
plt.xlabel("Time")
plt.ylabel("ROC Value")
plt.show()

#plot the supertrend indicator on a chart
wins = 80
plt.figure(1)
plt.plot(klines_5m_dataset.iloc[:, 4][-wins:], 'k', lw = 1)
plt.plot(supertrend[-wins:], 'red', lw=1)
plt.show()


# Plot the trades made by the buyer in blue and the trades made by the maker in red
plt.plot(aggTrades_dataset[aggTrades_dataset.iloc[:, 6] == True].iloc[:, 1], aggTrades_dataset[aggTrades_dataset.iloc[:, 6] == True].iloc[:, 2], marker="o", linestyle="", color="blue", label="Buyer Trades")
plt.plot(aggTrades_dataset[aggTrades_dataset.iloc[:, 6] == False].iloc[:, 1], aggTrades_dataset[aggTrades_dataset.iloc[:, 6] == False].iloc[:, 2], marker="o", linestyle="", color="red", label="Maker Trades")

'''# Plot the best price matches as circles and the other trades as squares
plt.plot(aggTrades_dataset[aggTrades_dataset.iloc[:, 7] == True].iloc[:, 1], aggTrades_dataset[aggTrades_dataset.iloc[:, 7] == True].iloc[:, 2], marker="o", linestyle="", color="green", label="Best Price Matches")
plt.plot(aggTrades_dataset[aggTrades_dataset.iloc[:, 7] == False].iloc[:, 1], aggTrades_dataset[aggTrades_dataset.iloc[:, 7] == False].iloc[:, 2], marker="s", linestyle="", color="purple", label="Other Trades")'''

# Add a legend to the chart
plt.legend()

# Show the chart
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(aggTrades_dataset.iloc[:, 5], aggTrades_dataset.iloc[:, 1], aggTrades_dataset.iloc[:, 2])
ax.set_xlabel('Timestamp')
ax.set_ylabel('Price')
ax.set_zlabel('Quantity')

# Show the plot
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(trades_dataset.iloc[:, 4], trades_dataset.iloc[:, 1], trades_dataset.iloc[:, 2])
ax.set_xlabel('Time')
ax.set_ylabel('Price')
ax.set_zlabel('Quantity')

# Show the plot
plt.show()

#introduzir todos os indicadores em todos os timeframes
#criar grafico que represente os dados dos cvs trades e aggTrades
