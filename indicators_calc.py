import sys
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import talib


klines = pd.read_csv(sys.argv[1], header=None)

def getHigh(klines):
    return klines.iloc[:, 2].to_numpy()
    
def getLow(klines):
    return klines.iloc[:, 3].to_numpy()

def getOpen(klines):
    return klines.iloc[:, 1].to_numpy()

def getClose(klines):
    return klines.iloc[:, 4].to_numpy()

def price(klines):
    return klines.iloc[:, 1].to_numpy()

def Accumulation_distribution(klines):
    c = klines.iloc[:, 4].to_numpy()
    high = klines.iloc[:, 2].to_numpy()
    low = klines.iloc[:, 3].to_numpy()
    vol = klines.iloc[:, 5].to_numpy()
    ad = np.empty([0])

    for i in range(len(c)):
        money_flow_multiplier = ((c[i] - low[i]) - (high[i] - c[i])) / (high[i] - low[i])
        money_flow_volume = money_flow_multiplier * vol[i]
        if i == 0:
            ad = np.append(ad, money_flow_volume)
        else:
            ad = np.append(ad, ad[i-1] + money_flow_volume)
    return ad
    
def volume(klines):
    volume = klines.iloc[:, 5].to_numpy()
    return volume
    
def stochastic_oscillator(klines):
    n = 14
    high = klines.iloc[:, 2].to_numpy()
    low = klines.iloc[:, 3].to_numpy()
    close = klines.iloc[:, 4].to_numpy()
    stochastic_oscillator = np.empty([0])
    for i in range(len(close) - n + 1):
        highest_high = np.max(high[i:i + n])
        lowest_low = np.min(low[i:i + n])
        stochastic_oscillator = np.append(stochastic_oscillator, (close[i + n - 1] - lowest_low) / (highest_high - lowest_low) * 100)
    return stochastic_oscillator

def rate_of_change(klines):
    # Get the close prices from the dataset
    close_prices = klines.iloc[:, 4].to_numpy()

    # Calculate the ROC using the following formula:
    # ROC = ((current close price / close price n periods ago) - 1) * 100
    n = 9 # number of periods to look back
    ROC = np.empty([0])
    for i in range(n, len(close_prices)):
        ROC = np.append(ROC, ((close_prices[i]/close_prices[i-n]) - 1) * 100)
    return ROC
    
def calc_supertrend(klines, atr_period=14, multiplier=3):
    # Calculate the average true range (ATR)
    high = klines.iloc[:, 2].to_numpy()
    low = klines.iloc[:, 3].to_numpy()
    close = klines.iloc[:, 4].to_numpy()

    tr = np.zeros(len(close))
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(close[i-1] - low[i]))
    
    atr = np.zeros(len(close))
    try:
        atr[atr_period-1] = np.mean(tr[:atr_period])
    except:
        pass
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

def relative_strength(klines, n=14):
    """
    compute the n period relative strength indicator
    http://stockcharts.com/school/doku.php?id=chart_school:glossary_r#relativestrengthindex
    http://www.investopedia.com/terms/r/rsi.asp
    """
    p = klines.iloc[:, 1:5].to_numpy()
    prices = np.empty([0])
    for i in range(len(p)):
        avg = (p[i][0] + p[i][1] + p[i][2] + p[i][3]) / 4
        #print(avg)
        prices = np.append(prices, avg)
    #print(prices)

    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed >= 0].sum()/n
    down = -seed[seed < 0].sum()/n
    try:
        rs = up/down
    except:
        pass
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

def onBalance_volume(klines):
    close_prices = klines.iloc[:, 4].to_numpy()
    volumes = klines.iloc[:, 5].to_numpy()

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
    return obv
    
def macd(klines):
    p = klines.iloc[:, 1:5].to_numpy()
    prices = np.empty([0])
    for i in range(len(p)):
        avg = (p[i][0] + p[i][1] + p[i][2] + p[i][3]) / 4
        prices = np.append(prices, avg)
    macd, macdsignal, macdhist = talib.MACD(prices, fastperiod=12, slowperiod=26, signalperiod=9)
    return macd

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
    try:
        a[:n] = a[n]
    except:
        pass
    return a
    
def bool(klines, period=21):
    closed = klines.iloc[:, 4].to_numpy()
    n = period  # Number of periods for the moving average
    std = np.std(closed)  # Standard deviation of closing prices

    # Calculate moving average
    try:
        moving_avg = np.convolve(closed, np.ones(n)/n, mode='valid')
    except:
        pass
        
    # Calculate upper and lower Bollinger Bands
    upper_band = moving_avg + 2 * std
    lower_band = moving_avg - 2 * std
    return moving_avg, upper_band, lower_band

high = getHigh(klines)
low = getLow(klines)
close = getClose(klines)
open = getOpen(klines)
ad = Accumulation_distribution(klines)
vol = volume(klines)
so = stochastic_oscillator(klines)
roc = rate_of_change(klines)
sp = calc_supertrend(klines)
rsi = relative_strength(klines)
obv = onBalance_volume(klines)
macd = macd(klines)
ema9 = moving_average(macd, 9,  type='exponential')
ema21 = moving_average(macd, 21, type='exponential')
ema50 = moving_average(macd, 50,  type='exponential')
ema100 = moving_average(macd, 100,  type='exponential')
ema200 = moving_average(macd, 200,  type='exponential')
moving_avg, upper_band, lower_band = bool(klines)

print(len(high))
print(len(low))
print(len(close))
print(len(open))
print(len(ad))
print(len(vol))
print(len(so))
print(len(roc))
print(len(sp))
print(len(rsi))
print(len(obv))
print(len(macd))
print(len(ema9))
print(len(ema21))
print(len(ema50))
print(len(ema100))
print(len(ema200))
print(len(moving_avg))
print(len(upper_band))
print(len(lower_band))

max_length = max(high.shape[0], low.shape[0], close.shape[0], open.shape[0], ad.shape[0], vol.shape[0], so.shape[0], roc.shape[0], sp.shape[0], rsi.shape[0], obv.shape[0], macd.shape[0], ema9.shape[0], ema21.shape[0], ema50.shape[0], ema100.shape[0], ema200.shape[0], moving_avg.shape[0], upper_band.shape[0], lower_band.shape[0])

num_nans = max_length - len(high)
high = np.pad(high, (num_nans, 0), mode='constant', constant_values=(np.nan,))
num_nans = max_length - len(low)
low = np.pad(low, (num_nans, 0), mode='constant', constant_values=(np.nan,))
num_nans = max_length - len(close)
close = np.pad(close, (num_nans, 0), mode='constant', constant_values=(np.nan,))
num_nans = max_length - len(open)
open = np.pad(open, (num_nans, 0), mode='constant', constant_values=(np.nan,))
num_nans = max_length - len(ad)
ad = np.pad(ad, (num_nans, 0), mode='constant', constant_values=(np.nan,))
num_nans = max_length - len(vol)
vol = np.pad(vol, (num_nans, 0), mode='constant', constant_values=(np.nan,))
num_nans = max_length - len(so)
so = np.pad(so, (num_nans, 0), mode='constant', constant_values=(np.nan,))
num_nans = max_length - len(roc)
roc = np.pad(roc, (num_nans, 0), mode='constant', constant_values=(np.nan,))
num_nans = max_length - len(sp)
sp = np.pad(sp, (num_nans, 0), mode='constant', constant_values=(np.nan,))
num_nans = max_length - len(rsi)
rsi = np.pad(rsi, (num_nans, 0), mode='constant', constant_values=(np.nan,))
num_nans = max_length - len(obv)
obv = np.pad(obv, (num_nans, 0), mode='constant', constant_values=(np.nan,))
num_nans = max_length - len(macd)
macd = np.pad(macd, (num_nans, 0), mode='constant', constant_values=(np.nan,))
num_nans = max_length - len(ema9)
ema9 = np.pad(ema9, (num_nans, 0), mode='constant', constant_values=(np.nan,))
num_nans = max_length - len(ema21)
ema21 = np.pad(ema21, (num_nans, 0), mode='constant', constant_values=(np.nan,))
num_nans = max_length - len(ema50)
ema50 = np.pad(ema50, (num_nans, 0), mode='constant', constant_values=(np.nan,))
num_nans = max_length - len(ema100)
ema100 = np.pad(ema100, (num_nans, 0), mode='constant', constant_values=(np.nan,))
num_nans = max_length - len(ema200)
ema200 = np.pad(ema200, (num_nans, 0), mode='constant', constant_values=(np.nan,))
num_nans = max_length - len(moving_avg)
moving_avg = np.pad(moving_avg, (num_nans, 0), mode='constant', constant_values=(np.nan,))
num_nans = max_length - len(upper_band)
upper_band = np.pad(upper_band, (num_nans, 0), mode='constant', constant_values=(np.nan,))
num_nans = max_length - len(lower_band)
lower_band = np.pad(lower_band, (num_nans, 0), mode='constant', constant_values=(np.nan,))

data = {'Open': open, 'High': high, 'Low': low, 'Close': close, 'Accumulation Distribution': ad, 'Volume': vol, 'Stochastic Oscillator': so, 'Rate of Change': roc, 'Supertrend': sp, 'relative strength': rsi, 'On Balance Volume': obv, 'Macd': macd, 'Ema9': ema9, 'ema21': ema21, 'ema50': ema50, 'ema100': ema100, 'ema200': ema200, 'moving average': moving_avg, 'upper band': upper_band, 'lower band': lower_band}

df = pd.DataFrame(data)

df.to_csv(sys.argv[2], index=True)
