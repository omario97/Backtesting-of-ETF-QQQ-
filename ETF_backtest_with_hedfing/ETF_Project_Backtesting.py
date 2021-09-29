# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 16:18:59 2021

@author: omar_
"""

# ETF_Project_Backtesting

# Import libraries
from scipy.stats import norm
import scipy.stats as sp
import matplotlib.pyplot as plt
import numpy as np
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import pandas as pd
import threading
import time
import numpy as np
from copy import deepcopy


# ticker of choice
ticker = "QQQ"


class TradeApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = {}

    def error(self, reqId, errorCode, errorString):
        print("Error {} {} {}".format(reqId, errorCode, errorString))

    def historicalData(self, reqId, bar):
        if reqId not in self.data:  # our dict is empty, so when we do web socket api the data comes and then we have to chechk wheter there is a match or to create
            self.data[reqId] = [{"Date": bar.date, "Open": bar.open, "High": bar.high,
                                 "Low": bar.low, "Close": bar.close, "Volume": bar.volume}]
        else:
            self.data[reqId].append({"Date": bar.date, "Open": bar.open, "High": bar.high,
                                     "Low": bar.low, "Close": bar.close, "Volume": bar.volume})

    def historicalDataEnd(self, reqId, start, end):
        super().historicalDataEnd(reqId, start, end)
        # just to tell us that it is ended with exctraction
        print("HistoricalDataEnd. ReqId:", reqId, "from", start, "to", end)
        ticker_event.set()

# every function in Tradeapp is a wrapper function. which we can manipulate.


def stock(symbol, sec_type="STK", currency="USD", exchange="SMART"):
    contract = Contract()
    contract.symbol = symbol
    contract.secType = sec_type
    contract.currency = currency
    contract.exchange = exchange
    return contract


def histData(req_num, contract, endDate, duration, candle_size):
    """extracts historical data"""
    app.reqHistoricalData(reqId=req_num,
                          contract=contract,
                          endDateTime=endDate,
                          durationStr=duration,
                          barSizeSetting=candle_size,
                          whatToShow='TRADES',
                          useRTH=1,
                          formatDate=1,
                          keepUpToDate=0,
                          chartOptions=[])	 # EClient function to request contract details


def histVolatility(req_num, contract, endDate, duration, candle_size):
    """extracts historical data"""
    app.reqHistoricalData(reqId=req_num,
                          contract=contract,
                          endDateTime=endDate,
                          durationStr=duration,
                          barSizeSetting=candle_size,
                          whatToShow='HISTORICAL_VOLATILITY',
                          useRTH=1,
                          formatDate=1,
                          keepUpToDate=0,
                          chartOptions=[])	 # EClient function to request contract details


def histImpliedVolatility(req_num, contract, endDate, duration, candle_size):
    """extracts historical data"""
    app.reqHistoricalData(reqId=req_num,
                          contract=contract,
                          endDateTime=endDate,
                          durationStr=duration,
                          barSizeSetting=candle_size,
                          whatToShow='OPTION_IMPLIED_VOLATILITY',
                          useRTH=1,
                          formatDate=1,
                          keepUpToDate=0,
                          chartOptions=[])	 # EClient function to request contract details

# storing trade app object in dataframe


def dataDataframe(symbol, TradeApp_obj):
    df_data = {}
    df_data[symbol] = pd.DataFrame(TradeApp_obj.data[symbol.index(ticker)])
    df_data[symbol].set_index("Date", inplace=True)
    return df_data


# Establish a web socket connection
def connection():
    app.run()


ticker_event = threading.Event()
app = TradeApp()
app.connect(host='127.0.0.1', port=7497, clientId=23)
con_thread = threading.Thread(target=connection, daemon=True)
con_thread.start()
time.sleep(1)  # some latency added to ensure that the connection is established

# 15 years worth of 1 day bar data

months = '15 Y'
pr_bar = '1 day'

# ps when working with 1 ticker there is just string and not list of string
histData(ticker.index(ticker), stock(ticker), '', months, pr_bar,)
ticker_event.wait()
historicaldata = dataDataframe(ticker, app)
app.data = {}  # reset key
time.sleep(2)

ticker_event.clear()
histVolatility(ticker.index(ticker), stock(ticker), '', months, pr_bar,)
ticker_event.wait()
historical_vol = dataDataframe(ticker, app)
app.data = {}  # reset key
time.sleep(2)

ticker_event.clear()
histImpliedVolatility(ticker.index(ticker), stock(ticker), '', months, pr_bar,)
ticker_event.wait()
hist_implied_vol = dataDataframe(ticker, app)
app.data = {}  # reset key
time.sleep(2)
# need to wait


# make them all a data frame.
# easier to work with one df when focusing on 1 ticker
hist_ohlc = pd.DataFrame(historicaldata['QQQ'])
hist_vol = pd.DataFrame(historical_vol['QQQ'])
hist_imp_vol = pd.DataFrame(hist_implied_vol['QQQ'])

# 15 Y of daily data (implie vol looks like it is given in annual term)
hist_ohlc.to_csv(
    r'C:\Users\omar_\OneDrive\Skrivebord\ETF_backtest_with_hedfing\hist_ohlc.csv')
hist_vol.to_csv(
    r'C:\Users\omar_\OneDrive\Skrivebord\ETF_backtest_with_hedfing\hist_vol.csv')
hist_imp_vol.to_csv(
    r'C:\Users\omar_\OneDrive\Skrivebord\ETF_backtest_with_hedfing\hist_imp_vol.csv')


'''The reason for why i have exported it is because i had not specified
    how the dates so if i run the above code again another day, the results for the backtest would be different.
    How ever, i would try to make the code as much dynamic and reuseable for another ticker.'''
# %reset -f

''' We all have heard of the saying of buy and hold, this is an improvment
    in the sense you won't potentially loose all your money in 
    every bust in the economic cycle.'''

''' The plan for the backtest is:
    i) Find out the distribution we are working with
    ii) Compute the RSI for the qqq data (only signal we will use when we want to hedge)
    iii) we will use BS to calc the put option(European) with hist volatility
    iii) we would then solve for the put price given implied vol observed in the mkt at time(t)
         The reason is for me to be better at vecotrizing applications(we could actually just use implied vol right away)
    IV)  we would compute how many option we have to buy in order to hedge (6 month experations)
    V) Also we are going to do another scenario where we use leap option from the start in order to 
        be ready for black swan events. 
    VI) also we have to compute KPI's '''

# import pacakges
pd.pandas.set_option('display.max_columns', None)

# import the files saved from localdrive at time(t)
ohlc = pd.read_csv(
    r'C:\Users\omar_\OneDrive\Skrivebord\ETF_backtest_with_hedfing\hist_ohlc.csv')
#vol = pd.read_csv(r'C:\Users\omar_\OneDrive\Skrivebord\ETF_backtest_with_hedfing\hist_vol.csv')
implied_vol = pd.read_csv(
    r'C:\Users\omar_\OneDrive\Skrivebord\ETF_backtest_with_hedfing\hist_imp_vol.csv')


# i) let us find what kind of dist the underlying data has.

def ret_ditribution(df):
    data = df.copy()
    ret = np.log(data['Close']) - \
        np.log(data['Close'].shift())  # vectorized operation
    ret.rename('Returns', inplace=True)
    ret.dropna(inplace=True)
    return ret


ret_vector = ret_ditribution(ohlc)
ret_vector.plot.box()  # a lot of outliers in the returns,

# plotting the distribution
ret_vector.hist(bins=100)
plt.xlabel('Returns')
plt.ylabel("Count")
plt.title('Returns')
plt.grid(False)
plt.show()

'''We can see that the distribution has outliers and to be more quantitative 
   we can calc som descriptive stats inlduing the 3,4 moment to verify our
   hypothesis of normality'''

print(ret_vector.describe())
# Negative skewed meanining when bad day on the mkt happens, it can go real bad.
print(sp.skew(ret_vector))
print(sp.kurtosis(ret_vector))  # leptokurtis > 3  -> should = 0

# talk about the shift in distribution when T-> increases i.e dragged volatility


# lets start by computing the KPI's for a buy-and-hold strategy of the security.

# transform the dates into datetime object
ohlc['date'] = pd.to_datetime(
    ohlc['Date'], format='%Y%m%d')  # useful in many cases
ohlc.drop(columns='Date', inplace=True)
ohlc.set_index('date', inplace=True)


def sec_ret(DF):
    df = DF.copy()
    df['ret'] = df['Close'].pct_change()
    df["cum_return"] = (1 + df["ret"]).cumprod()
    return df['ret']


ohlc['ret'] = sec_ret(ohlc)


def CAGR(DF):
    df = DF.copy()
    df["cum_return"] = (1 + df["ret"]).cumprod()
    n = len(df['ret'])/252
    CAGR = (df["cum_return"].tolist()[-1])**(1/n) - 1
    return CAGR


def volatility(DF):
    df = DF.copy()
    vol = df["ret"].std() * np.sqrt(252)
    return vol


def sharpe(DF, rf):
    df = DF.copy()
    sr = (CAGR(df) - rf)/volatility(df)
    return sr


def calculateMaxDD(DF):
    df = DF.copy()
    Roll_Max = df['Close'].cummax()
    Daily_Drawdown = df['Close']/Roll_Max - 1.0
    Max_Daily_Drawdown = Daily_Drawdown.cummin()
    min_value = Max_Daily_Drawdown.min()
    return min_value


# Stats
buy_and_hold_dict = {}
buy_and_hold_dict['CAGR'] = CAGR(ohlc)
buy_and_hold_dict['vol'] = volatility(ohlc)
buy_and_hold_dict['Sharp_Ratio'] = sharpe(ohlc, 0.015)
buy_and_hold_dict['Max DD'] = calculateMaxDD(ohlc)


print(buy_and_hold_dict)
# since we fixed the datetime ibject the grph in now more clean
ohlc['Close'].plot()
# The max DD was about 31% and it was around the financial crisis were the dd lasted over 2 year


''' Now we are going to do a backtest were we actually go on and hedge on signal(RSI) to see if we can improve on buy and hold '''
# I will be working with ohlc and implied_vol data set.
ohlc.head()
ohlc = ohlc[['Open', 'High', 'Low', 'Close']]
implied_vol.head()
implied_vol['date'] = pd.to_datetime(
    implied_vol['Date'], format='%Y%m%d')  # useful in many cases
implied_vol.drop(columns='Date', inplace=True)
implied_vol.set_index('date', inplace=True)

# we will use gorupby
ohlc = ohlc.join(implied_vol['Close'], on='date', lsuffix='_left')
ohlc = ohlc.rename(columns={'Close_left': 'Close', 'Close': 'implied_vol'})

# check if missing value
ohlc.isna().sum()  # 3 Nan values
# The reason is we gonna calc the option prices, but we are not going to buy option every day, we will only buy option if signal is triggered
ohlc.ffill(axis=0, inplace=True)
ohlc.isna().sum()


# we will now move to calc the techincal indicator RSI(Relative Strength index)
# inflection points are: 70 overbought, 30 oversold and 50 for neutral
def RSI(DF, n=14):
    df = DF.copy()
    df["change"] = df["Close"] - df["Close"].shift(1)
    df["gain"] = np.where(df["change"] >= 0, df["change"], 0)
    df["loss"] = np.where(df["change"] < 0, -1*df["change"], 0)
    df["avgGain"] = df["gain"].ewm(alpha=1/n, min_periods=n).mean()
    df["avgLoss"] = df["loss"].ewm(alpha=1/n, min_periods=n).mean()
    df["rs"] = df["avgGain"]/df["avgLoss"]
    df["rsi"] = 100 - (100 / (1 + df["rs"]))
    return df["rsi"]


ohlc['rsi'] = RSI(ohlc, n=14)
ohlc.dropna(inplace=True)


# we now need to calc option prices, BS-merton put option pricing.
# we will calculate the 3 month option pricec for a  delta = 0.5 i.e K = S
# since we are assuming annualized volatility and rf , then we can get 3 month T by saying 0.25


def BSVal(S, sigma, rf, T, K, Call=True):
    """
    Calculates the Black-Scholes option value for a European option
    """
    d1 = (np.log(S/K)+(rf+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if Call == True:
        val = S*norm.cdf(d1)-K*np.exp(-rf*T)*norm.cdf(d2)
    else:
        val = K*np.exp(-rf*T)*norm.cdf(-d2)-S*norm.cdf(-d1)
    return val


ohlc['put'] = BSVal(ohlc['Close'], ohlc['implied_vol'],
                    0.015, 0.25, ohlc['Close'], True)

# option_vals =  ohlc.apply(lambda x: BSVal(x['Close'],x['implied_vol'],0.015,0.25, x['Close'],False))  #this method is acctually better but need to refine it.
ohlc.head()  # ps the multipler for pr option is 100 for qqq etf.


# Now we have to make the signal column + have a column where indicates whic dates we bought and that it is still in use so we dont buy nesecarry
# for this we need a inital value to begin with.

pos_size = 100000
ohlc['return'] = ohlc['Close'].pct_change()
ohlc['return'].fillna(0, inplace=True)
ohlc['wealth'] = np.zeros(ohlc['return'].shape)
ohlc['wealth'][0] = pos_size

for i in range(1, len(ohlc['wealth'])):
    # slow, but this is aceptble with the data set.
    ohlc['wealth'].iloc[i] = ohlc['wealth'].iloc[i-1] * \
        (1 + ohlc['return'].iloc[i])

# if we were to hedge at time t how many contract would we need to buy(contract size = 100)
ohlc['nr_shares'] = round(ohlc['wealth'] / ohlc['Close'])
ohlc['Premium_paid_for_hedge'] = ohlc['put'] * 100 * (ohlc['nr_shares'] / 100)
ohlc['option_bought'] = np.ceil((ohlc['Premium_paid_for_hedge'] / 100))

# now the idea is create a signal on when we should buy a put at time(t) in order
# to hedge our downside and adjust our wealth accordingly.
# the signal should be a dcreasing RSI from its high with
# we only need to hedge when the market is overbought and there is pullback occuring
#we are using 10 lookback periods for the RSI change calculations
ohlc['signal'] = np.zeros(ohlc['Close'].shape)
for i in range(10, len(ohlc['Close'])):
    if ohlc['rsi'].iloc[i] > 70 and ohlc['rsi'].iloc[i-10:i].pct_change().dropna().sum() < 0.05:    
        ohlc['signal'].iloc[i] = 1
    else:
        ohlc['signal'].iloc[i] = 0
        
#now we have to refine the signal by not having dublicates plus the option has expiration T = 60(20 trading days) 
for i in range(0, len(ohlc['signal'])):
    if ohlc['signal'].iloc[i] == 1:
       ohlc['signal'].iloc[i+1:i+59] = 0 


#option profit/loss, remember that this is a ATM option that is why we store k in a variable in the loop
ohlc['option_pnl'] = np.zeros(ohlc['Close'].shape)
ohlc['strike_t'] = np.zeros(ohlc['Close'].shape)
for i in range(0, len(ohlc['Close'])):
    if ohlc['signal'].iloc[i] == 1:
         ohlc['strike_t'].iloc[i] = ohlc['Close'].iloc[i]
         ohlc['option_pnl'].iloc[i] = -(ohlc['Premium_paid_for_hedge'].iloc[i])
         

for i in range(0, len(ohlc['Close'])):
    if ohlc['signal'].iloc[i] == 1:
         k = ohlc['strike_t'].iloc[i] 
         o_bought = ohlc['option_bought'].iloc[i]
         prem = ohlc['Premium_paid_for_hedge'].iloc[i]
         try:
             ohlc['option_pnl'].iloc[i+ 59 ] = (np.fmax(k - ohlc['Close'].iloc[i + 59], 0)) * o_bought *100 #59 days
         except:
             print('out of bound')
         finally:
             continue
         


#we need to fix the profit calc
ohlc['comb_wealth'] = ohlc['wealth'] + ohlc['option_pnl']
ohlc['comb_wealth'].plot()
df_new = pd.DataFrame( ohlc['comb_wealth'],index = ohlc['comb_wealth'].index)
df_new.rename(columns = {'comb_wealth':'Close'},inplace = True)
df_new['ret'] = sec_ret(df_new)

buy_and_hold_dict_new = {}
buy_and_hold_dict_new['CAGR'] = CAGR(df_new)
buy_and_hold_dict_new['vol'] = volatility(df_new)
buy_and_hold_dict_new['Sharp_Ratio'] = sharpe(df_new, 0.015)
buy_and_hold_dict_new['Max DD'] = calculateMaxDD(df_new)

print(buy_and_hold_dict)
print(buy_and_hold_dict_new)

'''The result rejected my hypothesis, with hedging the metric was worsen
    There is ofcourse to consider market timing, the option(theory regrading volatility smile) and their greeks.
    The expiration of the contract. There can be other things such as 
    the indicator is not statisitical significance and/or the parameter and 
    condition is not optimized(ps: this can introduce overfitting)'''


    
