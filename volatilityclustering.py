# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# list of stocks to analyze
yfin_stock_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'SPY', 'NVDA', 'MDB', 'GME', 'HOOD', 'OKTA', 'NFLX', 'AMD', 'META', 'JPM', 'PLTR', 'BA']

# get price data from yfinance
def historicalreturns(tickers_list, start_date = '2024-01-01', end_date = '2025-01-01') :

    # dictionary of tickers and daily returns
    daily_returns = {}

    for ticker in tickers_list :

        # download data from yfinance
        df = yf.download(ticker, start=start_date, end=end_date, progress=False).stack(future_stack=True).rename_axis(['Date', 'Ticker']).reset_index(level=1)
        df.columns.name = None # drop column name
        df['Returns'] = np.log(df['Close']).diff().fillna(0)   # add log returns
        daily_returns[ticker] = df[['Returns']] # save to dictionary daily_returns 
    
    return daily_returns


# get historical returns for all yfin_stock_tickers 
daily_returns = historicalreturns(tickers_list = yfin_stock_tickers, start_date='2023-01-01', end_date = '2024-12-31')


# function to add rolling volatility component for each ticker in dictionary daily_returns
def rollingwindow (dict, window = 22) :
    
    # vol dict
    rollingvol_dict = {}
    # unpack dictionary dict
    for ticker, df in dict.items() :
        
        rollingvol_list = [] # store rolling vol for given ticker

        for i in range(window, len(df['Returns'])) :

            x = df['Returns'][i - window : i] # type = pd.Series
            mu = (1/window) * np.ones(window).dot(x) # mean of return window
            variance = (1/(window - 1)) * (x - mu).T.dot(x - mu) # sample variance of return over window
            sigma = np.sqrt(variance) # sample std dev of return over window
            rollingvol_list.append(sigma)  # append rolling window vol to list

        # collect rolling vol
        rollingvol_dict[ticker] = rollingvol_list

    # return rolling vol for all tickers
    return rollingvol_dict

# generate rolling vol df for yfin_stock_tickers          
rollingvol = pd.DataFrame(rollingwindow(daily_returns))
#print(rollingvol.T)  # display rolling vols to console 

# scaler object
scaler = StandardScaler()
rollingvol_scaled = scaler.fit_transform(rollingvol)

# clusters
clusters = 5
kmeans = KMeans(n_clusters = clusters, random_state= 0 , n_init='auto')
cluster_labels = kmeans.fit_predict(rollingvol_scaled.T)
#print(cluster_labels)

# pricipal component analysis
pca = PCA(n_components=2)
vpca = pca.fit_transform(rollingvol_scaled.T)
#print(pca.explained_variance_ratio_)


# assign coloring for clusters
colors = lambda c : 'red' if c == 0 else 'green' if c ==1 else 'blue' if c == 2 else 'brown' if c == 3 else 'purple'
plt_colors = list(map(colors, cluster_labels))

# plotting

fig = plt.figure()
ax = fig.add_subplot(111)

for stock, color, (xx, yy) in zip(yfin_stock_tickers, plt_colors, vpca) :
    ax.scatter(xx, yy, color = color)
    ax.annotate(stock, xy = (xx, yy))

ax.set_title("Stocks Grouped by Vol Behaviour")
ax.set_xlabel("Pricipal Component 1")
ax.set_ylabel("Pricipal Component 2")
plt.show()








