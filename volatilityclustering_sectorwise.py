# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from volatilityclustering import historicalreturns, rollingwindow
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

if __name__ == "__main__" :

        
    # list of sectors
    GICS_sectors = ['Materials', 'Consumer Discretionary', 'Health Care', 'Financials', 'Information Technology', 'Communication Services'\
                    ,'Real Estate', 'Utilities', 'Energy', 'Consumer Staples']

    # fetch s&p500 companies
    #url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    #sp500_df = pd.read_html(url)[0]

    # save to [symbol - company - GICS Sector and GICS Sub-Indutry] to .csv
    #sp500_df[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']].to_csv('sp500_gics.csv', index= False)

    sp500_gics = pd.read_csv('sp500_gics.csv')

    # check if symbol from wiki is ticker in yfinance

    def is_valid_ticker(ticker):        # Function to check ticker validity
        try:
            info = yf.Ticker(ticker).info
            return bool(info) and 'sector' in info
        except:
            return False
        
    not_found_symbol = []   # tickers not found in yfinance


    #sp500_gics['Yahoo_Ticker'] = sp500_gics['Symbol'].str.replace('.', '-', regex=False)    # Replace '.' with '-' for Yahoo Finance compatibility
    '''
    for idx, row in sp500_gics.iterrows():    # iterate through df
        yf_ticker = row['Symbol']
        if not is_valid_ticker(yf_ticker):
            not_found_symbol.append({
                'Original_Ticker': row['Symbol'],
                'Company': row['Security']
            })
    '''
    #print('not found are : \n', not_found_symbol)  # [{'Original_Ticker': 'BRK.B', 'Company': 'Berkshire Hathaway'}, {'Original_Ticker': 'BF.B', 'Company': 'Brown–Forman'}]

    # remove not_found_symbol from sp500_gics
    sp500_gics = sp500_gics[~sp500_gics['Symbol'].isin(['BRK.B', 'BF.B'])]

    ############################################################################################################################################################

    # Finance Sector
    sp500_finance = sp500_gics[sp500_gics['GICS Sector'] == 'Financials']

    # finance stocks to analyze
    yfin_finance_tickers = sp500_finance['Symbol'].to_list() #['AFL', 'ALL', 'AXP', 'AIG', 'AON', 'APO', 'ACGL', 'EG', 'FDS', 'FIS', 'FITB',  'ICE', 'IVZ', 'JKHY', 'JPM',  'KKR']

    # get historical returns for all yfin_finance_tickers 
    daily_returns_finance = historicalreturns(tickers_list = yfin_finance_tickers, start_date='2023-01-01', end_date = '2024-12-31')

    # generate rolling vol df for yfin_finance_tickers          
    rollingvol_finance = pd.DataFrame(rollingwindow(daily_returns_finance))

    # scaler object
    scaler = StandardScaler()
    rollingvol_finance_scaled = scaler.fit_transform(rollingvol_finance)

    # determine number of clusters

    inertias_finance = []
    sil_scores = []

    for clusters in range (2, 51) :
        kmeans = KMeans(n_clusters = clusters, random_state= 0 , n_init='auto')
        labels = kmeans.fit_predict(rollingvol_finance_scaled.T)
        score = silhouette_score(rollingvol_finance_scaled.T, labels)
        inertias_finance.append(kmeans.inertia_)
        sil_scores.append(score)

    # Plot inertia vs. number of clusters
    K_range = range(2, 51)
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, inertias_finance, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)

    # Plot Silhouette Scores
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, sil_scores, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs. Number of Clusters')
    plt.grid(True)

    # using number of clusters k = 13
    # clusters
    clusters = 13
    kmeans = KMeans(n_clusters = clusters, random_state= 0 , n_init='auto')
    cluster_labels = kmeans.fit_predict(rollingvol_finance_scaled.T)

    # pricipal component analysis
    pca = PCA(n_components=2)
    vpca = pca.fit_transform(rollingvol_finance_scaled.T)
    print(pca.explained_variance_ratio_)

    # assign coloring for clusters
    colors = lambda c: (
        'red' if c == 0 else
        'green' if c == 1 else
        'blue' if c == 2 else
        'brown' if c == 3 else
        'purple' if c == 4 else
        'orange' if c == 5 else
        'cyan' if c == 6 else
        'magenta' if c == 7 else
        'olive' if c == 8 else
        'pink' if c == 9 else
        'gray' if c == 10 else
        'teal' if c == 11 else
        'gold' if c == 12 else
        'black'  # fallback
    )
    plt_colors = list(map(colors, cluster_labels))

    # plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for stock, color, (xx, yy) in zip(yfin_finance_tickers, plt_colors, vpca) :
        ax.scatter(xx, yy, color = color)
        ax.annotate(stock, xy = (xx, yy))

    ax.set_title("Finance Stocks Grouped by Vol Behaviour")
    ax.set_xlabel("Pricipal Component 1")
    ax.set_ylabel("Pricipal Component 2")
    plt.show()


















 







