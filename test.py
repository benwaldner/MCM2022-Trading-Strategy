import pandas as pd
from util import *

def test_strategy(strategy_name):
    if strategy_name == "signal":
        bitcoin = pd.read_csv("./result/GFTD_bitcoin_best.csv", index_col="date", parse_dates=['date'])
        gold = pd.read_csv("./result/GFTD_gold_best.csv", index_col="date", parse_dates=['date'])
        # bitcoin = pd.read_csv("./result/MA_bitcoin_best.csv", index_col="date", parse_dates=['date'])
        # gold = pd.read_csv("./result/MA_gold_best.csv", index_col="date", parse_dates=['date'])
        bitcoin = bitcoin.rename(columns={"equity_curve": 'bitcoin'})
        gold = gold.rename(columns={"equity_curve": 'gold'})
        data = pd.merge(bitcoin, gold, on='date', how='outer')
        data = data[['bitcoin', 'gold']]
        data['cash'] = 0
        portfolio_info = {
            'bitcoin': pd.DataFrame([[1, 0, 0]]*data.shape[0], columns=['bitcoin', 'gold', 'cash']),
            'gold': pd.DataFrame([[0, 1, 0]]*data.shape[0], columns=['bitcoin', 'gold', 'cash']),
            'cash': pd.DataFrame([[0, 0, 1]]*data.shape[0], columns=['bitcoin', 'gold', 'cash']),
        }
    return data, portfolio_info
        


data, portfolio_info = test_strategy('signal')
# run portfolio
result = mean_variance_sliding_result_large(data, portfolio_info, 1, 20, 0.6, 50, True)
print(result)
