import pandas as pd
from util import *

def read_data(path, name):
    df = pd.read_csv(path)
    df.columns = ['date', name]
    df['date'] = pd.to_datetime(df['date'])
    return df


def load_raw_data():
    gold = read_data('./raw_data/LBMA-GOLD.csv', 'gold')
    bit = read_data('./raw_data/BCHAIN-MKPRU.csv', 'bitcoin')

    data = pd.merge(bit, gold, how='outer', on='date')
    return data

# test the strategy
def test_strategy(strategy_name):
    # read the data
    if strategy_name == "MA":
        bitcoin_strat = pd.read_csv("./result/MA_bitcoin_best.csv", index_col="date", parse_dates=['date'])
        gold_strat = pd.read_csv("./result/MA_gold_best.csv", index_col="date", parse_dates=['date'])
    if strategy_name == "GFTD":
        bitcoin_strat = pd.read_csv("./result/GFTD_bitcoin_best.csv", index_col="date", parse_dates=['date'])
        gold_strat = pd.read_csv("./result/GFTD_gold_best.csv", index_col="date", parse_dates=['date'])

    # modify the format
    data = load_raw_data()
    data.index = data.pop('date')
    data['cash'] = 0
    bitcoin_strat = bitcoin_strat.rename(columns={"equity_curve": 'bitcoin_strat'})
    gold_strat = gold_strat.rename(columns={"equity_curve": 'gold_strat'})
    data2 = pd.merge(bitcoin_strat, gold_strat, on='date', how='outer')
    data2 = data2[['bitcoin_strat', 'gold_strat']]
    data = pd.merge(data, data2, on='date', how='outer')

    # calculate the weight
    w_bit = pd.DataFrame([[0, 0, 1]]*data.shape[0], columns=['bitcoin', 'gold', 'cash'])
    w_bit.index = data.index
    w_bit.loc[bitcoin_strat[bitcoin_strat['pos']==1].index, 'bitcoin'] = 1
    w_bit.loc[bitcoin_strat[bitcoin_strat['pos']==1].index, 'cash'] = 0
    w_bit.reset_index(drop=True, inplace=True)

    w_gold = pd.DataFrame([[0, 0, 1]]*data.shape[0], columns=['bitcoin', 'gold', 'cash'])
    w_gold.index = data.index
    w_gold.loc[gold_strat[gold_strat['pos']==1].index, 'gold'] = 1
    w_gold.loc[gold_strat[gold_strat['pos']==1].index, 'cash'] = 0
    w_gold.reset_index(drop=True, inplace=True)

    portfolio_info = {
        'bitcoin': pd.DataFrame([[1, 0, 0]]*data.shape[0], columns=['bitcoin', 'gold', 'cash']),
        'gold': pd.DataFrame([[0, 1, 0]]*data.shape[0], columns=['bitcoin', 'gold', 'cash']),
        'cash': pd.DataFrame([[0, 0, 1]]*data.shape[0], columns=['bitcoin', 'gold', 'cash']),
        "gold_strat": w_gold,
        "bitcoin_strat": w_bit
    }
    return data, portfolio_info
        
if __name__ == '__main__':
    data, portfolio_info = test_strategy('MA')
    # run portfolio
    result = mean_variance_sliding_result_large(data, portfolio_info, 1, 20, 0.6, 50, True)
    print(result)