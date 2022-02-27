from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from util import mean_variance_sliding_result_large

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
def test_strategy(strategies):
    # read the data
    print('Using strategies', strategies)
    data = load_raw_data()
    data.index = data.pop('date')
    data['cash'] = 1
    portfolio_info = {
        'bitcoin': pd.DataFrame([[1, 0, 0]]*data.shape[0], columns=['bitcoin', 'gold', 'cash']),
        'gold': pd.DataFrame([[0, 1, 0]]*data.shape[0], columns=['bitcoin', 'gold', 'cash']),
        'cash': pd.DataFrame([[0, 0, 1]]*data.shape[0], columns=['bitcoin', 'gold', 'cash']),
    }
    # modify the format
    for name in strategies:
        strategy = pd.read_csv(f'./result/{name}_best.csv', index_col='date', parse_dates=['date'])
        if name != 'rotation':
            strategy = strategy.rename(columns={'equity_curve': name})
        else:
            strategy = strategy.rename(columns={'strategy_net': name})
        data = pd.merge(data, strategy[name], on='date', how='left')
        component = pd.DataFrame([[0, 0, 0]]*data.shape[0], columns=['bitcoin', 'gold', 'cash'])
        component.index = data.index
        if 'bitcoin' in name:
            # ema, ma bitcoin
            component.loc[strategy[strategy['pos']==1].index, 'bitcoin'] = 1
            component.loc[strategy[strategy['pos']==0].index, 'cash'] = 1
        elif 'gold' in name:
            # ema, ma gold
            component.loc[strategy[strategy['pos']==1].index, 'gold'] = 1
            component.loc[strategy[strategy['pos']==0].index, 'cash'] = 1
        elif 'rotation' in name:
            # rotation
            component.loc[strategy[strategy['pos']=='bitcoin'].index, 'bitcoin'] = 1
            component.loc[strategy[strategy['pos']=='gold'].index, 'gold'] = 1
            component.loc[strategy[strategy['pos']=='cash'].index, 'cash'] = 1
        component.reset_index(drop=True, inplace=True)
        portfolio_info[name] = component

    return data, portfolio_info

def plot_component(portfolio: pd.DataFrame):
    portfolio = portfolio.copy()

    labels = list(portfolio.columns)
    prop = portfolio.to_numpy()
    x = list(portfolio.index)
    lwr = np.array([0] * len(x))
    for i in range(len(labels)):
        p = prop[:, i]
        upr = lwr + p
        plt.fill_between(x, lwr, upr, label=labels[i])
        lwr = upr

    plt.legend()
    plt.ylabel('Proportion')
    plt.xlabel('Date')
    plt.title('Portfolio components over time')
    plt.show()

        
if __name__ == '__main__':
    strategies = ['EMA_bitcoin', 'EMA_gold', 
                  'GFTD_bitcoin', 'GFTD_gold', 
                  'MA_bitcoin', 'MA_gold',
                #   'mom_bitcoin', 'mom_gold',
                  'rotation']
    data, portfolio_info = test_strategy(strategies)

    # # testing for update period and period_len
    # update_periods = [1, 5, 10, 20, 30, 50, 100, 300] # row
    # period_lens = [10, 20, 30, 50, 100, 300, 10000]   # column

    # results = []
    # for period_len in period_lens:
    #     res = []
    #     for update_period in update_periods:
    #         print(period_len, update_period)
    #         result, portfolio = mean_variance_sliding_result_large(data, portfolio_info, 10, update_period, 
    #                                                         0.6, period_len, True, 
    #                                                        return_portfolio_weight=True,
    #                                                        benchmark='rotation')
    #         res.append(result['value'].iloc[-1])
    #     results.append(res)
    # info = pd.DataFrame(results)
    # info.columns = update_periods
    # info.index = period_lens
    # info.to_csv('update-len-large.csv', index=True)
    # print(info)

    # run portfolio
    result, portfolio = mean_variance_sliding_result_large(data, portfolio_info, 3, 5, 0.6, 50, True, 
                                                           return_portfolio_weight=True,
                                                           benchmark='rotation',
                                                           return_commission=True)
    quit()
    gammas = [0, 1, 2, 3, 5, 7, 10]
    fee_rate = []
    for gamma in gammas:
        result, portfolio = mean_variance_sliding_result_large(data, portfolio_info, gamma, 5, 0.6, 50, True, 
                                                           return_portfolio_weight=True,
                                                           benchmark='bitcoin',
                                                           return_commission=True)
        rate = (result['commission'] / result['value']).tolist()
        fee_rate.append(rate)
    df = pd.DataFrame(fee_rate).transpose()
    df.columns = gammas
    print(df.describe())
    df.to_csv('commission-gamma.csv', index=True)
    # result[['value', 'commission']].plot()
    # plt.show()
    # result['cum_commission'] = result['commission'].cumsum()
    # result[['value', 'commission', 'cum_commission']].plot()
    # plt.show()
    # plt.plot(result['commission'] / result['value'])
    # avg = (result['commission'] / result['value']).mean()
    # plt.plot([result.index[0], result.index[-1]], [avg, avg], '--')
    # plt.show()
    # print(result['value'].iloc[-1])
    portfolio.index = data.index
    plot_component(portfolio)
    plot_component(result[['prop_bitcoin', 'prop_gold', 'prop_cash']])
    result['value'] /= 1000
    result['bitcoin'] /= 621.65
    result['gold'] /= 1324.60
    result['cash'] = 1
    print(result)
    result.plot()
    plt.show()
    print(portfolio)
    portfolio.plot()
    plt.show()