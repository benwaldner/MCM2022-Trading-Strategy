import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import quantstats as qs
from simplejson import load

# performance evaluation


def evaluate_investment(source_data, tittle, time='date'):
    temp = source_data.copy()
    # keep the backtesting result
    results = pd.DataFrame()

    # calculate cumulative net worth
    results.loc[0, 'cumulative_return'] = round(temp[tittle].iloc[-1], 2)

    # calculate annual return
    annual_return = (temp[tittle].iloc[-1]) ** ('1 days 00:00:00' /
                                                (temp[time].iloc[-1] - temp[time].iloc[0]) * 365) - 1
    results.loc[0, 'annual_return'] = str(round(annual_return * 100, 2)) + '%'

    # calculate maximum drawdown
    # maximum value curve till today
    temp['max2here'] = temp[tittle].expanding().max()
    # drawdown
    temp['dd2here'] = temp[tittle] / temp['max2here'] - 1
    # calculate maximum drawdown and its recover period
    end_date, max_draw_down = tuple(temp.sort_values(
        by=['dd2here']).iloc[0][[time, 'dd2here']])
    # begin_period
    start_date = temp[temp[time] <= end_date].sort_values(
        by=tittle, ascending=False).iloc[0][time]
    # delete irrelevant variables
    temp.drop(['max2here', 'dd2here'], axis=1, inplace=True)
    results.loc[0, 'max_drawdown'] = format(max_draw_down, '.2%')
    results.loc[0, 'drawdown_begin'] = str(start_date)
    results.loc[0, 'drawdown_end'] = str(end_date)
    # annual_return divided by maximum drawdown
    results.loc[0,
                'return/drawdown'] = round(annual_return / abs(max_draw_down), 2)
    # sharpe ratio
    results.loc[0, 'sharpe_ratio'] = temp[tittle].pct_change(
    ).mean()/temp[tittle].pct_change().std()*np.sqrt(252)

    results.index = ['Performance']
    return results.T

# rotation strategy


def rotation_strategy(df, trade_rate1, trade_rate2, momentum_days):
    # return calculation
    df['gold_pct'] = df['gold'].pct_change()
    df['bitcoin_pct'] = df['bitcoin'].pct_change()
    df.rename(columns={'gold': 'gold_close',
              'bitcoin': 'bitcoin_close'}, inplace=True)
    df.reset_index(inplace=True, drop=False)

    # momentum calculation
    df['gold_mom'] = df['gold_close'].pct_change(periods=momentum_days)
    df['bitcoin_mom'] = df['bitcoin_close'].pct_change(periods=momentum_days)

    # rotation principle
    # gold can only be traded during trading days
    df.loc[(df['gold_mom'] > df['bitcoin_mom']) & (
        df['gold_close'].notnull()), 'style'] = 'gold'
    df.loc[df['gold_mom'] < df['bitcoin_mom'], 'style'] = 'bitcoin'
    df.loc[(df['gold_mom'] < 0) & (df['bitcoin_mom'] < 0), 'style'] = 'empty'

    # maintain the same position
    df['style'].fillna(method='ffill', inplace=True)
    # position can only be changed in the next day
    df['pos'] = df['style'].shift(1)
    # delete the day when position is nan
    df.dropna(subset=['pos'], inplace=True)
    # calculate return of the strategy
    df.loc[df['pos'] == 'gold', 'strategy_pct'] = df['gold_pct']
    df.loc[df['pos'] == 'bitcoin', 'strategy_pct'] = df['bitcoin_pct']
    df.loc[df['pos'] == 'empty', 'strategy_pct'] = 0

    # time when changing the position
    df.loc[df['pos'] != df['pos'].shift(1), 'trade_time'] = df['date']
    # correct the price change on the rebalancing day to the opening price buying price change
    df.loc[(df['trade_time'].notnull()) & (df['pos'] == 'gold'), 'strategy_pct_adjust'] = df['gold_close'] / (
        df['gold_close'] * (1 + trade_rate1)) - 1
    df.loc[(df['trade_time'].notnull()) & (df['pos'] == 'bitcoin'), 'strategy_pct_adjust'] = df['bitcoin_close'] / (
        df['bitcoin_close'] * (1 + trade_rate2)) - 1
    df.loc[df['trade_time'].isnull(), 'strategy_pct_adjust'] = df['strategy_pct']

    # deduct the commission fee
    df.loc[(df['trade_time'].shift(-1).notnull()) & (df['pos'] == 'gold'),
           'strategy_pct_adjust'] = (1 + df['strategy_pct']) * (1 - trade_rate1) - 1
    df.loc[(df['trade_time'].shift(-1).notnull()) & (df['pos'] == 'bitcoin'),
           'strategy_pct_adjust'] = (1 + df['strategy_pct']) * (1 - trade_rate2) - 1

    # fill the return with 0 when position is empty
    df['strategy_pct_adjust'].fillna(value=0.0, inplace=True)
    del df['strategy_pct'], df['style']

    df.reset_index(drop=True, inplace=True)
    # calculate net worth
    df['gold_net'] = df['gold_close'] / df['gold_close'][0]
    df['bitcoin_net'] = df['bitcoin_close'] / df['bitcoin_close'][0]
    df['strategy_net'] = (1 + df['strategy_pct_adjust']).cumprod()

    # evaluate the strategy
    res = evaluate_investment(df, 'strategy_net', time='date')
    return res, df


# tune the parameter
def parameter_tuning(data, para_range=range(10, 100, 2), key='return/drawdown'):
    # para_range: range of parameter
    # key: pick the best strategy with the highest 'return/drawdown', 'strat_return', 'annual_return', 'max_drawdown', or 'sharpe_ratio'
    performance = pd.DataFrame()
    for momentum_days in range(10, 100, 2):
        res, df = rotation_strategy(
            data.copy(), trade_rate1=1e-4, trade_rate2=2e-4, momentum_days=momentum_days)
        performance.loc[str(momentum_days),
                        'strat_return'] = res['Performance'][0]
        performance.loc[str(momentum_days),
                        'annual_return'] = res['Performance'][1]
        performance.loc[str(momentum_days),
                        'max_drawdown'] = res['Performance'][2]
        performance.loc[str(momentum_days),
                        'drawdown_begin'] = res['Performance'][3]
        performance.loc[str(momentum_days),
                        'drawdown_end'] = res['Performance'][4]
        performance.loc[str(momentum_days),
                        'return/drawdown'] = res['Performance'][5]
        performance.loc[str(momentum_days),
                        'sharpe_ratio'] = res['Performance'][6]
    performance.index.name = 'parameter'
    performance = performance.sort_values(by=key, ascending=False)
    return performance


def load_data():
    gold = pd.read_csv("./raw_data/LBMA-GOLD.csv",
                       names=['date', 'gold'], parse_dates=['date'], index_col='date', skiprows=1)
    bitcoin = pd.read_csv("./raw_data/BCHAIN-MKPRU.csv", names=[
                          'date', 'bitcoin'], parse_dates=['date'], index_col='date', skiprows=1)
    bitcoin.sort_index(inplace=True)
    gold.sort_index(inplace=True)
    data = pd.merge(bitcoin, gold, on='date', how='outer')
    return data


if __name__ == '__main__':
    data = load_data()
    # pick the best strategy with the highest annual return over maximum drawdown
    performance = parameter_tuning(
        data, para_range=range(10, 100, 2), key='return/drawdown')
    # obtain the best parameter
    best_para = performance.index[0]
    res, df = rotation_strategy(data.copy(
    ), trade_rate1=1e-4, trade_rate2=2e-4, momentum_days=int(performance.index[0]))
    df.set_index('date', inplace=True)

    # plot the curve
    # df[['bitcoin_net','strategy_net', 'gold_net']].plot(kind='line', grid=True, figsize=(9,6), legend=True)
    # plt.savefig('./images/rotation.png', dpi=1000)

    # save the result
    # df.to_csv("./result/rotation.csv", index=True, header=True)
    # res.to_csv("./result/rotation_performance.csv", index=True, header=True)
    # obtain professional indicators
    qs.reports.html(df['strategy_net'],  output="rotation.html")
    # compare with bitcoin only
    # qs.reports.html(df['strategy_net'], benchmark=df['bitcoin_net'], output="rotation.html")
    # compare with gold only
    # qs.reports.html(df['strategy_net'], benchmark=df['gold_net'], output="rotation.html")
    print(res)
