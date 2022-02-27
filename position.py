import pandas as pd
import numpy as np

# read original data


def load_data():
    gold = pd.read_csv("./raw_data/LBMA-GOLD.csv",
                       names=['date', 'close'], parse_dates=['date'], index_col='date', skiprows=1)
    bitcoin = pd.read_csv("./raw_data/BCHAIN-MKPRU.csv", names=[
                          'date', 'close'], parse_dates=['date'], index_col='date', skiprows=1)

    bitcoin.sort_values(by=['date'], inplace=True)
    gold.sort_values(by=['date'], inplace=True)

    gold['pre_close'] = gold['close'].shift()
    bitcoin['pre_close'] = bitcoin['close'].shift()
    gold.reset_index(drop=False, inplace=True)
    bitcoin.reset_index(drop=False, inplace=True)
    return gold, bitcoin


# create position based on the signal
def position_at_close(df):
    """
    根据signal产生实际持仓。考虑涨跌停不能买入卖出的情况。
    所有的交易都是发生在产生信号的K线的结束时
    :param df:
    :return:
    """
    # ===由signal计算出实际的每天持有仓位
    # 在产生signal的k线结束的时候，进行买入
    df['signal'].fillna(method='ffill', inplace=True)
    df['signal'].fillna(value=0, inplace=True)  # 将初始行数的signal补全为0
    df['pos'] = df['signal'].shift()
    df['pos'].fillna(value=0, inplace=True)  # 将初始行数的pos补全为0

    # ===删除无关中间变量
    # df.drop(['signal'], axis=1, inplace=True)
    return df


# calculate equity curve
def equity_curve_with_long_at_close(df, c_rate=2.5/10000, t_rate=1.0/1000, slippage=0.01):
    """
    计算股票的资金曲线。只能做多，不能做空。并且只针对满仓操作
    每次交易是以当根K线的收盘价为准。
    :param df:
    :param c_rate: 手续费，commission fees，默认为万分之2.5
    :param t_rate: 印花税，tax，默认为千分之1。etf没有
    :param slippage: 滑点，股票默认为0.01元，etf为0.001元
    :return:
    """

    # ==找出开仓、平仓条件
    condition1 = df['pos'] != 0
    condition2 = df['pos'] != df['pos'].shift(1)
    open_pos_condition = condition1 & condition2

    condition1 = df['pos'] != 0
    condition2 = df['pos'] != df['pos'].shift(-1)
    close_pos_condition = condition1 & condition2

    # ==对每次交易进行分组
    df.loc[open_pos_condition, 'start_time'] = df['date']
    df['start_time'].fillna(method='ffill', inplace=True)
    df.loc[df['pos'] == 0, 'start_time'] = pd.NaT

    # ===基本参数
    initial_cash = 1000  # 初始资金，默认为1000刀

    # ===在买入的K线
    # 在发出信号的当根K线以收盘价买入
    df.loc[open_pos_condition, 'stock_num'] = initial_cash * \
        (1 - c_rate) / (df['pre_close'] + slippage)

    # 买入股票之后剩余的钱，扣除了手续费
    df['cash'] = initial_cash - df['stock_num'] * \
        (df['pre_close'] + slippage) * (1 + c_rate)

    # 收盘时的股票净值
    df['stock_value'] = df['stock_num'] * df['close']

    # ===在买入之后的K线
    # 买入之后现金不再发生变动
    df['cash'].fillna(method='ffill', inplace=True)
    df.loc[df['pos'] == 0, ['cash']] = None

    # 股票净值随着涨跌幅波动
    group_num = len(df.groupby('start_time'))
    if group_num > 1:
        t = df.groupby('start_time').apply(
            lambda x: x['close'] / x.iloc[0]['close'] * x.iloc[0]['stock_value'])
        t = t.reset_index(level=[0])
        df['stock_value'] = t['close']
    elif group_num == 1:
        t = df.groupby('start_time')[['close', 'stock_value']].apply(
            lambda x: x['close'] / x.iloc[0]['close'] * x.iloc[0]['stock_value'])
        df['stock_value'] = t.T.iloc[:, 0]

    # ===在卖出的K线
    # 股票数量变动
    df.loc[close_pos_condition, 'stock_num'] = df['stock_value'] / \
        df['close']  # 看2006年初

    # 现金变动
    df.loc[close_pos_condition, 'cash'] += df.loc[close_pos_condition, 'stock_num'] * (df['close'] - slippage) * (
        1 - c_rate - t_rate)
    # 股票价值变动
    df.loc[close_pos_condition, 'stock_value'] = 0

    # ===账户净值
    df['net_value'] = df['stock_value'] + df['cash']

    # ===计算资金曲线
    df['equity_change'] = df['net_value'].pct_change(fill_method=None)
    df.loc[open_pos_condition, 'equity_change'] = df.loc[open_pos_condition,
                                                         'net_value'] / initial_cash - 1  # 开仓日的收益率
    df['equity_change'].fillna(value=0, inplace=True)
    df['equity_curve'] = (1 + df['equity_change']).cumprod()
    df['equity_curve_base'] = (df['close'] / df['pre_close']).cumprod()

    # ===删除无关数据
    # df.drop(['start_time', 'stock_num', 'cash', 'stock_value', 'net_value'], axis=1, inplace=True)

    return df
