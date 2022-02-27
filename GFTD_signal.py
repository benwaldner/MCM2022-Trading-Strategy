import pandas as pd
import numpy as np
from position import *


def gftd_para_list(n1=list(range(2, 5, 1)), n2=list(range(2, 5, 1)), n3=list(range(2, 5, 1))) -> list:
    para_list = []
    for i in n1:
        for j in n2:
            for k in n3:
                para_list.append([i, j, k])
    return para_list


# Guangfa TD Signal
def signal_gftd(df, para: list = None) -> pd.DataFrame:
    """
    广发TD策略V2，只能做多不能做空，形成卖出形态会转换成平多仓
    :param df:  原始数据
    :param para:  参数，[n1, n2, n3]
    :return:
    """
    # 辅助函数，先跳过两个函数的内容
    def is_buy_count(i, pre_close) -> bool:
        """
        判断是否计数为买入形态，需要A，B，C三个条件同时满足才行
        :param i: 当前循环的index
        :param pre_close: 上一次计数的收盘价，第一次为None，会忽略C条件
        :return: bool
        """
        # A. 收盘价大于或等于之前第 2 根 K 线最高价;
        a = df.at[i, 'close'] >= df.at[i-2, 'close']
        # B. 最高价大于之前第 1 根 K 线的最高价;
        b = df.at[i, 'close'] > df.at[i-1, 'close']
        # C. 收盘价大于之前第 1 个计数的收盘价。
        c = (df.at[i, 'close'] > pre_close) if pre_close is not None else True
        return a and b and c

    def is_sell_count(i, pre_close) -> bool:
        """
        判断是否计数为卖出形态，需要A，B，C三个条件同时满足才行
        :param i: 当前循环的index
        :param pre_close: 上一次计数的收盘价，第一次为None，会忽略C条件
        :return: bool
        """
        # A. 收盘价小于或等于之前第 2 根 K 线最低价;
        a = df.at[i, 'close'] <= df.at[i-2, 'close']
        # B. 最低价小于之前第 1 根 K 线的最低价;
        b = df.at[i, 'close'] < df.at[i-1, 'close']
        # C. 收盘价小于之前第 1 个计数的收盘价。
        c = (df.at[i, 'close'] < pre_close) if pre_close is not None else True
        return a and b and c

    # ===参数
    if para is None:
        para = [4, 4, 4]  # 默认为4，4，4
    n1, n2, n3 = para

    # ===寻找启动点
    # 计算ud
    df['ud'] = 0  # 首先设置为0
    # 根据收盘价比较设置1或者-1
    df.loc[df['close'] > df.shift(n1)['close'], 'ud'] = 1
    df.loc[df['close'] < df.shift(n1)['close'], 'ud'] = -1

    # 对最近n2个ud进行求和
    df['udc'] = df['ud'].rolling(n2).sum()

    # 找出所有形成买入或者卖出的启动点，并且赋值为1或者-1
    # -1代表买入启动点，1代表卖出启动点
    df.loc[df['udc'].abs() == n2, 'checkpoint'] = df['udc'] / n2

    # 找出所有启动点的索引值，即checkpoint那一列非空的所有行
    check_point_index = df[df['checkpoint'].notnull()].index

    # ===生成买入或者卖出信号
    # [主循环] 从前往后，针对启动点的索引值进行循环
    for index in check_point_index:
        # 我们实际使用1代表买入，和启动点（checkpoint）正好相反，
        # 取负数就能计算得到可能使用的信号值，这里卖出信号是-1，之后会有处理
        signal = -df.at[index, 'checkpoint']

        # 缓存信号形成过程中的最高价和最低价，用于计算止损价格
        min_price = df.loc[index - n2: index, 'close'].min()
        max_price = df.loc[index - n2: index, 'close'].max()

        pre_count_close = None  # 之前第1个计数的收盘价，默认为空
        cum_count = 0  # 满足计数形态的累计值，默认清零
        stop_lose_price = 0  # 止损价格

        # [子循环] 从启动点（checkpoint）下一根k线开始往后，搜索满足buy count和sell count的形态
        for index2 in df.loc[index + 1:].index:
            close = df.at[index2, 'close']  # 当前收盘价
            min_price = min(min_price, close)  # 计算信号开始形成到这一步的最低价
            max_price = max(max_price, close)  # 计算信号开始形成到这一步的最高价

            # ==如果当前是启动点，并且当前k线满足buy count的形态
            # 1. 累计加一
            # 2. 缓存当前收盘价
            # 3. 记录止损价格（这一步并不会放到df中）
            if signal == 1 and is_buy_count(index2, pre_count_close):
                # 买入启动点
                cum_count += 1
                pre_count_close = close  # 更新前一个计数收盘价
                stop_lose_price = min_price
            elif signal == -1 and is_sell_count(index2, pre_count_close):
                # 卖出启动点
                cum_count += 1
                pre_count_close = close  # 更新前一个计数收盘价
                stop_lose_price = max_price

            # ==如果遇到新的启动点，重新开始计数
            #   退出子循环，继续主循环的下一个启动点处理
            if df.at[index2, 'checkpoint'] > 0 or df.at[index2, 'checkpoint'] < 0:
                break

            # ==如果累计计数达到n3，发出交易信号
            #   退出子循环，继续主循环的下一个启动点处理
            if cum_count == n3:
                # 设置当前信号
                df.loc[index2, 'signal'] = max(
                    signal, 0)  # 如果是-1就赋值为0，这个信号函数不包含做空
                # 设置产生信号的时候的止损价格
                df.loc[index2, 'stop_lose_price'] = stop_lose_price
                break

    # ===新增了signal（信号）列和对应的stop_lose_price（止损价）列
    # ===处理止损信号
    df['stop_lose_price'].fillna(
        method='ffill', inplace=True)  # 设置当前信号下所有行的止损价格
    df['cur_sig'] = df['signal']
    df['cur_sig'].fillna(method='ffill')
    stop_on_long_condition = (df['cur_sig'] == 1) & (
        df['close'] < df['stop_lose_price'])
    stop_on_short_condition = (df['cur_sig'] == 0) & (
        df['close'] > df['stop_lose_price'])
    df.loc[stop_on_long_condition |
           stop_on_short_condition, 'signal'] = 0  # 设置止损平仓信号

    # ===信号去重复
    temp = df[df['signal'].notnull()][['signal']]
    temp = temp[temp['signal'] != temp['signal'].shift(1)]
    df['signal'] = temp['signal']

    # ===去除不要的列
    df.drop(['ud', 'udc', 'checkpoint', 'stop_lose_price',
            'cur_sig'], axis=1, inplace=True)

    # ===由signal计算出实际的每天持有仓位
    # signal的计算运用了收盘价，是每根K线收盘之后产生的信号，到第二根开盘的时候才买入，仓位才会改变。
    df['pos'] = df['signal'].shift()
    df['pos'].fillna(method='ffill', inplace=True)
    df['pos'].fillna(value=0, inplace=True)  # 将初始行数的position补全为0
    return df


def main_gftd(df, c_rate, para_list):
    # 循环测试最好参数
    rtn = pd.DataFrame()
    for para in para_list:
        try:
            temp_df = signal_gftd(df.copy(), para=para)  # 计算信号，需要用copy，防止改变df
            temp_df = position_at_close(temp_df)  # 计算持仓
            temp_df = equity_curve_with_long_at_close(
                temp_df, c_rate=c_rate, t_rate=0, slippage=0)  # 资金曲线
            str_return = temp_df.iloc[-1]['equity_curve']
            base_return = temp_df.iloc[-1]['equity_curve_base']
            # cumulative return
            rtn.loc[str(para), 'cumulative_return'] = round(str_return, 2)
            # annual return
            annual_return = (str_return) ** ('1 days 00:00:00' /
                                             (temp_df['date'].iloc[-1] - temp_df['date'].iloc[0]) * 365) - 1
            rtn.loc[str(para), 'annual_return'] = str(
                round(annual_return * 100, 2)) + '%'

            # calculate maximum drawdown
            # maximum value curve till today
            temp = temp_df.copy()
            temp['max2here'] = temp['equity_curve'].expanding().max()
            # drawdown
            temp['dd2here'] = temp['equity_curve'] / temp['max2here'] - 1
            # calculate maximum drawdown and its recover period
            end_date, max_draw_down = tuple(temp.sort_values(
                by=['dd2here']).iloc[0][['date', 'dd2here']])
            # begin_period
            start_date = temp[temp['date'] <= end_date].sort_values(
                by='equity_curve', ascending=False).iloc[0]['date']
            # delete irrelevant variables
            temp.drop(['max2here', 'dd2here'], axis=1, inplace=True)
            rtn.loc[str(para), 'max_drawdown'] = format(max_draw_down, '.2%')
            rtn.loc[str(para), 'drawdown_begin'] = str(start_date)
            rtn.loc[str(para), 'drawdown_end'] = str(end_date)
            # annual_return divided by maximum drawdown
            rtn.loc[str(para),
                    'return/drawdown'] = round(annual_return / abs(max_draw_down), 2)
            # sharpe ratio
            rtn.loc[str(para), 'sharpe_ratio'] = temp['equity_curve'].pct_change(
            ).mean()/temp['equity_curve'].pct_change().std()*np.sqrt(252)

            # return of the asset itself
            rtn.loc[str(para), 'base_return'] = base_return
        except:
            continue
    return rtn


def tune_parameter(gold, bitcoin):
    # tune model parameters
    rtn1 = main_gftd(gold, 1e-4, para_list=gftd_para_list(n1=list(range(2, 7, 1)), n2=list(
        range(2, 7, 1)), n3=list(range(2, 7, 1))))
    rtn1.index.name = 'parameter'
    rtn1.dropna(inplace=True)
    # rtn1 = rtn1.sort_values(by='max_drawdown', ascending=True)
    # rtn1 = rtn1.sort_values(by='return/drawdown', ascending=False)
    rtn1 = rtn1.sort_values(by='sharpe_ratio', ascending=False)
    rtn1.to_csv('./result/GFTD_signal_gold.csv', index=True, header=True)
    n1_gold, n2_gold, n3_gold = eval(rtn1.index[0])

    rtn2 = main_gftd(bitcoin, 2e-4, para_list=gftd_para_list(n1=list(range(2, 7, 1)), n2=list(
        range(2, 7, 1)), n3=list(range(2, 7, 1))))
    rtn2.index.name = 'parameter'
    rtn2.dropna(inplace=True)
    # rtn2 = rtn2.sort_values(by='max_drawdown', ascending=True)
    # rtn2 = rtn2.sort_values(by='return/drawdown', ascending=False)
    rtn2 = rtn2.sort_values(by='sharpe_ratio', ascending=False)
    rtn2.to_csv('./result/GFTD_signal_bitcoin.csv', index=True, header=True)
    n1_bitcoin, n2_bitcoin, n3_bitcoin = eval(rtn2.index[0])
    return (n1_gold, n2_gold, n3_gold), (n1_bitcoin, n2_bitcoin, n3_bitcoin)


if __name__ == '__main__':
    # read the data
    gold, bitcoin = load_data()
    (n1_gold, n2_gold, n3_gold), (n1_bitcoin, n2_bitcoin,
                                  n3_bitcoin) = tune_parameter(gold, bitcoin)
    g  # get the performance evaluation under the best performance
    result1, gold_best = main_gftd(gold, 1e-4, [[n1_gold, n2_gold, n3_gold]])
    result1.index.name = 'parameter'
    result2, bitcoin_best = main_gftd(
        bitcoin, 1e-4, [[n1_bitcoin, n2_bitcoin, n3_bitcoin]])
    result2.index.name = 'parameter'
    # save to csv file
    gold_best.to_csv("MA_gold_best.csv", index=True, header=True)
    bitcoin_best.to_csv("MA_bitcoin_best.csv", index=True, header=True)

    print("==================================================================")
    print(result1)
    print("==================================================================")
    print(result2)
    print("==================================================================")
