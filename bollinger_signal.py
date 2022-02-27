import pandas as pd
import numpy as np
from position import *
import talib



# 交易信号signal的部分（以MA均线为例），此处根据不同策略更改
# 如果超参数个数不同比如是三个那需要更改上面的para_list函数，这里的para，
def bollinger_signal(df, para=20) -> pd.DataFrame:
    """
    简单的移动平均线策略，只能做多。
    当短期均线上穿长期均线的时候，做多，当短期均线下穿长期均线的时候，平仓
    :param df:
    :param para: ma_short, ma_long
    :return: 最终输出的df中，新增字段：signal，记录发出的交易信号
    """

   # bollinger band
    Bolling_timeperiod = para
    F_value_up = 2  
    F_value_down = 2 

    df['middleband'] = df['close'].rolling(window=Bolling_timeperiod).mean() 
    std = df['close'].rolling(window=Bolling_timeperiod).std(ddof=0) # 指明自由度为0，这样计算标准差时分母为N，与TA-Lib一致，如果不这样设置，我们计算标准差的公式分母是N-1。
    df['upperband'] = df['middleband'] + std * F_value_up #分别计算上下布林带
    df['lowerband'] = df['middleband'] - std * F_value_down

    # ===找出做多信号
    condition1 = df['close'] < df['lowerband']  # 短期均线 > 长期均线
    condition2 = df['close'].shift(1) >  df['lowerband'].shift(1) # 上一周期的短期均线 <= 长期均线
    # 将产生做多信号的那根K线的signal设置为1，1代表做多
    df.loc[condition1 & condition2, 'signal'] = 1

    # ===找出做多平仓信号
    condition1 = df['close'] > df['upperband']  # 短期均线 < 长期均线
    condition2 = df['close'].shift(1) < df['upperband'].shift(1)  # 上一周期的短期均线 >= 长期均线
    df.loc[condition1 & condition2, 'signal'] = 0  # 将产生平仓信号当天的signal设置为0，0代表平仓

    # ===删除无关中间变量
    df.drop(['upperband', 'lowerband', 'middleband'], axis=1, inplace=True)
    return df

# 策略评价指标生成，除了第四行的函数名其他不用更改


def main_MA(df, c_rate, para_list):
    # 循环测试最好参数
    rtn = pd.DataFrame()
    for para in para_list:
        temp_df = bollinger_signal(
            df.copy(), para=para)  # 计算信号，需要用copy，防止改变df
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
        # rtn.loc[str(para), 'sharpe_ratio'] = temp['equity_curve'].pct_change(
        # ).mean()/temp['equity_curve'].pct_change().std()*np.sqrt(252)

        rtn.loc[str(para), 'base_return'] = base_return  # base就是股票自己的累积收益
    return rtn, temp_df

# 超参数调参
def tune_parameter(gold, bitcoin):
    # tune model parameters
    rtn1, _ = main_MA(gold, 1e-4, range(4, 100, 2))
    rtn1.index.name = 'parameter'
    rtn1.dropna(inplace=True)
    rtn1 = rtn1.sort_values(by='return/drawdown', ascending=False)
    rtn1.to_csv('./result/bollinger_signal_gold.csv', index=True, header=True)
    n1_gold = eval(rtn1.index[0])

    rtn2, _ = main_MA(bitcoin, 2e-4, range(4, 100, 2))
    rtn2.index.name = 'parameter'
    rtn2.dropna(inplace=True)
    rtn2 = rtn2.sort_values(by='return/drawdown', ascending=False)
    rtn2.to_csv('./result/bollinger_signal_bitcoin.csv', index=True, header=True)
    n1_bitcoin = eval(rtn2.index[0])
    return (n1_gold), (n1_bitcoin)


if __name__ == '__main__':
    # read the data
    gold, bitcoin = load_data()
    # get the best parameter
    (n1_gold), (n1_bitcoin) = tune_parameter(gold, bitcoin)
    # get the performance evaluation under the best performance
    result1, gold_best = main_MA(gold, 1e-4, [[n1_gold]])
    result1.index.name = 'parameter'
    result2, bitcoin_best = main_MA(bitcoin, 1e-4, [[n1_bitcoin]])
    result2.index.name = 'parameter'
    # save to csv file
    gold_best.to_csv("./result/bollinger_gold_best.csv", index=True, header=True)
    bitcoin_best.to_csv("./result/bollinger_bitcoin_best.csv", index=True, header=True)

    print("==================================================================")
    print(result1)
    print("==================================================================")
    print(result2)
    print("==================================================================")
