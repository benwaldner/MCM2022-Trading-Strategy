from typing import Dict
from cvxopt import matrix
import cvxopt
from cvxopt.solvers import lp, qp
import quantstats as qs
import pandas as pd
import numpy as np
import warnings

cvxopt.solvers.options['show_progress'] = False

def execute_transactions(pre_bitcoin, pre_gold, cur_bitcoin, cur_gold, pre_amount, a_g, a_b,
        gold_price):
    assert not np.isnan(gold_price) or pre_gold == cur_gold, "Gold cannot be traded in this day"
    pre_bitcoin_amount = pre_amount * pre_bitcoin
    pre_gold_amount = pre_amount * pre_gold
    pre_cash_amount = pre_amount - pre_bitcoin_amount - pre_gold_amount
    # no change in portfolio
    if pre_bitcoin == cur_bitcoin and pre_gold == cur_gold:
        return pre_bitcoin_amount, pre_gold_amount, pre_cash_amount
    # need to do transaction
    c = np.array([0., 0., 0., 0., -1.])
    G = -np.identity(5)
    h = np.zeros((5, 1))
    A = np.array([
        [-1, 1/(1+a_b), 0, 0, -cur_bitcoin],
        [0, 0, -1, 1/(1+a_g), -cur_gold],
        [1-a_b, -1, 1-a_g, -1, -(1-cur_bitcoin-cur_gold)]
    ])
    b = -np.array([
        [pre_bitcoin_amount],
        [pre_gold_amount],
        [pre_cash_amount]
    ])

    c = matrix(c)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix(b)

    res = lp(c, G, h, A, b)
    solution = np.array(res['x']).reshape(-1)
    if len(solution) == 1:
        import pdb; pdb.set_trace()
    bit2cash, cash2bit, gold2cash, cash2gold, cur_amount = solution.tolist()
    if res['status'] != 'optimal':
        warnings.warn('LP in transaction not working as expected.', UserWarning)
        bit2cash = np.clip(bit2cash, 0, pre_bitcoin_amount)
        cash2bit = np.clip(cash2bit, 0, pre_cash_amount)
        gold2cash = np.clip(gold2cash, 0, pre_gold_amount)
        cash2gold = np.clip(cash2gold, 0, pre_cash_amount)
    if np.isnan(gold_price):
        gold2cash = 0
        cash2gold = 0
    cur_bitcoin_amount = pre_bitcoin_amount + cash2bit/(1+a_b) - bit2cash
    cur_gold_amount = pre_gold_amount + cash2gold/(1+a_g) - gold2cash
    cur_cash_amount = pre_cash_amount - cash2bit - cash2gold + bit2cash*(1-a_b) + gold2cash*(1-a_g)
    # cur_amount = solution.reshape(-1)[-1]
    # cur_bitcoin_amount = cur_amount * cur_bitcoin
    # cur_gold_amount = cur_amount * cur_gold
    # cur_cash_amount = cur_amount - cur_bitcoin_amount - cur_gold_amount
    return cur_bitcoin_amount, cur_gold_amount, cur_cash_amount

def backtest(data: pd.DataFrame, init_amount, a_g=1e-4, a_b=2e-4):
    """```data``` contains the proportion of each asset
    every day.
    """
    data = data.copy()
    data = data.sort_index()
    data.reset_index(drop=True)
    amount = [init_amount]
    pre_gold_price = data.iloc[0]['gold']
    commission = []
    for i in range(1, data.shape[0]):
        pre_bitcoin = data.iloc[i-1]['prop_bitcoin']
        pre_gold = data.iloc[i-1]['prop_gold']
        cur_bitcoin = data.iloc[i]['prop_bitcoin']
        cur_gold = data.iloc[i]['prop_gold']
        pre_bitcoin_amount, pre_gold_amount, pre_cash_amount = \
             execute_transactions(pre_bitcoin, pre_gold, 
                                  cur_bitcoin, cur_gold,
                                  amount[-1], a_g, a_b,
                                  data.iloc[i-1]['gold'])
        commission_fee = amount[-1] - (pre_bitcoin_amount+pre_gold_amount+pre_cash_amount)
        commission.append(commission_fee)
        # if pre_gold_amount < -1e-3 or pre_bitcoin_amount < -1e-3:
        #     import pdb; pdb.set_trace()
        #     print(pre_bitcoin_amount, pre_gold_amount, pre_cash_amount)
        #     pre_bitcoin_amount, pre_gold_amount, pre_cash_amount = \
        #         execute_transactions(pre_bitcoin, pre_gold, 
        #                             cur_bitcoin, cur_gold,
        #                             amount[-1], a_g, a_b,
        #                             data.iloc[i-1]['gold'])
        cur_bitcoin_amount = data.iloc[i]['bitcoin']/data.iloc[i-1]['bitcoin'] \
                             * pre_bitcoin_amount
        if np.isnan(pre_gold_price):
            # the first day of the period is not a trading day
            cur_gold_amount = pre_gold_amount
            if not np.isnan(data.iloc[i]['gold']):
                pre_gold_price = data.iloc[i]['gold']
        elif np.isnan(data.iloc[i]['gold']):
            # today is not a trading day
            cur_gold_amount = pre_gold_amount
        else:
            # today is a trading day and the last available gold price is known
            cur_gold_amount = data.iloc[i]['gold']/pre_gold_price \
                                * pre_gold_amount
            pre_gold_price = data.iloc[i]['gold']
        cur_cash_amount = pre_cash_amount
        cur_amount = cur_bitcoin_amount + cur_gold_amount + cur_cash_amount
        amount.append(cur_amount)
    commission.append(0)   # the last day
    data['value'] = amount
    return data, commission

def get_proportion_large(mu, cov, gamma=1):
    num = mu.size
    P = gamma * cov
    q = -mu
    G = - np.identity(num)
    h = np.zeros((num, 1))
    A = np.ones((1, num))
    b = np.ones((1, 1))
    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix(b)

    res = qp(P, q, G, h, A, b)
    x = res['x']
    x = np.array(x).reshape(-1)
    assert abs(x.sum() - 1) < 1e-5, f"{x.sum()} != 1"
    return x

def get_returns_statistic(data: pd.DataFrame):
    """return mean, co-variance"""
    data = data.copy()
    ret : pd.DataFrame = data / data.shift(1) - 1
    ret['cash'] = 0
    ret = ret.dropna()
    mu = ret.mean()
    cov = ret.cov()
    return mu.to_numpy(), cov.to_numpy()

def quantize(portfolio, total=1000):
    quantized_portfolio = [int(x*total) for x in portfolio]
    quantized_portfolio[2] = total - quantized_portfolio[0] - quantized_portfolio[1]
    quantized_portfolio = [x/total for x in quantized_portfolio]
    return quantized_portfolio

def mean_variance_sliding_portfolio_large(data, gamma, init_mu, init_cov, update_period=1, alpha=1, period_len=None):
    portfolios = [] # strategies
    portfolio = get_proportion_large(init_mu, init_cov, gamma).tolist()
    portfolios.append(portfolio)
    for i in range(1, data.shape[0]):
        if i % update_period == 0:
            # update the porfolio
            start = 0 if period_len is None or i-period_len < 0 else i-period_len
            mu, cov = get_returns_statistic(data.iloc[start:i])
            # update init mu, cov
            init_mu = init_mu*(1-alpha) + alpha*mu if not np.isnan(mu).any() else init_mu
            init_cov = init_cov*(1-alpha) + alpha*cov if not np.isnan(cov).any() else init_cov
            portfolio = get_proportion_large(init_mu, init_cov, gamma).tolist()
        portfolios.append(portfolio)
    portf = pd.DataFrame(portfolios)
    portf.columns = data.columns
    return portf

def convert_strategies_to_components(data: pd.DataFrame, portfolios: pd.DataFrame, portfolio_info: Dict[str, pd.DataFrame]):

    def convert(s: pd.Series):
        weights = s.to_numpy()
        names = list(s.index)
        day = s.name
        info = pd.concat([portfolio_info[key].iloc[day] for key in names], axis=1).transpose()
        components = weights.reshape(1, -1) @ info.to_numpy()
        components = pd.DataFrame(components)
        components.columns = info.columns
        return components.iloc[0]

    # convert to components and post process for trading days and quantized
    components = portfolios.apply(convert, 1)
    # adjust portfolios for trading days
    components.loc[0] = 0
    components.loc[0, 'cash'] = 1
    for i in range(1, components.shape[0]):
        if np.isnan(data.iloc[i-1]['gold']):
            # yesterday is not a trading day
            # whether cash is enough
            if components.loc[i-1, 'gold'] - components.loc[i, 'gold'] > components.loc[i, 'cash']:
                # cash not enough
                components.loc[i, 'bitcoin'] = 1 - components.loc[i-1, 'gold']
            components.loc[i, 'gold'] = components.loc[i-1, 'gold']
    components['cash'] = 1 - components['bitcoin'] - components['gold']
    # quantize
    columns = components.columns
    index = components.index
    data = components.to_numpy().tolist()
    q_data = [quantize(d) for d in data]
    output = pd.DataFrame(q_data)
    output.columns = [f'prop_{k}' for k in columns]
    output.index = index
    return output

def mean_variance_sliding_result_large(data, portfolio_info, gamma, update_period, alpha=0.5, 
        period_len=None, plot=True, init_mu=None, init_cov=None, return_portfolio_weight=False,
        benchmark=None, return_commission=False):
    """
    Construct a portfolio based on mean-variance optimization and return the result on data.

    Args:
        * ```data```: a DataFrame which contains all available strategies' net value.
        * ```portfolio_info```: Dict[pd.DataFrame], a dictionary of DataFrames. The keys of
            the dictionary are the names of the strategies, which are also the keys in ```data```.
            Each DataFrame contains proportion of each component at each day.
        * ```gamma```: tolerance of risk. Larger ```gamma``` means more risk adversed.
        * ```update_period```: number of days before reweighting the portfolio.
        * ```alpha```: a internal parameter for updating the estimated mean and covariance (between 0 and 1).
        * ```period_len```: length of the period that is used to estimate the mean and covariance.
        * ```plot```: whether to generate a report using QuantStat.
        * ```init_mu```: np.ndarray, an initial guess of the expected daily return for each strategy.
        * ```init_cov```: np.ndarray, an initial guess of the covariance of the daily return for each strategy.
        * ```return_commission```: whether to return the commission fee also.
    
    Return: a DataFrame containing the net value of the portfolio.

    Reminder: ```data``` should have a column of all zero value with the column name 'cash'.
    """
    num = data.shape[1] # number of available strategies
    if init_mu is None:
        init_mu = np.random.normal(0.001, 0.001, num)
    if init_cov is None:
        init_cov = np.diag(np.random.normal(1e-4, 1e-5, num)) + np.random.normal(5e-5, 1e-5, (num, num))
        init_cov = np.abs(init_cov)
    data = data.copy()
    portfolio = mean_variance_sliding_portfolio_large(data, gamma, init_mu, init_cov, update_period, alpha, period_len)
    component = convert_strategies_to_components(data, portfolio, portfolio_info)
    component.index = data.index
    data = pd.concat([data, component], axis=1)
    result, commission = backtest(data, 1000)
    result.index = data.index # date for qs report
    # display(result)
    ret = (result['value'] / result['value'].shift(1)).dropna() - 1
    print(f'Mean return {ret.mean()}, Std {ret.std()}')
    if plot:
        if benchmark:
            qs.reports.html(result['value'], result[benchmark], output='report.html')
        else:
            qs.reports.html(result['value'], output='report.html')
    if return_commission:
        result['commission'] = commission
    if return_portfolio_weight:
        return result, portfolio
    return result



# example
# load data
def read_data(path, name):
    df = pd.read_csv(path)
    df.columns = ['date', name]
    df['date'] = pd.to_datetime(df['date'])
    return df

def load_raw_data():
    gold = read_data('LBMA-GOLD.csv', 'gold')
    bit = read_data('BCHAIN-MKPRU.csv', 'bitcoin')

    data = pd.merge(bit, gold, how='outer', on='date')
    return data


if __name__ == '__main__':
    data = load_raw_data()
    data.index = data.pop('date')
    data['cash'] = 0
    portfolio_info = {
        'bitcoin': pd.DataFrame([[1, 0, 0]]*data.shape[0], columns=['bitcoin', 'gold', 'cash']),
        'gold': pd.DataFrame([[0, 1, 0]]*data.shape[0], columns=['bitcoin', 'gold', 'cash']),
        'cash': pd.DataFrame([[0, 0, 1]]*data.shape[0], columns=['bitcoin', 'gold', 'cash']),
    }
    # run portfolio
    result, portfolio = mean_variance_sliding_result_large(data, portfolio_info, 1, 20, 0.6, 50, True, 
                                                           return_portfolio_weight=True,
                                                           benchmark='bitcoin')
    print(result)
    print(portfolio)