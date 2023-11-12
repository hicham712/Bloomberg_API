import numpy as np
from Portfolio_Strategy_Class import Config, Quote, Portf_Strategy, BLP
from datetime import datetime
from typing import List
import matplotlib.pyplot as plt
import pandas as pd


# Class that take in input a config, a portfolio strategy and will backtest it
class Backtester:
    def __init__(self, config: Config, port_strategy: Portf_Strategy):
        self._config = config
        self.portf_strategy = port_strategy
        self._calendar = port_strategy.lst_business_days
        self._universe = port_strategy.all_tickers
        self._timedelta = config.timedelta
        self.BLP = BLP()
        self._quote_by_pk = port_strategy.quote_by_pk
        self._weight_by_pk = port_strategy.weight_by_pk
        self.dict_tickers_by_ts = port_strategy.dict_tickers
        self._level_by_ts = dict()
        self.lst_historical_ptf_price = self.compute_levels()

        # collect the level of our portfolio factor and its datetime
        self.ptf_spot = [quote.close for quote in self._level_by_ts.values()]
        self.ts_ptf = [quote.ts_date for quote in self._level_by_ts.values()]
        self.ptf_return = np.diff(self.ptf_spot) / self.ptf_spot[:-1]
        # get risk free rate from our datetime
        self.risk_free_rate = self.get_rskfree_rate(self.ts_ptf[0], self.ts_ptf[-1])

        self.dict_output_factor = self.compute_dict_from_levels(self.ptf_spot)
        self.dict_output_index = self.compute_dict_from_levels(
            self.portf_strategy.df_mainindex_price.iloc[:, 0].tolist())

        # Plot all the necessary graphes
        self.df_global_output = pd.DataFrame([self.dict_output_factor, self.dict_output_index],
                                             index=[f" Factor : {self._config.strategy_type}", self._config.main_index])
        self.plot_drawdown()
        self.plot_rolling_var()
        # Generate or not the PORT file
        if self._config.bl_generate_outputPORT:
            self.generate_outputPORT()

        self.BLP.closeSession()

    # Method that create the output excel PORT fil for bloomberg
    def generate_outputPORT(self):
        i_row = 0
        df = pd.DataFrame(columns=["PORTFOLIO NAME", "SECURITY_ID", "Weight", "Date"])
        # Browse all the rebalnce date of the porfolio
        for ts in self.portf_strategy.lst_rebalance_date:
            # get the good universe from a current date
            ts_ = datetime.combine(ts, datetime.min.time(), )
            previous_compo_date = self.near_ts(ts_)
            universe = self.dict_tickers_by_ts[previous_compo_date]
            # Loop on stocks
            for underlying_code in universe:
                weight = self._weight_by_pk.get((self.portf_strategy.strategy_type, underlying_code,
                                                 ts))  # self.portf_strategy.business_timedelta(ts, -1)))
                if weight is not None:
                    row = {'PORTFOLIO NAME': f"Factor {self._config.strategy_type}_{self._config.optimize_method}",
                           'SECURITY_ID': underlying_code, 'Weight': weight.value, 'Date': ts}
                    # fill the row
                    df.loc[i_row] = row.values()
                    i_row += 1
        # Save
        self.dict_outputPORT = df.copy()
        self.dict_outputPORT.to_excel(f"PORT_Factor {self._config.strategy_type}_{self._config.optimize_method}.xlsx")

    def compute_dict_from_levels(self, hist_quotes_level):
        """
          Calculates various performance metrics for a portfolio of assets based on a list of quotes.
          Returns a dictionary containing the following metrics:
          - overall_return: overall performance of the portfolio
          - annualized_return: annualized performance of the portfolio
          - daily_volatility: daily volatility of the portfolio
          - monthly_volatility: monthly volatility of the portfolio
          - annualized_volatility: annualized volatility of the portfolio
          - sharpe_ratio: Sharpe ratio of the portfolio
          - max_drawdown: maximum drawdown of the portfolio
          - var_95: historical VaR at 95% confidence level
          """
        # Calculate returns based on the quotes
        returns = np.diff(hist_quotes_level) / hist_quotes_level[:-1]

        # Calculate overall performance of the portfolio
        overall_return = returns[-1]

        # Calculate annualized performance of the portfolio
        num_periods = len(returns)
        annualized_return = np.round((1 + overall_return) ** (250 / num_periods) - 1, 5)

        # Calculate daily, monthly and annualized volatility of the portfolio
        daily_volatility = np.round(np.std(returns, ddof=1), 5)
        monthly_volatility = np.round(daily_volatility * np.sqrt(21), 5)
        annualized_volatility = np.round(daily_volatility * np.sqrt(252), 5)

        return_rsk_free = self.risk_free_rate.pct_change().dropna().values.mean()
        # Calculate Sharpe ratio of the portfolio
        excess_returns = returns - return_rsk_free
        sharpe_ratio = np.round(np.mean(excess_returns) / np.std(excess_returns, ddof=1), 5)

        # Calculate maximum drawdown of the portfolio
        max_drawdown = (np.maximum.accumulate(hist_quotes_level) - hist_quotes_level) / np.maximum.accumulate(
            hist_quotes_level)
        max_drawdown = np.round(np.max(max_drawdown), 5)

        # Calculate historical VaR at 95% confidence level
        var_95 = np.percentile(returns, 5)

        # Return results as a dictionary
        dict_results = {
            'overall_return': overall_return,
            'annualized_return': annualized_return,
            'daily_volatility': daily_volatility,
            'monthly_volatility': monthly_volatility,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95
        }
        return dict_results

    def plot_drawdown(self, ):
        """
        Plot the historical drawdown of a pandas series.

        Args:
        serie (pandas.Series): A pandas series.
        """
        hist_return = self.ptf_return
        cum_returns = np.cumprod(1 + hist_return) - 1
        rolling_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - rolling_max) / (rolling_max + 1e-16)

        # Create a PLT figure with a line chart of the drawdown
        fig, ax = plt.subplots()
        ax.plot(self.ts_ptf[:len(drawdown)], drawdown)
        ax.set(title='Historical Drawdown', xlabel='Time', ylabel='Drawdown')
        # plt.show()
        plt.gcf().autofmt_xdate()
        plt.savefig("Drawdown of the factor")

    def plot_rolling_var(self, window=252):
        """
        Plot the rolling value at risk (VaR) of a pandas series of returns.

        Args:
        returns (pandas.Series): A pandas series of returns.
        window (int): The size of the rolling window. Default is 252 (1 year).

        Returns:
        None.
        """
        returns = self.ptf_return
        if len(returns) < window:
            raise ValueError(
                f'The time between the start date and the end date is too small to plot a historical Value at risk with {window} window')

        var_95 = np.percentile(returns, 5)
        rolling_var = []
        for int_i in range(window, len(returns)):
            current_returns = returns[(int_i - window):int_i]
            rolling_var.append(np.percentile(current_returns, 5))

        fig, ax = plt.subplots()
        ax.plot(self.ts_ptf[:-1], returns,
                label=f'Returns of {self._config.strategy_type} factor & with a {self._config.optimize_method} optimization method')
        ax.plot(self.ts_ptf[window:-1], rolling_var, linestyle='--', color='red',
                label=f"Rolling var - {window} day window")

        ax.axhline(y=var_95, xmin=0, xmax=len(returns), c='yellow', linewidth=4, zorder=0, label="Historical VAR")
        ax.set(title=f'Rolling VaR (95%) - {window} day window', xlabel='Time', ylabel='Returns')
        ax.annotate(f"VaR (95%): {var_95:.2%}", xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top')
        plt.gcf().autofmt_xdate()
        plt.legend()

        plt.savefig("Value At risk 95%")

    def _compute_perf(self, ts: datetime) -> float:
        perf_ = 0.
        previous_rebalancing_date = self.near_ts(ts)
        universe = self.dict_tickers_by_ts[previous_rebalancing_date]
        for underlying_code in universe:
            weight = self._weight_by_pk.get(
                (self.portf_strategy.strategy_type, underlying_code, self.portf_strategy.business_timedelta(ts, -1)))
            if weight is not None:
                value = weight.value
                current_quote = self._quote_by_pk.get((underlying_code, self.portf_strategy.business_timedelta(ts, -1)))
                previous_quote = self._quote_by_pk.get(
                    (underlying_code, self.portf_strategy.business_timedelta(ts, -2)))

                if current_quote is not None and previous_quote is not None:
                    perf_ += value * (current_quote.close / previous_quote.close - 1)
                else:
                    raise ValueError(f'missing quote for {underlying_code} at {ts - self._timedelta} '
                                     f'or {ts - self._timedelta * 2}')
        return perf_

    def compute_levels(self) -> List[Quote]:
        for ts in self._calendar:
            # self._compute_weight(ts)
            ts_date_previous = self.portf_strategy.business_timedelta(ts, -1)
            if ts == self._config.start_ts or ts == self._calendar[0]:
                quote = Quote(close=self._config.basis, ts_date=ts_date_previous)

                self._level_by_ts[ts_date_previous] = quote
            else:
                perf = self._compute_perf(ts)
                close = self._level_by_ts.get(self.portf_strategy.business_timedelta(ts, -2)).close * (1 + perf)
                quote = Quote(close=close, ts_date=ts)
                self._level_by_ts[ts_date_previous] = quote

        return list(self._level_by_ts.values())

    def plot_levels(self):
        """
        Plot the levels of the portfolio
        :return: a graphe of the perf of the strategy (without rebalancing it)
        """

        if self._config.bl_index_comparaison:
            plt.figure(
                f'Backtest: {self._config.strategy_type} Strategy & with a {self._config.optimize_method} optimization method')
            plt.plot(self.ts_ptf, self.ptf_spot, color='r', label=f"Factor {self._config.strategy_type}")
            plt.plot(self.ts_ptf, self.portf_strategy.df_mainindex_price, color='g', label=self._config.main_index)
            plt.gcf().autofmt_xdate()
            plt.legend()
            plt.savefig(f"Backtest_{self._config.strategy_type}_{self._config.optimize_method}")
            plt.show()

        else:
            plt.figure(
                f'Backtest: {self._config.strategy_type} Strategy & with a {self._config.optimize_method} optimization method')
            plt.plot(self.ts_ptf, self.ptf_spot)
            plt.gcf().autofmt_xdate()
            plt.savefig(f"Backtest_{self._config.strategy_type}_{self._config.optimize_method}")
            plt.show()

    # method that get the risk free rate in order to do our analysis
    def get_rskfree_rate(self, dt_start, dt_end) -> pd.DataFrame:
        bdh_prices = self.BLP.bdh(strSecurity=self._config.rsk_free_rate, strFields="PX_LAST", startdate=dt_start,
                                  enddate=dt_end, )

        bdh_prices = bdh_prices["PX_LAST"]
        return bdh_prices

    # method that find the closest composition of the index from a current datetime
    def near_ts(self, df_current: datetime):
        """
        Find nearest balancing datetime
        """
        dt_current_ = df_current
        while dt_current_ not in self.dict_tickers_by_ts.keys():
            dt_current_ -= self._timedelta
        return dt_current_


if __name__ == '__main__':
    """
    now = datetime.now(tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    conf = Config(
        start_ts=now - timedelta(days=450),
        end_ts=now - timedelta(days=7),
        optimize_method=Optimize_method.DIVERSIFY,
        main_index="SX5E Index",
        strategy_type=Strategy_type.VALUE,
        bl_index_comparaison=True,
        rebalance_period = 30,
        int_check_index = 60,
        bl_generate_outputPORT= False,
    )
    a = ["PX_LAST", "CUR_MKT_CAP", "PX_TO_BOOK_RATIO"]
    strat = Portf_Strategy(conf, )
    backtest = Backtester(conf, strat)
    backtest.plot_levels()
    output = backtest.df_global_output
    output.to_excel("Factor and Index performance.xlsx")
    """
