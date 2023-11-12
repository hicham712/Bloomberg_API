import pandas as pd
import numpy as np
from Blp_Class import BLP
from datetime import datetime, timezone, timedelta
from scipy.optimize import minimize
from dataclasses import dataclass
from enum import Enum
from typing import List
from sklearn.covariance import ShrunkCovariance

# ------------- Creation of different enum class ---------------------------------
class Strategy_type(Enum):
    VALUE = "Value"
    MOMENTUM = "Momentum"
    SIZE = "Size"

class Index_names(Enum):
    SPX = "SPX Index"
    CAC = "CAC Index"
    SX5E = "SX5E Index"
    DAX = "DAX Index"
    FTSE = "UKX Index"
    NDX = "NDX Index"

@dataclass
class Optimize_method(Enum):
    SHARPE = "Sharpe_ratio"
    MINVVAR = "Min_var"
    DIVERSIFY = "Diversify"


@dataclass
class Quote:
    symbol: str = None
    close: float = None
    ts_date: datetime.date = None


@dataclass
class Weight:
    underlying_code: str = None
    ts: datetime = None
    value: float = None

# Creation of the config class that will take all the user inputs
@dataclass
class Config:
    start_ts: datetime
    end_ts: datetime
    strategy_type: Strategy_type = Strategy_type.MOMENTUM
    rebalance_period: int = 20
    int_check_index: int = 60
    main_index: Index_names = Index_names.SX5E
    rsk_free_rate: str = "USGG10YR  Index"
    basis: int = 100
    current_ts: datetime = None
    optimize_method: Optimize_method = Optimize_method.MINVVAR
    bl_index_comparaison: bool = True
    bl_generate_outputPORT: bool = True

    def __post_init__(self):
        if self.start_ts >= self.end_ts:
            raise ValueError("self.start_ts must be before self.end_ts")
        # if len(self.universe) == 0:
        #    raise ValueError("self.universe should contains at least one element")
        if self.end_ts > datetime.now().replace(hour=0, minute=0, second=0,):
            raise ValueError("End date cannot be superior to today")
        self.current_ts = self.start_ts
        self.int_total_days = (self.end_ts - self.start_ts).days
        self.int_secure_time_ = max(self.rebalance_period, self.int_check_index)*2

    @property
    def timedelta(self):
        return timedelta(days=1)

    def next_business_nday(self, n_day: int, holidays: list = []):
        business_days_to_add = n_day
        current_date = self.current_ts
        timedelta_ = self.timedelta
        while business_days_to_add > 0:
            current_date += timedelta_
            weekday = current_date.weekday()
            if weekday >= 5:  # sunday = 6
                continue
            if current_date in holidays:
                continue
            business_days_to_add -= 1
        return current_date

    def calendar(self, back_ward_time: int = 0, bl_secure_time:bool=False) -> List[datetime]:
        # renvoyer une liste de date comprise entre start_ts et end_ts.
        timedelta_ = self.timedelta
        tmp = self.start_ts - (timedelta_ * (back_ward_time + 1))
        # Prise en compte d'un coussin de sécurité pour les période de rebalncement
        if bl_secure_time:
            int_secure_time_ = self.int_secure_time_
        else:
            int_secure_time_ = 0
        end_ts = self.end_ts + int_secure_time_ * timedelta_
        self.current_ts = tmp
        calendar_ = []
        while tmp < end_ts:
            tmp = self.next_business_nday(1)
            calendar_.append(tmp)
            self.current_ts = tmp
        return calendar_


# Class Portfolio strategy of our factor
class Portf_Strategy:
    def __init__(self, config: Config, blp_: BLP = BLP):
        # Retrieve all the datas from the config
        self.config = config
        self.strategy_type = config.strategy_type
        self.BLP = blp_()
        self.optimize_method = config.optimize_method
        self.start_date = config.start_ts
        self.end_date = config.end_ts
        self.rebalancing_period = config.rebalance_period
        self.main_index = config.main_index
        self.lst_business_days = config.calendar()
        # containing 252 business days before the start_date
        self.lst_all_business_days = config.calendar(config.int_total_days + 380, True)
        # Construction of the dict that contains all the index composition for each datetime (in keys)
        self.dict_tickers = self.create_dict_ticker()
        # get all the tickers that we will need to import datas
        self.all_tickers = self.collect_all_tickers()
        self.dct_tickers_data = self.BLP.bdh(strSecurity=self.all_tickers,
                                             strFields=["PX_LAST", "CUR_MKT_CAP", "TOT_COMMON_EQY"]
                                             , startdate=self.lst_all_business_days[0],
                                             enddate=self.lst_all_business_days[-1], )
        self._quote_by_pk = {}
        self.fill_quote()
        self.df_mainindex_price: pd.DataFrame = None
        # Compute the strategy
        # methode that compute the weigh of its strategy
        self._weight_by_pk = dict()
        self.lst_rebalance_date = self.create_rebalancing_list()
        # Final method that fill _weight_by_pk
        self.compute_weights()
        if config.bl_index_comparaison:
            self.create_index_historic()
        self.BLP.closeSession()

    # ------------------------------COMPUTE THE STRATEGY-------------------------------------------------------------------

    def compute_weights(self):
        lst_rebalance_date = self.lst_rebalance_date
        for dt_date in self.lst_business_days:
            self.get_signals(dt_date)
            self.dict_long_ptf = self.compute_returns_cov(self.lst_to_long, dt_date)
            self.dict_short_ptf = self.compute_returns_cov(self.lst_to_short, dt_date)

            # Check if it is a rebalncing date
            if dt_date.date() in lst_rebalance_date:
                bl_rebalance = True
                # optimize and fill
                optimized_result = self.opti_ptf(self.dict_long_ptf,)
                self.fill_weight_by_pk(optimized_result, dt_date, self.dict_long_ptf, True, bl_rebalance, )
                optimized_result = self.opti_ptf(self.dict_short_ptf,)
                self.fill_weight_by_pk(optimized_result, dt_date, self.dict_short_ptf, False, bl_rebalance, )
            else:
                bl_rebalance = False
                # optimize and fill
                self.fill_weight_by_pk(optimized_result, dt_date, self.dict_long_ptf, True, bl_rebalance, )
                self.fill_weight_by_pk(optimized_result, dt_date, self.dict_short_ptf, False, bl_rebalance, )

    def fill_weight_by_pk(self, optimized_result, dt_date, dict_ptf, bool_long: bool, bl_optimize: bool, ):
        int_weight = 0
        optimized_result_ = optimized_result
        # if we have to optimze this date
        if bl_optimize:
            for ticker in dict_ptf['Returns'].keys():
                if bool_long:
                    self._weight_by_pk[(self.strategy_type, ticker, dt_date.date())] = Weight(ticker, dt_date, round(
                        optimized_result_[int_weight], 4))
                else:
                    self._weight_by_pk[(self.strategy_type, ticker, dt_date.date())] = Weight(ticker, dt_date, round(
                        - optimized_result_[int_weight], 4))
                int_weight += 1
        # take the same weight as before
        else:
            previous_date = self.business_timedelta(dt_date, -1)
            for ticker in dict_ptf['Returns'].keys():
                self._weight_by_pk[(self.strategy_type, ticker, dt_date.date())] = self._weight_by_pk.get(
                    (self.strategy_type, ticker, previous_date))

    def opti_ptf(self, dict_ptf: dict):

        equal_weightss = np.full((len(dict_ptf['Covariance']), 1), 1 / len(dict_ptf['Covariance']))
        np_covv = np.array(dict_ptf['Covariance']).round(5)
        bound = tuple((0.01, 1) for w in equal_weightss)
        constraint = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

        np_daily_returns = np.array(dict_ptf['Returns'])
        # Optimization
        if self.optimize_method == Optimize_method.MINVVAR.value:
            optimized_result = minimize(self.portfolio_sd, x0=equal_weightss, args=np_covv, method='SLSQP',
                                        bounds=bound, constraints=constraint)
        if self.optimize_method == Optimize_method.SHARPE.value:
            optimized_result = minimize(self.sharpe_ratio, x0=equal_weightss, args=(np_covv, np_daily_returns),
                                        method='SLSQP', bounds=bound, constraints=constraint)
        if self.optimize_method == Optimize_method.DIVERSIFY.value:
            optimized_result = minimize(self.diversify_ratio, x0=equal_weightss, args=np_covv,
                                        method='SLSQP', bounds=bound, constraints=constraint)

        return optimized_result['x']


    def get_signals(self, dt_date: datetime):
        """
        Create dataframes containing the signals Momentum, Value and Size

        Returns
        -------
        List containing list of tickers referencing to signals.

        """
        # df to recup concidering signal
        dt_date_ = dt_date.date()
        # implement the SIZE strategy
        if self.strategy_type == Strategy_type.SIZE.value:
            df_signal = self.dct_tickers_data['CUR_MKT_CAP'].loc[dt_date_]  # CUR_MKT_CAP = MKT_Cap
        # implement the VALUE strategy
        elif self.strategy_type == Strategy_type.VALUE.value:
            df_mkt_cap = self.dct_tickers_data['TOT_COMMON_EQY'].loc[dt_date_]  # TOT_COMMON_EQY = BOOK_Value
            df_size = self.dct_tickers_data['CUR_MKT_CAP'].loc[dt_date_]
            df_signal = df_mkt_cap.div(df_size)
        # implement the MOMENTUM strategy
        elif self.strategy_type == Strategy_type.MOMENTUM.value:
            df_price = self.dct_tickers_data['PX_LAST']
            dt_date_21 = self.business_timedelta(dt_date, -20)
            dt_date_250 = self.business_timedelta(dt_date, -250)
            df_signal = (df_price.loc[dt_date_21] / df_price.loc[dt_date_250]) - 1

        # List to long or to short choosen by first quartile and last quartile
        df_signal.dropna()
        lst_sorted = df_signal.sort_values().index
        frst_qtl = round(len(lst_sorted) * 25 / 100)
        last_qtl = round(len(lst_sorted) * 75 / 100)

        self.lst_to_long = lst_sorted[0:frst_qtl]
        self.lst_to_short = lst_sorted[last_qtl:]

    def compute_returns_cov(self, lst_tickers: list, dt_date: datetime):
        """
        Method that compute the cov and the return for a given list of ticker at a given date
        """
        all_df = self.dct_tickers_data['PX_LAST'][lst_tickers]
        df_price = all_df.loc[self.business_timedelta(dt_date, (-self.rebalancing_period)):dt_date.date()]
        df_price = df_price.dropna(axis='columns')
        # exception, ne doit pas se produire
        if df_price.empty:
            print(f"An error lead to security event has occured for this date {dt_date}")
            df_price = all_df.loc[dt_date.date(): self.business_timedelta(dt_date, self.rebalancing_period)]
            df_price = df_price.dropna(axis='columns')

        df_returns = df_price.pct_change().dropna(axis=0, how='any', inplace=False, )
        df_daily_returns = (1+df_returns.mean()) ** 252 - 1
        df_daily_cov = ShrunkCovariance().fit(df_returns).covariance_ * 252

        return {'Returns': df_daily_returns, 'Covariance': df_daily_cov}

    def fill_quote(self):
        """
        Method that import all the datas and fill a dictionary with quotes
        """
        df_price = self.dct_tickers_data["PX_LAST"]
        for ts_date in df_price.index:
            for ticker in df_price.columns:
                self._quote_by_pk[ticker, ts_date] = Quote(symbol=ticker, close=df_price.loc[ts_date, ticker],
                                                           ts_date=ts_date)
    # ----------------- Optimization statics method------------------------------------
    @staticmethod
    def portfolio_sd(np_weights, np_cov):
        # result = np.transpose(np_weights) @ (np_cov) @ np_weights
        result = np.dot(np_weights, np.dot(np_cov, np_weights))
        return np.sqrt(result)

    @staticmethod
    def sharpe_ratio(np_weights, np_cov, np_daily_returns):
        return -(np.transpose(np_weights) @ np_daily_returns) / (
            np.sqrt(np.transpose(np_weights) @ (np_cov) @ (np_weights)))

    @staticmethod
    def diversify_ratio(np_weights, np_cov):
        # vol moyennenisé par les poids
        weighted_vol = np.sqrt(np.diag(np_cov) @ np_weights.T)
        # portfolio vol
        port_vol = np.sqrt(np_weights.T @ np_cov @ np_weights)
        diversification_ratio = weighted_vol / port_vol
        # return negative for minimization problem
        return -diversification_ratio


    # ------------COMPUTE DATAS-------------------------------------------------------------------------------------

    def create_dict_ticker(self):
        """
        For each balancing period / BDS to calculate the ticker from the index / fill lst_ticker
        """
        lst_business_days = self.lst_business_days
        int_check_index = self.config.int_check_index
        dict_tickers = {}
        int_date = 0
        while int_date < len(self.lst_business_days):
            lst_tmp_ticker = self.ticker_from_date(lst_business_days[int_date])
            # add " Equity" to all tickers
            lst_tmp_ticker = list(map(lambda lambda1: lambda1 + " Equity", lst_tmp_ticker))
            # fill output
            dict_tickers[lst_business_days[int_date]] = lst_tmp_ticker
            int_date += int_check_index

        return dict_tickers
    # BDS composition for dates
    def ticker_from_date(self, dt_date: datetime) -> list:  # A TESTE
        str_date = dt_date.strftime('%Y%m%d')
        bds_data = self.BLP.bds(strSecurity=self.main_index, strFields="INDX_MWEIGHT_HIST",
                                strOverrideField="END_DATE_OVERRIDE",
                                strOverrideValue=str_date)
        aa = list(bds_data.keys())
        return aa

    def collect_all_tickers(self):
        """
        check all the tickers that are unique and we want to import
        """
        lst_all_tickers = []
        for dt_date, lst_ticker in self.dict_tickers.items():
            lst_all_tickers += lst_ticker

        lst_all_tickers = list(np.unique(lst_all_tickers))

        return lst_all_tickers

    def business_timedelta(self, current_date: datetime, int_business_days: int = 1):
        """
        Function that allow us to move from a date to another by counting the busines days
        """
        lst_business_days = self.lst_all_business_days
        if current_date in lst_business_days:
            index_current_date = lst_business_days.index(current_date)
            index_date = index_current_date + int_business_days
            if index_date < 0:
                raise ValueError(f"You have to import more dates, index value = {index_date}")
            return lst_business_days[index_date].date()
        else:
            raise ValueError(f" {current_date} is not in the list of business days")

    def create_index_historic(self) -> pd.DataFrame:
        """
        BDH to get all the datas
        """
        bdh_prices = self.BLP.bdh(strSecurity=self.main_index, strFields="PX_LAST", startdate=self.lst_business_days[0],
                                  enddate=self.lst_business_days[-1], )

        bdh_prices = bdh_prices["PX_LAST"]
        self.df_mainindex_price = bdh_prices / bdh_prices.iloc[0, 0] * 100

    def create_rebalancing_list(self):
        dt_current_date = self.lst_business_days[0].date()
        dt_end = self.lst_business_days[-1].date()
        lst_rebalanced_dates = []
        while dt_current_date < dt_end:
            lst_rebalanced_dates.append(dt_current_date)
            dt_current_date = self.business_timedelta(
                datetime.combine(dt_current_date, datetime.min.time(), ), # tzinfo=timezone.utc
                self.config.rebalance_period)

        return lst_rebalanced_dates

    @property
    def weight_by_pk(self):
        return self._weight_by_pk

    @property
    def quote_by_pk(self):
        return self._quote_by_pk


if __name__ == '__main__':
    now = datetime.now(tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    conf = Config(
        start_ts=now - timedelta(days=365),
        end_ts=now - timedelta(days=7),
        optimize_method=Optimize_method.MINVVAR,
        main_index="SX5E Index",
        strategy_type=Strategy_type.SIZE,
        bl_index_comparaison=True,
    )
    a = ["PX_LAST", "CUR_MKT_CAP", "PX_TO_BOOK_RATIO"]
    strat = Portf_Strategy(conf, )
    df = strat.dct_tickers_data
    # book value / market cap = facteur Value
    # MOmentum = le prix à 20 jours vs le prix à 250 jours
    # size =  market cap

    weight_pk = strat.weight_by_pk
    print(weight_pk)

    # Quote_by_pk  / self._quote_by_pk.get((underlying_code, ts - self._timedelta))
    #
    """
    # Save for Hicham
    df["PX_LAST"].to_excel("Price_Histo.xlsx")
    df["CUR_MKT_CAP"].to_excel("MKT_CAP_Histo.xlsx")
    df["TOT_COMMON_EQY"].to_excel("TOT_COMMON_EQY.xlsx")
    all_tick = strat.dict_tickers
    dff = pd.DataFrame.from_dict(all_tick)
    dff.columns = dff.columns.strftime('%Y-%m-%d')
    dff.to_excel("dict_ticker_time.xlsx")

    blp = BLP()
    strFields = ["PX_LAST"]
    tickers = ["ATO FP Equity", "TTE FP Equity"]
    startDate = dt.datetime(2020, 10, 1)
    endDate = dt.datetime(2020, 11, 3)
    prices = blp.bdh(strSecurity=tickers, strFields=strFields, startdate=startDate, enddate=endDate)
    aa = prices['PX_LAST']
    print(prices)


    BDP

    blp = BLP()
    strFields2 = ["PX_LAST"]
    strFields = ["AMT_OUTSTANDING", "PX_LAST"]
    tickers = ["USF1067PAD80" + " CORP", "FR001400E7I7" + " CORP"]
    date = '20221123'
    test4 = blp.bdp(strSecurity=tickers, strFields=strFields, strOverrideField="AMOUNT_OUTSTANDING_AS_OF_DT",
                    strOverrideValue=date)
    blp.closeSession()

    
    blp = BLP()
    index = "SX5E Index" # ["SX5E Index", "CAC Index"] #
    strFields = "INDX_MWEIGHT_HIST"  # ["INDX_MWEIGHT_HIST" , "INDX_MWEIGHT_PX"]
    date = '20221123'
    bds_data = blp.bds(strSecurity=index, strFields=strFields, strOverrideField="END_DATE_OVERRIDE",
                    strOverrideValue=date)
    blp.closeSession()
    """
    # TEST SAVE
    # convert into dataframe
    # df = pd.DataFrame(data=bds_data, index = [1])
    # convert into excel
    # df.to_xlsx("output_bds.xlsx")
