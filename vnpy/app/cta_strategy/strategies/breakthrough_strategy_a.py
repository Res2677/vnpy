from datetime import time
import numpy as np
import pandas as pd
import peakutils
from transitions import Machine
from vnpy.app.cta_strategy import (
    CtaTemplate,
    TickData,
    TradeData,
    OrderData
)
from vnpy.trader.constant import (
    Status
)

TICK_COLUMNS = ['datetime', 'open', 'last', 'high', 'low', 'prev_close', 'volume', 'total_turnover',
                'a1', 'a2', 'a3', 'a4', 'a5', 'a1_v', 'a2_v', 'a3_v', 'a4_v', 'a5_v',
                'b1', 'b2', 'b3', 'b4', 'b5', 'b1_v', 'b2_v', 'b3_v', 'b4_v', 'b5_v',
                'current_volume', 'mean', 'TIME']


class BreakthroughStrategyA(CtaTemplate):
    """"""

    author = "EBrain"

    stop_ratio = 0.95
    open_price_times = 0.01
    one_point_one = 1.1
    min_window_size = 4
    tick_count_limit = 1 * 20
    sell_vol_limit = 100
    fixed_buy_size = 100

    _last_point_flag = 0
    _strategy_flag = ''
    _market_open_price = None
    _window_open_price = None
    _window_open_datetime = None
    _last_trough_price = None

    _last_peak_idx = None
    _last_trough_time = None
    _last_break_time = None
    _last_window_first_peak_price = None
    _buy_time = None
    _buy_price = 0
    _buy_volume = fixed_buy_size
    _sell_price = 0
    _stop_price = 0
    _df_1min_last_window_peak_before_flag = False

    _judged_tick_count = 0
    _break_hold = 0
    _last_b1_price = 0
    _last_b2_price = 0
    _last_volume = 0

    parameters = ["stop_ratio",
                  "open_price_times",
                  "min_window_size",
                  "tick_count_limit",
                  "sell_vol_limit",
                  "fixed_buy_size"]

    variables = ["_last_point_flag",
                 "_strategy_flag",
                 "_market_open_price",
                 "_window_open_price",
                 "_last_trough_price",
                 "_last_peak_idx",
                 "_buy_price",
                 "_buy_volume",
                 "_judged_tick_count",
                 "_break_hold",
                 "_last_b1_price",
                 "_last_b2_price"]

    STATES = ['start', 'peak', 'break', 'open', 'opening', 'watch', 'close', 'stop']
    TRANSITIONS = [
        {'trigger': 'new_peak', 'source': 'start', 'dest': 'peak', 'after': 'on_new_peak'},
        {'trigger': 'new_break', 'source': 'peak', 'dest': 'break', 'after': 'on_new_break'},
        {'trigger': 'new_open', 'source': 'break', 'dest': 'open', 'after': 'on_new_open'},
        {'trigger': 'new_watch', 'source': 'open', 'dest': 'watch', 'after': 'on_new_watch'},
        {'trigger': 'new_close', 'source': 'watch', 'dest': 'close', 'after': 'on_new_close'},
        {'trigger': 'new_stop', 'source': 'watch', 'dest': 'stop', 'after': 'on_new_stop'}
    ]

    def __init__(self,
                 cta_engine,
                 strategy_name,
                 vt_symbol,
                 setting):
        """"""
        super().__init__(cta_engine,
                         strategy_name,
                         vt_symbol,
                         setting)

        self._df_tick = pd.DataFrame(columns=TICK_COLUMNS)
        self._df_1min = pd.DataFrame(columns=TICK_COLUMNS)
        self._df_1min_first_peak = pd.DataFrame(columns=TICK_COLUMNS)
        self._df_1min_last_window_peak_before = pd.DataFrame(columns=TICK_COLUMNS)
        self._temp_pk = pd.DataFrame()
        self._temp_th = pd.DataFrame()

        self._order_state = 'order_start'
        self._tick_date = None

        self.machine = Machine(model=self,
                               states=self.STATES,
                               transitions=self.TRANSITIONS,
                               initial='start')

    def on_init(self):
        """
        Callback when strategy is inited.
        """
        self.write_log("strategy inited.")

    def on_start(self):
        """
        Callback when strategy is started.
        """
        self.write_log("strategy started.")

    def on_stop(self):
        """
        Callback when strategy is stopped.
        """
        self.write_log("strategy stopped.")

    def on_order(self, order: OrderData):
        """
        Callback of new order data update.
        """
        if self._order_state == 'order_watch':
            self.write_log('[x] order status:{} volume:{} traded:{}'
                           .format(order.status,
                                   order.volume,
                                   order.traded))
            if order.status == Status.ALLTRADED:
                self._order_state = 'order_open'
            elif order.status in set([Status.REJECTED, Status.CANCELLED]):
                self._order_state = 'order_end'
        return

    def on_trade(self, trade: TradeData):
        """
        Callback of new trade data update.
        """
        self.put_event()  # update strategy status to UI

    @staticmethod
    def con_p_t(indexes, index2, price_list):
        peak_df = pd.DataFrame({"peak": [1] * len(indexes)}, index=indexes)
        though_df = pd.DataFrame({"though": [1] * len(index2)}, index=index2)
        df = pd.concat([peak_df, though_df])
        df["st"] = df.index
        df = df.sort_values(by=["st"])
        df = df.set_index("st")
        df["last"] = price_list
        len_df = df.shape[0]
        drop_list = []
        for one in np.arange(len_df)[:-1]:
            if (df.iloc[one, 0] == 1) and (df.iloc[one + 1, 0] == 1):
                if df.iloc[one, 2] >= df.iloc[one + 1, 2]:
                    drop_list.append(df.iloc[one + 1:one + 2, :].index.values[0])
                else:
                    drop_list.append(df.iloc[one:one + 1, :].index.values[0])
            if (df.iloc[one, 1] == 1) and (df.iloc[one + 1, 1] == 1):
                if df.iloc[one, 2] <= df.iloc[one + 1, 2]:
                    drop_list.append(df.iloc[one + 1:one + 2, :].index.values[0])
                else:
                    drop_list.append(df.iloc[one:one + 1, :].index.values[0])
        df.drop(drop_list, inplace=True)
        new_indexes = df[df.peak == 1].index.tolist()
        new_indexes2 = df[df.though == 1].index.tolist()
        # print("new list is {} and {}".format(new_indexes,new_indexes2))
        return new_indexes, new_indexes2

    @staticmethod
    def find_peak_trough(df, quantile_percent=.25, bins=15):
        df1 = df.copy()
        df1.index.rename("id")
        df1 = df1.reset_index()
        cb = df1[["last"]].T.values[0]
        m = len(cb)
        if m > 30:
            m_dist = 4
        else:
            m_dist = m // bins
        k = np.mean(cb)
        new = np.array([one - k for one in cb])
        # indexes = peakutils.indexes(cb, thres=0.02/max(cb), min_dist=4)
        thres1 = np.percentile(new, quantile_percent)
        indexes = peakutils.indexes(new, thres=thres1, min_dist=m_dist)
        new2 = np.array([k - one for one in cb])
        thres2 = np.percentile(new2, quantile_percent)
        indexes2 = peakutils.indexes(new2, thres=thres2, min_dist=m_dist)
        tmp_index = np.sort(indexes.tolist() + indexes2.tolist())
        new_indexes, new_indexes2 = BreakthroughStrategyA.con_p_t(indexes=indexes, index2=indexes2, price_list=df1.loc[tmp_index, 'last'].tolist())
        #     df1[["last"]].plot()
        #     plt.plot(new_indexes,df1.loc[new_indexes,'last'].values.tolist(),'r+')
        #     plt.plot(new_indexes2,df1.loc[new_indexes2,'last'].values.tolist(),'k*')
        #     plt.show()
        if "index" in df1.columns.tolist():
            df1.drop(["index"], axis=1, inplace=True)
        return df1.loc[new_indexes, :], df1.loc[new_indexes2, :]

    @staticmethod
    def convergence_funcB(num, num1):
        flag = 1
        if len(num) < 2 or len(num1) < 2:
            flag = 0
        else:
            if num.iloc[0]['last'] < max(num[1:]['last']):
                flag = 0
        return flag

    @staticmethod
    def Tips1(df1, peak, through, col_list=['open', ['last', 'mean']]):
        flag = 1
        if len(peak) == 0 or len(through) == 0:
            flag = 0
        elif len(through) < 2 or len(peak) < 2 or peak.index.values[0] > through.index.values[0] or BreakthroughStrategyA.convergence_funcB(peak, through) == 0 or through["last"].values[0] <= 0.01 * (df1[col_list[0]].values[0] * 1.1):  # or judgeValues(df1,col_list[1])==0:
            flag = 0
        return flag

    @staticmethod
    def Tips2(df1, peak, through):
        flag = 1
        if len(peak) == 0 or len(through) == 0:
            flag = 0
        elif len(through) < 2 or len(peak) < 2 or peak.index.values[0] > through.index.values[0] or BreakthroughStrategyA.convergence_funcB(peak, through) == 0:  # or judgeValues(df1,col_list[1])==0:
            flag = 0
        return flag

    @staticmethod
    def Tips3(df1, peak, through, col_list=['open', ['last', 'mean'], ['last', 'current_volume', 'open']]):
        flag = 1
        if len(peak) == 0 or len(through) == 0:
            flag = 0
        elif len(peak) != 1 or len(through) != 1 or \
            peak.index.values[0] > through.index.values[0] or \
            (peak["last"].values[0] - through["last"].values[0]) <= 0.01 * (
                df1[col_list[0]].values[0] * 1.1):  # or judgeValues(df1,col_list[1])==0:
            flag = 0
        return flag

    @staticmethod
    def sell_first_seek(df_stored, df_tick, before=4):
        """Find first peaks and all peaks from the mix data.
        Args：
            df_stored: the pre-existing minute data of the current window. It contains the df_tick's minute.
            # df_minute：the DataFrame of one minute data, df_minute is not empty.
            df_tick: the DataFrame of one tick data.
            before：the minimum period for seeking the peak.
        Returns：
            temp_pk: All peaks' DataFrame.
            temp_th: All throughs' DataFrame.
            result_stop: the first peaks' DataFrame.
            last_point_flag: The flag of last point in the current window. -1 refers through, 0 refers nothing and 1 refers peak.
            strategy_flag: The flag of opening transaction, 'Tips1'/'Tips2'/'Tips3' refer three waves.
        """
        # The length of windows is short.
        if df_stored.shape[0] < before - 1:
            temp_pk = pd.DataFrame()
            temp_th = pd.DataFrame()
            result_stop = pd.DataFrame()
            last_point_flag = 0
            strategy_flag = ''
            return temp_pk, temp_th, result_stop, last_point_flag, strategy_flag
        df_m = df_stored.copy()
        # df_m = pd.concat([df_stored, df_m])
        # tick data's TIME %100 != 3
        if df_tick.TIME.values[0] == df_m.TIME.values[-1] + 3:
            df_m = pd.concat([df_m, df_tick.loc[:, ["TIME", "last", "open", "mean", "current_volume"]]])
            df_m = df_m.sort_values(by="TIME")
            df_m = df_m.reset_index().drop(["index"], axis=1)
        else:
            temp_pk = pd.DataFrame()
            temp_th = pd.DataFrame()
            result_stop = pd.DataFrame()
            last_point_flag = 0
            strategy_flag = ''
            return temp_pk, temp_th, result_stop, last_point_flag, strategy_flag

        last_point_flag = 0
        strategy_flag = ''
        result_stop = pd.DataFrame()
        temp_pk, temp_th = BreakthroughStrategyA.find_peak_trough(df_m)
        df_mm = df_m.iloc[:-1, :]
        if not temp_pk.empty:
            # temp_pk.rename(columns={"index": "INDEX"}, inplace=True)
            if df_m.TIME.values[-1] in temp_pk.TIME.tolist():
                last_point_flag = 1
        if not temp_th.empty:
            # temp_th.rename(columns={"index": "INDEX"}, inplace=True)
            if df_m.TIME.values[-1] in temp_th.TIME.tolist():
                last_point_flag = -1
                # first window
                #     if df_stored.shape[0] == before - 1 and df_stored.iloc[0]["TIME"] == 93100:
        if df_stored.iloc[0]["TIME"] == 93100:
            if not temp_pk.empty:
                index_max = temp_pk[["last"]].idxmax(axis=0, skipna=True).values[0]
                if temp_pk.loc[index_max]["last"] >= temp_pk.loc[index_max]["open"]:
                    result_stop = pd.DataFrame(temp_pk.loc[index_max]).T
                else:
                    result_stop = df_m.iloc[:1]
                    result_stop["last"] = [df_m.iloc[0]["open"]]
                    result_stop.TIME = [93000]
                    result_stop.index = [-1]  # insert into the first line
                #     print(f'{temp_pk},"$$$$$",{result_stop}')
            else:
                result_stop = df_m.iloc[:1]
                result_stop["last"] = [df_m.iloc[0]["open"]]
                result_stop.TIME = [93000]
                result_stop.index = [-1]
        else:
            if len(temp_pk) >= 2 and temp_pk.iloc[0]["last"] < temp_pk.iloc[-1]["last"]:
                # result_stop = pd.DataFrame(temp_pk.iloc[-1]).T
                result_stop = temp_pk.iloc[-1:]
            else:
                if temp_pk.empty:
                    result_stop = df_m.iloc[:1]
                    result_stop["last"] = [df_m.iloc[0]["open"]]
                    result_stop.TIME = [93000]
                    result_stop.index = [-1]
                else:
                    result_stop = temp_pk.iloc[:1]
        if "index" in result_stop.columns.tolist():
            result_stop.drop(["index"], axis=1, inplace=True)
        if "index" in temp_pk.columns.tolist():
            result_stop.drop(["index"], axis=1, inplace=True)
        if "index" in temp_th.columns.tolist():
            result_stop.drop(["index"], axis=1, inplace=True)

            # calculate strategy_flag
        if temp_pk.empty or temp_th.empty:
            pass
        else:
            if BreakthroughStrategyA.Tips1(df1=df_mm, peak=temp_pk, through=temp_th) == 1:
                strategy_flag = 'Tips1'
            elif BreakthroughStrategyA.Tips2(df1=df_mm, peak=temp_pk, through=temp_th) == 1:
                strategy_flag = 'Tips2'
            elif BreakthroughStrategyA.Tips3(df1=df_mm, peak=temp_pk, through=temp_th) == 1:
                strategy_flag = 'Tips3'
            else:
                strategy_flag = ''
        return temp_pk, temp_th, result_stop, last_point_flag, strategy_flag

    def reset(self):
        """Reset class to initial state
        Change a new window.
        """
        self._df_tick = pd.DataFrame(columns=TICK_COLUMNS)
        self._df_1min = pd.DataFrame(columns=TICK_COLUMNS)
        self._temp_pk = pd.DataFrame()
        self._temp_th = pd.DataFrame()
        self._market_open_price = None
        self._window_open_price = None
        self._window_open_datetime = None
        self._last_trough_price = None

        # self._df_1min_last_window_peak_before_flag = False
        # self._last_window_first_peak_price = None
        self._last_peak_idx = None
        self._last_trough_time = None
        self._last_break_time = None
        self._buy_time = None
        self._buy_price = 0
        self._sell_price = 0
        self._stop_price = 0
        self._last_b1_price = 0
        self._last_b2_price = 0

        self._judged_tick_count = 0
        self._break_hold = 0

        self._order_state = 'order_start'
        self.to_start()

    def reset_data_left(self):
        """
        Reset class to initial state
        leave peaks that found yet
        """
        # self._df_tick = pd.DataFrame(columns=df.columns)#
        # self._df_1min = pd.DataFrame(columns=df.columns)#
        self._temp_pk = pd.DataFrame()
        self._temp_th = pd.DataFrame()
        self._market_open_price = None
        self._window_open_price = None
        self._window_open_datetime = None
        self._last_trough_price = None

        # self._df_1min_last_window_peak_before_flag = False
        # self._last_window_first_peak_price = None
        self._last_peak_idx = None
        self._last_trough_time = None
        self._last_break_time = None
        self._buy_time = None
        self._buy_price = 0
        self._sell_price = 0
        self._stop_price = 0
        self._last_b1_price = 0
        self._last_b2_price = 0

        self._judged_tick_count = 0
        self._break_hold = 0

        self.to_start()

    def on_new_peak(self):
        # print('[x] find a new peak at idx:{} datetime_idx: {} price: {}'\
        #       .format(self._last_peak_idx,\
        #               self._df_1min.index[self._last_peak_idx],
        #               self._df_1min.iloc[self._last_peak_idx]['last']))
        # if self._last_trough_time:
        #     print('[x] last trough at idx:{} datetime_idx: {}'\
        #           .format(self._last_trough_time,\
        #                   self._df_1min[self._df_1min['TIME']==self.last_trough_time]))
        # else:
        #     print('[x] initial last trough at window open tick: {}, price:{}'\
        #           .format(self._window_open_datetime,
        #                   self._last_trough_price))
        # print("[x] current 1min price list: \n{}"\
        #       .format(self._df_1min['last']))
        # print("in this func")
        return

    def on_new_break(self):
        self.write_log('[x] find a new break at datetime_idx:{}'
                       .format(self._last_break_time))
        self.write_log("[x] current 1min price list: \n{}"
                       .format(self._df_1min['last']))
        return

    def on_new_open(self):
        self.write_log("[x] will open position after datetime_idx:{}"
                       .format(self._df_tick.index[-1]))
        return

    def on_new_watch(self):
        self.write_log('[x] opened a new position at datetime_idx:{}, price: {}, volume: {}'
                       .format(self._df_tick.index[-1],
                               self._buy_price,
                               self._buy_volume))
        self.write_log("[x] current tick price list after last 1min: \n{}"
                       .format(self._df_tick.iloc[self._df_tick.index >= self._df_1min.index[-1]]
                               [['last', 'a1', 'a2', 'a3', 'a4', 'a5',
                                 'b1', 'b2', 'b3', 'b4', 'b5']]))
        return

    def on_new_close(self):
        self.write_log('[x] close position at datetime_idx:{}, price: {}, volume: {}'
                       .format(self._df_tick.index[-1],
                               self._close_price,
                               self._buy_volume))
        self.write_log("[x] current tick price list after last 1min: \n{}"
                       .format(self._df_tick.iloc[self._df_tick.index >= self._df_1min.index[-1]]
                               [['last', 'a1', 'a2', 'a3', 'a4', 'a5',
                                 'b1', 'b2', 'b3', 'b4', 'b5']]))
        return

    def on_new_stop(self):
        self.write_log('[x] close position at datetime_idx:{}, price: {}, volume: {}'
                       .format(self._df_tick.index[-1],
                               self._stop_price,
                               self._buy_volume))
        self.write_log("[x] current tick price list after last 1min: \n{}"
                       .format(self._df_tick.iloc[self._df_tick.index >= self._df_1min.index[-1]]
                               [['last', 'a1', 'a2', 'a3', 'a4', 'a5',
                                 'b1', 'b2', 'b3', 'b4', 'b5']]))
        return

    @staticmethod
    def tickdata_to_series(tickdata: TickData):
        """
        translate TickData to Series format
        """
        tick_series = pd.Series()
        tick_series['datetime'] = tickdata.datetime
        tick_series['open'] = tickdata.open_price
        tick_series['last'] = tickdata.last_price
        tick_series['high'] = tickdata.high_price
        tick_series['low'] = tickdata.low_price
        tick_series['prev_close'] = tickdata.pre_close
        tick_series['volume'] = tickdata.volume
        tick_series['total_turnover'] = tickdata.open_interest
        tick_series['a1'] = tickdata.ask_price_1
        tick_series['a2'] = tickdata.ask_price_2
        tick_series['a3'] = tickdata.ask_price_3
        tick_series['a4'] = tickdata.ask_price_4
        tick_series['a5'] = tickdata.ask_price_5
        tick_series['a1_v'] = tickdata.ask_volume_1
        tick_series['a2_v'] = tickdata.ask_volume_2
        tick_series['a3_v'] = tickdata.ask_volume_3
        tick_series['a4_v'] = tickdata.ask_volume_4
        tick_series['a5_v'] = tickdata.ask_volume_5
        tick_series['b1'] = tickdata.bid_price_1
        tick_series['b2'] = tickdata.bid_price_2
        tick_series['b3'] = tickdata.bid_price_3
        tick_series['b4'] = tickdata.bid_price_4
        tick_series['b5'] = tickdata.bid_price_5
        tick_series['b1_v'] = tickdata.bid_volume_1
        tick_series['b2_v'] = tickdata.bid_volume_2
        tick_series['b3_v'] = tickdata.bid_volume_3
        tick_series['b4_v'] = tickdata.bid_volume_4
        tick_series['b5_v'] = tickdata.bid_volume_5
        return tick_series

    @staticmethod
    def is_trading_time(check):
        if (check >= time(9, 30) and check <= time(11, 30)
           or check >= time(13, 00) and check <= time(15, 00)):
            return True
        else:
            return False

    def on_tick(self, tickdata: TickData):
        """
        Callback of new tick data update
        """
        if not self.trading:
            return

        tick = self.tickdata_to_series(tickdata)

        if not self._tick_date:
            self._tick_date = tick['datetime'].date()

        if not self._tick_date == tick['datetime'].date():
            self.reset()
            self._tick_date = tick['datetime'].date()

        if not self.is_trading_time(tick['datetime'].time()):
            return

        # if time > 14:55, stop all open positions
        if tick['datetime'].time() > time(14, 55) and self.pos > 0:
            self._stop_volume = self.pos
            self._stop_price = tick['b1']
            self.to_stop()

        # calculate current volume
        if self._last_volume == 0:
            tick['current_volume'] = 0
        else:
            tick['current_volume'] = tick['volume'] - self._last_volume
            self._last_volume = tick['volume']
        self._last_volume = tick['volume']

        # calculate mean
        if tick['volume'] == 0:
            tick['mean'] = 0
        else:
            tick['mean'] = tick['total_turnover'] / tick['volume']

        idx_tick = tick['datetime']
        # initialize class parameters
        if not self._market_open_price:
            self._market_open_price = tick['open']
        if not self._last_trough_price:
            self._window_open_price = tick['last']
            self._window_open_datetime = tick['datetime']
            self._last_trough_price = self._window_open_price

        # generate resampled tick dataframe
        tick['TIME'] = 0
        self._df_tick.loc[idx_tick] = list(tick)
        df_tick_resample = self._df_tick.resample('3S').ffill()
        df_tick_resample["datetime"] = df_tick_resample.index.values
        df_tick_resample["TIME"] = ["".join(str(t).split(" ")[1].split(":")[i] for i in range(3)) for t in df_tick_resample.index]
        df_tick_resample["TIME"] = df_tick_resample["TIME"].astype(int)
        self._df_tick["TIME"] = ["".join(str(t).split(" ")[1].split(":")[i] for i in range(3)) for t in self._df_tick.index]
        self._df_tick["TIME"] = self._df_tick["TIME"].astype(int)
        self._df_1min["TIME"] = ["".join(str(t).split(" ")[1].split(":")[i] for i in range(3)) for t in self._df_1min.index]
        self._df_1min["TIME"] = self._df_1min["TIME"].astype(int)

        # generate minute dataframe based on resampled tick
        last_resample_time = df_tick_resample.index[-1]

        if last_resample_time.second == 0:
            self._df_1min.loc[last_resample_time] = list(tick)

        # entering Finite-State-Machine, 7 states total:
        # start: when a new PEAK matches the amplitute condition, trans to peak
        # peak: when tick after the PEAK matches C point condition, trans to break,
        #       if failed trans to start
        # break: if sell_volume_5 matches condition, trans to open
        # open: open position, trans to watch if success, trans to start if failed
        # watch: decide wether to close or to stop, trans to close or stop
        # close: close position, trans to start
        # stop: stop position, trans to start
        if self.state == 'start':
            # if not new_min:
            if last_resample_time.second % 100 != 3:
                return

            # wait for min_window_size, To be optimal
            if self._df_1min.shape[0] < self.min_window_size:
                return

            # find peak & trough in df_1min
            # lst_1min_price = list(self._df_1min['last'])
            # idx_peak, idx_trough = find_peak_trough(lst_1min_price)

            # print("666666666666666",self._df_1min,df_tick_resample.iloc[-1:])
            # raise Exception
            # min_last_window_peak_before = self._df_1min_last_window_peak_before
            # 用于判断是不是新窗口的信号 , 如果是新窗口 , 就把self._df_1min截取
            # print("flag is .....................\n",self._df_1min_last_window_peak_before_flag)
            if self._df_1min_last_window_peak_before_flag:  # need to add this
                self._df_1min = self._df_1min[self._df_1min['TIME'] >= self._df_1min_last_window_peak_before.iloc[-1]['TIME']]
            else:
                pass
            # self.write_log(f"窗口的起始时间点...................{self._df_1min.iloc[0]['TIME']}")
            last_first_peak = self._df_1min_first_peak
            self._temp_pk, self._temp_th, self._df_1min_first_peak, self._last_point_flag, self._strategy_flag = BreakthroughStrategyA.sell_first_seek(df_stored=self._df_1min, df_tick=df_tick_resample.iloc[-1:], before=self.min_window_size)
            # idx_peak = self._temp_pk.index.values
            # idx_trough = self._temp_th.index.values
            # print('this is result : \n',idx_peak,self._temp_pk, idx_trough,self._temp_th, self._df_1min_first_peak, self._last_point_flag, self._strategy_flag)

            # update last_trough, it's trough right after the newest peak
            # or just the newest trough

            if last_first_peak.empty:
                # self.reset_data_left()
                return
            elif (not self._df_1min_first_peak.empty) and (last_first_peak.iloc[-1]['last'] < self._df_1min_first_peak['last'].values[0]):
                # print("in this model..........................................................................")
                # print(last_first_peak['last'].values[0] ,self._df_1min_first_peak['last'].values[0])
                change_time = self._df_1min_first_peak['TIME'].values[0]
                self.write_log(f"window start at {self._df_1min.iloc[0]['TIME']}, changed time is :{change_time}")
                if change_time == 93000:
                    pass
                else:
                    for t in self._df_1min['TIME'][::-1]:
                        if t < change_time:
                            ctime = t
                            break
                    # print("ctime is ..................\n",ctime)
                    self._df_1min_last_window_peak_before = self._df_1min[self._df_1min['TIME'] == ctime]
                    self._df_1min_last_window_peak_before_flag = True
                    self.reset_data_left()
                    # self.reset()
                    # print(" the flag in this state ...............:\n",self._df_1min_last_window_peak_before_flag)
                    return
            else:
                self._df_1min_last_window_peak_before_flag = False
                # pass
            if self._strategy_flag == 'Tips1':
                if self._last_point_flag != 1:
                    self._last_window_first_peak_price = self._temp_pk.iloc[0]['last']
                    self._last_trough_time = self._df_1min.iloc[-1]['TIME']  # 指的是最后的min Time
                    self.new_peak()
                else:
                    return
            else:
                return
            # if self._strategy_flag == "Tips1":
            #     if self._last_point_flag == -1:
            #         self._last_window_first_peak_price = self._temp_pk.iloc[0]['last']
            #         self._last_trough_price = self._df_1min.iloc[-1]['last']
            #         self._last_trough_time = self._df_1min.iloc[-1]['TIME']
            #         self.new_break()
            #     else:
            #         # self.reset_data_left()
            #         return
            # else:
            #     return
                # self.reset_data_left()
            return
        elif self.state == 'peak':
            # 5）时间窗口内开仓时刻的价格 >= 同时时刻的均价；
            # 6）寻找A点：抓取全天所有交易时间tick数据逐笔成交明细，
            # 抓取最高价格(包含开盘价格)。如果后面有出现大于A的价格，
            # 则更新，A即之前当天逐笔成交出现过的最高价格。
            # 或者A价格附近正负1 % 内3秒成交出现大于等于10000手，
            # 则立马开仓，开仓委托价格为A * 0. * 5 %，平仓策略同下

            last_window_first_peak_price = self._last_window_first_peak_price
            if tick['last'] < tick['mean']:
                # self.reset()
                return
            if self._judged_tick_count > self.tick_count_limit:
                # self.reset()  # reset, transit back to state: start
                return
            self._judged_tick_count += 1
            if tick['last'] > last_window_first_peak_price:
                self._last_break_time = self._df_tick.iloc[-1]['TIME']
                self.new_break()
            else:
                self.reset_data_left()
                # self.reset()
            return
        elif self.state == 'break':
            last_window_first_peak_price = self._last_window_first_peak_price
            if tick['last'] > last_window_first_peak_price:
                sum_sell_vol = tick['a1_v'] + tick['a2_v'] + tick['a3_v'] + \
                    tick['a4_v'] + tick['a5_v']
                if sum_sell_vol > self.sell_vol_limit:
                    self.new_open()  # transit to state: open
            else:
                # return
                self.reset_data_left()
                # self.reset()
            return
        elif self.state == 'open':
            last_window_first_peak_price = self._last_window_first_peak_price
            if tick['last'] > last_window_first_peak_price:
                self._buy_price = tick['a3']
                self._last_b1_price = tick['b1']
                self._buy_time = tick['TIME']
                # self._last_b2_price = tick['b2']
                self.new_watch()
            else:
                self.to_break()
            return
        elif self.state == 'watch':
            buy_time = self._buy_time

            last_first_peak = self._df_1min_first_peak
            peakdf, _, self._df_1min_first_peak, peak_flag, _ = BreakthroughStrategyA.sell_first_seek(df_stored=self._df_1min, df_tick=df_tick_resample.iloc[-1:], before=self.min_window_size)
            # self.write_log(f"watching....................................................{peakdf}")

            if (not peakdf.empty) and (peakdf.iloc[-1]['TIME'] > buy_time):
                self._close_price = tick['last']
                self.new_close()
            elif tick['last'] < self.stop_ratio * self._market_open_price:
                self._stop_price = tick['b1']
                self.new_stop()
            else:
                pass

            if (not last_first_peak.empty) and (not self._df_1min_first_peak.empty) and (last_first_peak.iloc[0]["last"] < self._df_1min_first_peak.iloc[0]["last"]):
                change_time = self._df_1min_first_peak['TIME'].values[0]
                self.write_log(f"changed time is :{change_time}")
                for t in self._df_1min['TIME'][::-1]:
                    if t < change_time:
                        ctime = t
                        break
                # print("ctime is ..................\n",ctime)
                self._df_1min_last_window_peak_before = self._df_1min[self._df_1min['TIME'] == ctime]
                self._df_1min_last_window_peak_before_flag = True
                self.reset_data_left()
            return
        elif self.state == 'close':
            self.write_log('[x]entering close state')
            # TODO:
            # close position
            self.reset_data_left()
            return
        elif self.state == 'stop':
            self.write_log('[x]entering stop state')
            # TODO:
            # stop position
            self.reset_data_left()
            return
        else:
            return
