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
                'current_volume']


class BreakthroughStrategyC(CtaTemplate):
    """"""

    author = "MigToLoveYou"

    min_window_size = 4
    h_amp_ratio = 0.02
    y_amp_ratio = 0.005
    vol_ratio = 2
    tick_count_limit = 2 * 20
    break_hold_limit = 2
    sell_vol_limit = 100
    fixed_buy_size = 100

    _market_open_price = None
    _window_open_price = None
    _window_open_datetime = None
    _last_trough_price = None

    _last_peak_idx = None
    _last_trough_idx = None
    _last_break_idx = None
    _buy_price = 0
    _buy_volume = fixed_buy_size
    _close_price = 0
    _close_volume = 0
    _stop_price = 0
    _stop_volume = 0

    _judged_tick_count = 0
    _break_hold = 0
    _last_b1_price = 0
    _last_b2_price = 0
    _last_volume = 0

    parameters = ["min_window_size",
                  "h_amp_ratio",
                  "y_amp_ratio",
                  "vol_ratio",
                  "tick_count_limit",
                  "break_hold_limit",
                  "sell_vol_limit",
                  "fixed_buy_size"]

    variables = ["_market_open_price",
                 "_window_open_price",
                 "_last_trough_price",
                 "_last_peak_idx",
                 "_last_trough_idx",
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

    def reset(self):
        """"""
        self._df_tick = self._df_tick.iloc[0:0]
        self._df_1min = self._df_1min.iloc[0:0]
        self._market_open_price = None
        self._window_open_price = None
        self._window_open_datetime = None
        self._last_trough_price = None

        self._last_peak_idx = None    # 1min
        self._last_trough_idx = None  # 1min
        self._last_break_idx = None   # tick
        self._buy_price = 0
        self._buy_volume = self.fixed_buy_size
        self._stop_price = 0
        self._stop_volume = 0
        self._close_price = 0
        self._close_volume = 0

        self._judged_tick_count = 0
        self._break_hold = 0
        self._last_b1_price = 0
        self._last_b2_price = 0

        self._order_state = 'order_start'
        self.to_start()

    @staticmethod
    def is_trading_time(check):
        if (check >= time(9, 30) and check <= time(11, 30)
           or check >= time(13, 00) and check <= time(15, 00)):
            return True
        else:
            return False

    @staticmethod
    def find_peak_trough(lst_price, quantile_percent=.25, bins=15):
        len_price = len(lst_price)
        if len_price > 30:
            min_span = 4
        else:
            min_span = len_price // bins

        mean_price = np.mean(lst_price)
        norm_lst_price = np.array([price - mean_price for price in lst_price])
        thres_peak = np.percentile(norm_lst_price,
                                   quantile_percent)
        idx_peak = peakutils.indexes(norm_lst_price,
                                     thres=thres_peak,
                                     min_dist=min_span)

        norm_lst_price_reverse = np.array([mean_price - price for price in lst_price])
        thres_trough = np.percentile(norm_lst_price_reverse,
                                     quantile_percent)
        idx_trough = peakutils.indexes(norm_lst_price_reverse,
                                       thres=thres_trough,
                                       min_dist=min_span)

        return idx_peak, idx_trough

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

    def on_new_peak(self):
        self.write_log('[x] find a new peak at idx:{} datetime_idx: {} price: {}'
                       .format(self._last_peak_idx,
                               self._df_1min.index[self._last_peak_idx],
                               self._df_1min.iloc[self._last_peak_idx]['last']))
        if self._last_trough_idx:
            self.write_log('[x] last trough at idx:{} datetime_idx: {}'
                           .format(self._last_trough_idx,
                                   self._df_1min.index[self.last_trough_idx]))
        else:
            self.write_log('[x] initial last trough at window open tick: {}, price:{}'
                           .format(self._window_open_datetime,
                                   self._last_trough_price))
        self.write_log("[x] current 1min price list: \n{}"
                       .format(self._df_1min['last']))
        self.put_event()
        return

    def on_new_break(self):
        self.write_log('[x] find a new break at datetime_idx:{}'
                       .format(self._last_break_idx))
        self.write_log("[x] current 1min price list: \n{}"
                       .format(self._df_1min['last']))
        self.put_event()
        return

    def on_new_open(self):
        self.write_log("[x] will open position after datetime_idx: {}"
                       .format(self._df_tick.index[-1]))
        self.put_event()
        return

    def on_new_watch(self):
        self.write_log('[x] opened a new position at datetime_idx: {}, price: {}, volume: {}'
                       .format(self._df_tick.index[-1],
                               self._buy_price,
                               self._buy_volume))
        self.write_log("[x] current tick price list after last 1min: \n{}"
                       .format(self._df_tick.iloc[self._df_tick.index >= self._df_1min.index[-1]]
                               [['last', 'a1', 'a2', 'a3', 'a4', 'a5',
                                 'b1', 'b2', 'b3', 'b4', 'b5']]))
        self.put_event()
        return

    def on_new_close(self):
        self.write_log('[x] close position at datetime_idx: {}, price: {}, volume: {}'
                       .format(self._df_tick.index[-1],
                               self._close_price,
                               self._buy_volume))
        self.write_log("[x] current tick price list after last 1min: \n{}"
                       .format(self._df_tick.iloc[self._df_tick.index >= self._df_1min.index[-1]]
                               [['last', 'a1', 'a2', 'a3', 'a4', 'a5',
                                 'b1', 'b2', 'b3', 'b4', 'b5']]))
        self.put_event()
        return

    def on_new_stop(self):
        self.write_log('[x] close position at datetime_idx: {}, price: {}, volume: {}'
                       .format(self._df_tick.index[-1],
                               self._stop_price,
                               self._buy_volume))
        self.write_log("[x] current tick price list after last 1min: \n{}"
                       .format(self._df_tick.iloc[self._df_tick.index >= self._df_1min.index[-1]]
                               [['last', 'a1', 'a2', 'a3', 'a4', 'a5',
                                 'b1', 'b2', 'b3', 'b4', 'b5']]))
        self.put_event()
        return

    def on_tick(self, tickdata: TickData):
        """
        Callback of new tick data update.
        """
        if not self.trading:
            return

        tick = BreakthroughStrategyC.tickdata_to_series(tickdata)

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

        idx_tick = tick['datetime']
        # initialize class parameters
        if not self._market_open_price:
            self._market_open_price = tick['open']
        if not self._last_trough_price:
            self._window_open_price = tick['last']
            self._window_open_datetime = tick['datetime']
            self._last_trough_price = self._window_open_price

        # generate resampled tick dataframe
        self._df_tick.loc[idx_tick] = list(tick)
        df_tick_resample = self._df_tick.resample('3S').ffill()

        # generate minute dataframe based on resampled tick
        last_resample_time = df_tick_resample.index[-1]
        new_min = False
        if last_resample_time.second == 0:
            new_min = True
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
            if not new_min:
                return

            # wait for min_window_size
            if self._df_1min.shape[0] < self.min_window_size:
                return

            # find peak & trough in df_1min
            lst_1min_price = list(self._df_1min['last'])
            idx_peak, idx_trough = self.find_peak_trough(lst_1min_price)

            # update last_trough, it's trough right after the newest peak
            # or just the newest trough
            if idx_trough.size > 0:
                if idx_peak.size > 0:
                    for idx in idx_trough[::-1]:
                        if idx < idx_peak[-1]:
                            self._last_trough_price = self._df_1min.iloc[idx]['last']
                            self.last_trough_idx = idx
                            break
                else:
                    self._last_trough_price = self._df_1min.iloc[idx_trough[-1]]['last']
                    self.last_trough_idx = idx_trough[-1]

            # print(self.df_1min.iloc[idx_trough, :][['datetime', 'last', 'volume', 'mean']])

            if idx_peak.size > 0:
                peak_price = self._df_1min.iloc[idx_peak[-1]]['last']
                next_trough_price = peak_price
                for price in self._df_1min.iloc[idx_peak[-1]:]['last']:
                    if price < peak_price:
                        next_trough_price = price
                        break

                h_amplitude = peak_price - self._last_trough_price
                y_amplitute = peak_price - next_trough_price
                if h_amplitude > self._market_open_price * self.h_amp_ratio and\
                   y_amplitute < peak_price * self.y_amp_ratio:
                    self._last_peak_idx = idx_peak[-1]
                    self.new_peak()  # transit to state: peak
                else:
                    self.reset()  # reset, start from scratch
            return
        elif self.state == 'peak':
            peak_price = self._df_1min.iloc[self._last_peak_idx]['last']
            peak_volume = self._df_1min.iloc[self._last_peak_idx]['current_volume']
            # only search for a limit period of time
            if self._judged_tick_count > self.tick_count_limit:
                self.reset()  # reset, transit back to state: start
                return

            self._judged_tick_count += 1
            if tick['last'] >= peak_price:  # break should last for a few ticks
                self._break_hold += 1
                if self._break_hold > self.break_hold_limit:
                    if tick['current_volume'] > peak_volume:
                        self._last_break_idx = self._df_tick.index[-1]
                        self.new_break()  # transit to state: break
            else:
                # self.reset()
                self._break_hold = 0
            return
        elif self.state == 'break':
            peak_price = self._df_1min.iloc[self._last_peak_idx]['last']
            if tick['last'] >= peak_price:  # break must hold before open position
                sum_sell_vol = tick['a1_v'] + tick['a2_v'] + tick['a3_v'] +\
                    tick['a4_v'] + tick['a5_v']
                if sum_sell_vol > self.sell_vol_limit:
                    self.new_open()  # transit to state: open
            else:
                self.reset()  # reset, transit back to state: start
            return
        elif self.state == 'open':
            if self._order_state == 'order_start':
                peak_price = self._df_1min.iloc[self._last_peak_idx]['last']
                if tick['last'] >= peak_price:
                    self._buy_price = tick['a3']
                    self._last_b1_price = tick['b1']
                    self._last_b2_price = tick['b2']

                    self.buy(self._buy_price, self._buy_volume)
                    self._order_state = 'order_watch'
                else:
                    self.to_break()  # transit back to break
            elif self._order_state == 'order_open':
                # open success
                self._order_state = 'order_start'
                self.new_watch()
            elif self._order_state == 'order_end':
                # open failed
                self._order_state = 'order_start'
                self.to_peak()  # transit back to peak
            else:
                # state:order_watch is handled in func:on_order
                pass
            return
        elif self.state == 'watch':
            peak_price = self._df_1min.iloc[self._last_peak_idx]['last']
            if tick['b1'] >= self._last_b1_price:
                self._last_b1_price = tick['b1']
                self._last_b2_price = tick['b2']
            elif tick['b1'] <= self._last_b2_price:
                self._close_price = tick['b1']
                self._close_volume = self._buy_volume
                self.new_close()
            elif tick['last'] < peak_price:
                self._stop_price = tick['b1']
                self._stop_volume = self._buy_volume
                self.new_stop()
            return
        elif self.state == 'close':
            if self._order_state == 'order_start':
                self.sell(self._close_price, self._close_volume)
                self._order_state = 'order_watch'
            elif self._order_state == 'order_open':
                self.write_log('[x] position close success')
                self.reset()
            elif self._order_state == 'order_end':
                self.write_log('[x] position close failed, resell at b2 price: {}'
                               .format(tick['b2']))
                self._close_price = tick['b2']
                self._order_state = 'order_start'
            else:
                pass
            return
        elif self.state == 'stop':
            if self._order_state == 'order_start':
                self.sell(self._stop_price, self._stop_volume)
                self._order_state = 'order_watch'
            elif self._order_state == 'order_open':
                self.write_log('[x] position stop success')
                self.reset()
            elif self._order_state == 'order_end':
                self.write_log('[x] position stop failed, resell at b2 price: {}'
                               .format(tick['b2']))
                self._stop_price = tick['b2']
                self._order_state = 'order_start'
            return
        else:
            return
