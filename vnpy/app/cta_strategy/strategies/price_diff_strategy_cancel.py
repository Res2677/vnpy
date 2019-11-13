from datetime import datetime, time
#import pandas as pd
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
                ]


class PriceDiffStrategy_1(CtaTemplate):
    """"""

    author = "EBrain"

    stop_ratio = 0.95
    h_a1_b1 = 0.01
    h_a1_a2 = 0.01
    h1_ratio = 0.1
    h2_ratio = 0.1
    tick_count_limit = 1 * 20
    sell_vol_limit = 100
    fixed_buy_size = 100

    _buy_time = None
    _buy_price = 0
    _buy_volume = fixed_buy_size
    _close_price = 0
    _close_volume = 0
    _stop_price = 0
    _stop_volume = 0

    _judged_tick_count = 0
    _last_b1_price = 0
    _last_b2_price = 0
    _last_volume = 0

    parameters = ["stop_ratio",
                  "h_a1_b1",
                  "h_a1_a2",
                  "h1_ratio",
                  "h2_ratio",
                  "tick_count_limit",
                  "sell_vol_limit",
                  "fixed_buy_size"]

    variables = ["_buy_price",
                 "_buy_volume",
                 "_judged_tick_count",
                 "_last_b1_price",
                 "_last_b2_price"]

    STATES = ['start', 'open', 'opening', 'watch', 'close', 'stop']
    TRANSITIONS = [
        # {'trigger': 'new_peak', 'source': 'start', 'dest': 'peak', 'after': 'on_new_peak'},
        #{'trigger': 'new_break', 'source': 'peak', 'dest': 'break', 'after': 'on_new_break'},
        #{'trigger': 'new_open', 'source': 'break', 'dest': 'open', 'after': 'on_new_open'},
        {'trigger': 'new_open', 'source':'start', 'dest':'open', 'after':'on_new_open'},
        {'trigger': 'new_watch', 'source': 'open', 'dest': 'watch', 'after': 'on_new_watch'},
        {'trigger': 'new_close', 'source': 'watch', 'dest': 'close', 'after': 'on_new_close'},
        {'trigger': 'new_stop', 'source': 'watch', 'dest': 'stop', 'after': 'on_new_stop'},
        {'trigger': 'new_start', 'source': 'close', 'dest': 'start', 'after': 'on_new_start'},
        {'trigger': 'new_start', 'source': 'stop', 'dest': 'start', 'after': 'on_new_start'},
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
            if self._judged_tick_count > 20:
                self._order_state = 'order_end'
                self._judged_tick_count = 0
            if order.status == Status.SUBMITTING:
                self._judged_tick_count += 1
                return
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
        """Reset class to initial state
        Change a new window.
        """
        self._buy_time = None
        self._buy_price = 0
        self._close_price = 0
        self._close_volume = 0
        self._stop_price = 0
        self._stop_volume = 0
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
        self._buy_time = None
        self._buy_price = 0
        self._close_price = 0
        self._close_volume = 0
        self._stop_price = 0
        self._stop_volume = 0
        self._last_b1_price = 0
        self._last_b2_price = 0

        self._judged_tick_count = 0
        self._break_hold = 0

        self._order_state = 'order_start'
        self.to_start()


    def on_new_open(self):
        self.write_log("[x] will open position after datetime_idx:{}"
                       .format(self.tick.datetime.time()))
        print("[x] will open position after datetime_idx:{}"
         .format(self.tick.datetime.time()))
        return

    def on_new_watch(self):
        self.write_log('[x] opened a new position at datetime_idx:{}, price: {}, volume: {}'
                       .format(self.tick.datetime.time(),
                               self._buy_price,
                               self._buy_volume))
        print('[x] opened a new position at datetime_idx:{}, price: {}, volume: {}'
                       .format(self.tick.datetime.time(),
                               self._buy_price,
                               self._buy_volume))
        return

    def on_new_close(self):
        self.write_log('[x] close position at datetime_idx:{}, price: {}, volume: {}'
                       .format(self.tick.datetime.time(),
                               self._close_price,
                               self._buy_volume))
        print('[x] close position at datetime_idx:{}, price: {}, volume: {}'
                       .format(self.tick.datetime.time(),
                               self._close_price,
                               self._buy_volume))
        return

    def on_new_stop(self):
        self.write_log('[x] close position at datetime_idx:{}, price: {}, volume: {}'
                       .format(self.tick.datetime.time(),
                               self._stop_price,
                               self._buy_volume))
        return

    @staticmethod
    def is_trading_time(check):
        if (check >= time(9, 30) and check <= time(11, 30)
           or check >= time(13, 00) and check <= time(15, 00)):
            return True
        else:
            return False

    @staticmethod
    def str_to_datetime(raw_time):
        if len(raw_time) != 23:
            return None
        return datetime(int(raw_time[0:4]),
                        int(raw_time[5:7]),
                        int(raw_time[8:10]),
                        int(raw_time[11:13]),
                        int(raw_time[14:16]),
                        int(raw_time[17:19]))

    def on_tick(self, tickdata: TickData):
        """
        Callback of new tick data update
        """
        if not self.trading:
            return

        tick = tickdata
        self.tick = tickdata

        if not self._tick_date:
            self._tick_date = tick.datetime.date()

        if not self._tick_date == tick.datetime.date():
            self.reset()
            self._tick_date = tick.datetime.date()

        if not self.is_trading_time(tick.datetime.time()):
            return

        # if time > 14:55, stop all open positions
        if tick.datetime.time() > time(14, 55) and self.pos > 0:
            self._stop_volume = self.pos
            self._stop_price = tick.bid_price_1
            self.to_stop()

        # calculate current volume
        if self._last_volume == 0:
            tick.current_volume = 0
        else:
            tick.current_volume = tick.volume - self._last_volume
            self._last_volume = tick.volume
        self._last_volume = tick.volume

        # The following part is price-difference strategy.
        self._last_a1_price = tick.ask_price_1
        self._last_a2_price = tick.ask_price_2
        self._last_b1_price = tick.bid_price_1

        if self.state == 'start':
            print(f"{tick.datetime.time()} in start.......................")
            if self._last_a1_price:
                h1 = (self._last_a1_price - self._last_b1_price) / self._last_a1_price
                if h1 >= self.h_a1_b1:
                    self._buy_price = self._last_b1_price + (self._last_a1_price - self._last_b1_price) * self.h1_ratio
                    self._sell_price = self._last_a1_price - (self._last_a1_price - self._last_b1_price) * self.h1_ratio
                    self.new_open()
            if self._last_a2_price:
                h2 = (self._last_a2_price - self._last_a1_price) / self._last_a2_price
                if h2 >= self.h_a1_a2:
                    self._buy_price = self._last_a1_price
                    self._sell_price = self._last_a2_price - self.h2_ratio*(self._last_a2_price - self._last_a1_price)
                    self.new_open()
                else:
                    return
            else:
                return

        elif self.state == 'open':
            print(f"{tick.datetime.time()} in open.......................",self._order_state,"\n")
            # 若实时价格低于挂单的买价则可以买入，将数据添加至待卖出的sell_set中，否则继续挂单等待。
            # 若等待超过2个tick，则撤销挂单。

            if self._order_state == 'order_start':
                if self._buy_price >= self.tick.bid_price_1:
                    self._order_state = 'order_watch'
                    self.buy(self._buy_price, self._buy_volume)
                else:
                    self.to_open()
            elif self._order_state == 'order_open':
                # open success
                self._order_state = 'order_start'
                self.new_watch()
            elif self._order_state == 'order_end':
                # open failed
                self._order_state = 'order_start' #TODO Partly trade
                # self.to_peak()  # transit back to peak
                self.cancel_all()
                self.reset()
                self.to_start()
            else: #order_watch
                # if order is not fullfilled in tick_count_limit, cancel all order
                # print(f"self._judged_tick_count is {self._judged_tick_count}")
                pass

        elif self.state == 'watch':
            print(f"{tick.datetime.time()} in watch.......................", self._order_state, "\n")
            # 若实时价格可以卖出位于待卖出sell_set中的股票（价格及数量满足），则挂单卖出。否则继续等待直至符合条件或平仓。
            if self._sell_price <= self.tick.ask_price_1:
                self._close_price = self._sell_price
                self._close_volume = self._buy_volume
                #print(f"self.tick is {self.tick}")
                self.new_close()
            else:
                 return
        elif self.state == 'close':
            print(f"{tick.datetime.time()} in close.......................", self._order_state, "\n")

            if self._order_state == 'order_start':
                self._order_state = 'order_watch'
                self.sell(self._close_price, self._close_volume)
            elif self._order_state == 'order_open':
                self.write_log('[x] position close success')
                self.reset()
                # self.reset_data_left()#1015
                self.to_start()
                self._order_state = 'order_start'
            elif self._order_state == 'order_end':
                self.write_log('[x] position close failed, resell at b1 price: {}'
                               .format(self.sell_price))
                self._close_price = tick.bid_price_1
                # self._close_price = self.sell_price
                self._order_state = 'order_start'
            else:
                print("close state",self._order_state)
                # self._order_state = 'order_open'
                pass
            # return
        elif self.state == 'stop':
            if self._order_state == 'order_start':
                self.sell(self._stop_price, self._stop_volume)
                self._order_state = 'order_watch'
            elif self._order_state == 'order_open':
                self.write_log('[x] position stop success')
                self.reset()
                # self.reset_data_left()  # 1015
                self.to_start()
                self._order_state = 'order_start'
            elif self._order_state == 'order_end':
                self.write_log('[x] position stop failed, resell at b1 price: {}'
                               .format(tick.bid_price_1))
                self._stop_price = tick.bid_price_1
                self._order_state = 'order_start'
            else:
                pass
            # return
        else:
            return
