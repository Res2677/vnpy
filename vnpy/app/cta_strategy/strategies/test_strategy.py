from datetime import datetime, time
import numpy as np
import pandas as pd
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

class TestStrategy(CtaTemplate):
    """This is a test strategy"""

    author = "EBrain"

    test_parameter = 0.9
    test_variable = 0

    parameters = ["test_parameter"]
    variables = ["test_variable"]

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
        self.write_log("order received.")

    def on_trade(self, trade: TradeData):
        """
        Callback of new trade data update
        """
        self.put_event()

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

    def on_tick(self, tickdata: TickData):
        """
        Callback of new tick data update
        """
        if not self.trading:
            return

        tick = self.tickdata_to_series(tickdata)
        self.tick = tick

        print(f"{tick['datetime'].time()}")
