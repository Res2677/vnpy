"""
Feat gateway for simulation
"""
import sys
import json
import time

from copy import copy
from datetime import datetime, timedelta
from vnpy.api.websocket import WebsocketClient
from vnpy.trader.constant import Exchange
from vnpy.trader.gateway import BaseGateway
from vnpy.trader.object import (
    TickData,
    OrderData,
    TradeData,
    AccountData,
    ContractData,
    BarData,
    OrderRequest,
    CancelRequest,
    SubscribeRequest,
    HistoryRequest
)

WEBSOCKET_HOST = "ws://127.0.0.1:9002"


class FeatGateway(BaseGateway):
    """
    VN Trader gateway of Feature Corparation.
    """

    default_setting = {
        "proxy host": "",
        "proxy port": ""
    }

    exchanges = [Exchange.FEAT]

    def __init__(self, event_engine):
        super().__init__(event_engine, "FEAT")

        self.ws_api = FeatWebsocketApi(self)
        self.ws_api.start()

    def connect(self, setting: dict):
        proxy_host = setting["proxy host"]
        proxy_port = setting["proxy port"]

        if proxy_port.isdigit():
            proxy_port = int(proxy_port)
        else:
            proxy_port = 0

        self.ws_api.connect(proxy_host, proxy_port)

    def subscribe(self,req: SubscribeRequest):
        self.ws_api.subscribe(req)

    def close(self):
        self.ws_api.stop()

    def query_account(self):
        pass

    def query_position(self):
        pass

    def send_order(self):
        pass

    def cancel_order(self):
        pass




class FeatWebsocketApi(WebsocketClient):
    """
    FEAT web socket API
    """
    def __init__(self, gateway):
        super().__init__()

        self.gateway = gateway
        self.gateway_name = gateway.gateway_name
        self.host = WEBSOCKET_HOST

        self.connect_time = 0

        self.callbacks = {}
        self.ticks = {}

    def connect(
        self,
        proxy_host,
        proxy_port
    ):
        self.connect_time = int(datetime.now().strftime("%y%m%d%H%M%S"))
        self.init(WEBSOCKET_HOST)

    def unpack_data(self, data):
        return json.loads(data)

    def on_connected(self):
        self.gateway.write_log("Websocket API connected")

    def on_disconnected(self):
        self.gateway.write_log("Websocket API disconnected")

    def subscribe(self, req: SubscribeRequest):
        tick = TickData(
            symbol=req.symbol,
            exchange=req.exchange,
            name=req.symbol,
            datetime=datetime.now(),
            gateway_name=self.gateway_name,
        )
        self.ticks[req.symbol] = tick

        channel_ticker = f"ticker-{req.symbol}"
        self.callbacks[channel_ticker] = self.on_ticker

        req = {
            "op": "subscribe",
            "args": [channel_ticker]
        }
        self.send_packet(req)

    def on_packet(self, packet: dict):
        if "event" in packet:
            event = packet["event"]
            if event == "subscribe":
                return
            elif event == "error":
                msg = packet["message"]
                self.gateway.write_log("Websocket API request error: {msg}")
            elif event == "ready":
                self.gateway.write_log("Websocket API ready for transfer")
        else:
            channel = packet["table"]
            data = packet["data"]
            callback = self.callbacks.get(channel, None)

            if callback:
                callback(data)

    def on_ticker(self, d):
        symbol = d["symbol"]
        tick = self.ticks.get(symbol, None)
        if not tick:
            return

        tick.last_price = float(d["current"])
        tick.open_price = float(d["current"])
        tick.high_price = float(d["high"])
        tick.low_price  = float(d["low"])
        tick.volume = float(d["volume"])
        tick.datetime = datetime.strptime(
            d["time"], "%Y-%m-%d %H:%M:%S")

        tick.bid_price_1 = float(d["b1_p"])
        tick.ask_price_1 = float(d["a1_p"])
        tick.bid_volume_1 = float(d["b1_v"])
        tick.ask_volume_1 = float(d["a1_v"])
        tick.bid_price_2 = float(d["b2_p"])
        tick.ask_price_2 = float(d["a2_p"])
        tick.bid_volume_2 = float(d["b2_v"])
        tick.ask_volume_2 = float(d["a2_v"])
        tick.bid_price_3 = float(d["b3_p"])
        tick.ask_price_3 = float(d["a3_p"])
        tick.bid_volume_3 = float(d["b3_v"])
        tick.ask_volume_3 = float(d["a3_v"])
        tick.bid_price_4 = float(d["b4_p"])
        tick.ask_price_4 = float(d["a4_p"])
        tick.bid_volume_4 = float(d["b4_v"])
        tick.ask_volume_4 = float(d["a4_v"])
        tick.bid_price_5 = float(d["b5_p"])
        tick.ask_price_5 = float(d["a5_p"])
        tick.bid_volume_5 = float(d["b5_v"])
        tick.ask_volume_5 = float(d["a5_v"])

        #send tick to gateway, gateway push it to event_engine
        self.gateway.on_tick(copy(tick))
