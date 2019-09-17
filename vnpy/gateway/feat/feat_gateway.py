"""
Feat gateway for simulation
"""
import sys
import json
import time

from copy import copy
from datetime import datetime, timedelta
from threading import Lock
from vnpy.api.websocket import WebsocketClient
from vnpy.api.rest import Request, RestClient
from vnpy.trader.constant import (
    Direction,
    OrderType,
    Exchange,
    Product,
    Status
)
from vnpy.trader.gateway import BaseGateway

from vnpy.trader.object import (
    TickData,
    OrderData,
    TradeData,
    AccountData,
    ContractData,
    PositionData,
    BarData,
    OrderRequest,
    CancelRequest,
    SubscribeRequest,
    HistoryRequest
)

WEBSOCKET_HOST = "ws://127.0.0.1:9002"
REST_HOST = "http://192.168.91.130:8888"

ORDERTYPE_VT2FEAT = {
    OrderType.LIMIT: "LIMIT",
    OrderType.MARKET: "MARKET"
}
MARKETTYPE_FEAT2VT = {
    '深圳Ａ股': Exchange.SZSE,
    '上海Ａ股': Exchange.SSE
}
DIRECTION_VT2FEAT = {
    Direction.LONG: "BUY",
    Direction.SHORT: "SELL"
}


class FeatGateway(BaseGateway):
    """
    VN Trader gateway of Feature Corparation.
    """

    default_setting = {
        "proxy host": "",
        "proxy port": "",
        "session number": 3
    }

    exchanges = [Exchange.SSE,Exchange.SZSE]

    def __init__(self, event_engine):
        super().__init__(event_engine, "FEAT")

        self.order_count = 10000
        self.order_count_lock = Lock()
        self.orders = {}

        self.ws_api = FeatWebsocketApi(self)
        self.rest_api = FeatRestApi(self)
        self.ws_api.start()

    def connect(self, setting: dict):
        proxy_host = setting["proxy host"]
        proxy_port = setting["proxy port"]
        session_number = setting["session number"]

        if proxy_port.isdigit():
            proxy_port = int(proxy_port)
        else:
            proxy_port = 0

        self.rest_api.connect(session_number, proxy_host, proxy_port)
        self.ws_api.connect(proxy_host, proxy_port)

    def subscribe(self,req: SubscribeRequest):
        """Send contract event before subscribe, this contract info is used
           in cta_engine.send_order to validate the order by contract.
           In other gateway like okex or ctp, there is a seperate query_contract
           init step to do this, we implement this contract logic here for
           simplification.
        """
        contract = ContractData(
            symbol=req.symbol,
            exchange=Exchange.SSE,
            name=req.symbol,
            product=Product.EQUITY,
            size=1,
            pricetick=0.01,
            min_volume=1,
            history_data=True,
            gateway_name=self.gateway_name
        )
        self.on_contract(contract)
        self.ws_api.subscribe(req)

    def close(self):
        """"""
        self.rest_api.stop()
        self.ws_api.stop()

    def query_account(self):
        pass

    def query_position(self):
        pass

    def _new_order_id(self):
        with self.order_count_lock:
            self.order_count += 1
            return self.order_count

    def send_order(self, req: OrderRequest):
        """"""
        return self.rest_api.send_order(req)

    def cancel_order(self, req: CancelRequest):
        """"""
        self.rest_api.cancel_order(req)

    def on_order(self, order: OrderData):
        """"""
        self.orders[order.orderid] = order
        super().on_order(order)

    def get_order(self, orderid: str):
        """"""
        return self.orders.get(orderid, None)


class FeatRestApi(RestClient):
    """
    FEAT rest API
    """
    def __init__(self, gateway):
        super().__init__()

        self.gateway = gateway
        self.gateway_name = gateway.gateway_name

        self.order_count = 10000
        self.order_count_lock = Lock()

        self.connect_time = int(datetime.now().strftime("%y%m%d%H%M%S"))

    def connect(
        self,
        session_number: int,
        proxy_host: str,
        proxy_port: int
    ):
        """
        Initialize connection to REST server.
        """
        self.init(REST_HOST, proxy_host, proxy_port)
        self.start(session_number)
        self.gateway.write_log("REST API started.")
        # TODO: update time, contract, account, order once connected
        #self.query_time()
        #self.query_contract()
        self.query_account()
        #self.query_order()

    def _new_order_id(self):
        with self.order_count_lock:
            self.order_count += 1
            return self.order_count

    def query_account(self):
        """"""
        self.add_request(
            "GET",
            "/api/v1.0/positions",
            callback=self.on_query_account
        )

    def send_order(self, req: OrderRequest):
        """
        send order to exchange to trade
        """
        orderid = f"a{self.connect_time}{self._new_order_id()}"
        order = req.create_order_data(orderid, self.gateway_name)

        data = {
            "symbol": req.symbol,
            "type": ORDERTYPE_VT2FEAT[req.type],
            "action": DIRECTION_VT2FEAT[req.direction],
            "priceType": 0,
            "price": req.price,
            "amount": req.volume
        }

        if req.type == OrderType.MARKET:
            data["priceType"] = 4

        self.add_request(
            "POST",
            "/api/v1.0/orders",
            callback=self.on_send_order,
            data=data,
            extra=order,
            on_failed=self.on_send_order_failed,
            on_error=self.on_send_order_error,
            is_json=True
        )

        self.gateway.on_order(order)
        return order.vt_orderid

    def cancel_order(self, req: CancelRequest):
        """"""
        data = {
            "symbol": req.symbol,
            "client_iod": req.orderid
        }

        path = "/api/v1.0/orders/" + req.orderid
        self.add_request(
        "DELETE",
        path,
        callback=self.on_cancel_order,
        data=data,
        on_error=self.on_cancel_order_error,
        on_failed=self.on_cancel_order_failed,
        extra=req
        )

    def on_query_account(self, data, request):
        """"""
        account_data = data.get("subAccounts", None)
        if account_data:
            for k,v in account_data.items():
                account = AccountData(gateway_name=self.gateway_name,
                                      accountid=k,
                                      balance=v["可用金额"],
                                      frozen=v["冻结金额"])
                self.gateway.on_account(account)
            self.gateway.write_log("account query success")
        else:
            self.gateway.write_log("account query failed")

        pos_data = data.get("dataTable", None)
        if pos_data:
            pos_data = pos_data.get("rows", None)
            if pos_data:
                for pos in pos_data:
                    position = PositionData(gateway_name=self.gateway_name,
                                            symbol=pos[1],
                                            exchange=MARKETTYPE_FEAT2VT[pos[11]],
                                            direction=Direction.LONG,
                                            volume=pos[3],
                                            frozen=pos[5],
                                            price=pos[7],
                                            pnl=pos[6],
                                            yd_volume=pos[4]
                                            )
                    self.gateway.on_position(position)
                self.gateway.write_log("position query success")
            else:
                self.gateway.write_log("position query failed")
        else:
            self.gateway.write_log("position query failed")


    def on_send_order_failed(self, status_code: str, request: Request):
        """
        Callback when sending order failed on server.
        """
        order = request.extra
        order.status = Status.REJECTED
        self.gateway.on_order(order)
        msg = f"order failed, code: {status_code}, info:{request.response.text}"
        self.gateway.write_log(msg)

    def on_send_order_error(self, exception_type: type, exception_value: Exception,
        tb, request: Request):
        """
        Callback when sending order caused exception.
        """
        order = request.extra
        order.status = Status.REJECTED
        self.gateway.on_order(order)

        # Record exception if not ConnectionError
        if not issubclass(exception_type, ConnectionError):
            self.on_error(exception_type, exception_value, tb, request)

    def on_send_order(self, data, request):
        """
        update orderid to server orderid
        """
        order = request.extra
        order.orderid = data.get("id", None)
        self.gateway.on_order(order)

    def on_cancel_order_error(
    self, exception_type: type, exception_value: Exception, tb, request: Request
    ):
        """
        """
        if not issubclass(exception_type, ConnectionError):
            self.on_error(exception_type, exception_value, tb, request)

    def on_cancel_order(self, data, request):
        req = request.extra
        order = self.gateway.get_order(req.orderid)
        if order:
            order.status = Status.CANCELLED
            self.gateway.on_order(order)

    def on_cancel_order_failed(self, status_code: int, request: Request):
        req = request.extra
        order = self.gateway.get_order(req.orderid)
        if order:
            order.status = Status.REJECTED
            self.gateway.on_order(order)


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

        tick.name = d["name"]
        tick.last_price = float(d["last_price"])
        tick.open_price = float(d["open_price"])
        tick.high_price = float(d["high_price"])
        tick.low_price  = float(d["low_price"])
        tick.volume = float(d["volume"])
        tick.open_interest = float(d["open_interest"])
        tick.pre_close = float(d["pre_close"])
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
