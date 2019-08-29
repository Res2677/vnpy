import multiprocessing
from time import sleep
from datetime import datetime,time
from logging import INFO

from vnpy.event import EventEngine
from vnpy.trader.setting import SETTINGS
from vnpy.trader.engine import MainEngine

from vnpy.gateway.feat import FeatGateway

feat_setting ={
    "proxy host":"",
    "proxy port":""
}


def run_child():
    event_engine = EventEngine()
    main_engine = MainEngine(event_engine)
    main_engine.add_gateway(FeatGateway)
    main_engine.write_log("main engine created")
    print("main engine created")

    main_engine.connect(feat_setting, "FEAT")
    main_engine.write_log("connect FEAT interface")

    print("connect FEAT interface")

    sleep(10)

    while True:
        sleep(1)

if __name__ == "__main__":
    run_child()
