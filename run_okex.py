from vnpy.event import EventEngine
from vnpy.trader.engine import MainEngine
from vnpy.trader.ui import MainWindow, create_qapp
#from vnpy.gateway.okex import OkexGateway
from vnpy.gateway.feat import FeatGateway
from vnpy.app.cta_strategy import CtaStrategyApp
from vnpy.app.cta_backtester import CtaBacktesterApp
#from vnpy.app.algo_trading import AlgoEngine
#from vnpy.app.script_trader import ScriptEngine
#from vnpy.app.risk_manager import RiskManagerEngine
#from vnpy.app.data_recorder import RecorderEngine
#from vnpy.app.csv_loader import CsvLoaderEngine
#from vnpy.app.rpc_service import RpcEngine

def main():
    """Start VN Trader"""
    qapp = create_qapp()

    event_engine = EventEngine()

    main_engine = MainEngine(event_engine)
    #main_engine.add_gateway(OkexGateway)
    main_engine.add_gateway(FeatGateway)
    main_engine.add_app(CtaStrategyApp)
    main_engine.add_app(CtaBacktesterApp)
    #main_engine.add_app(AlgoEngine)
    #main_engine.add_app(ScriptEngine)
    #main_engine.add_app(RiskManagerEngine)
    #main_engine.add_app(RecorderEngine)
    #main_engine.add_app(CsvLoaderEngine)
    #main_engine.add_app(RpcEngine)

    main_window = MainWindow(main_engine, event_engine)
    main_window.showMaximized()

    qapp.exec()

if __name__ == "__main__":
    main()
