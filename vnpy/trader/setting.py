"""
Global setting of VN Trader.
"""

from logging import CRITICAL

from .utility import load_json

SETTINGS = {
    "font.family": "Arial",
    "font.size": 12,

    "log.active": True,
    "log.level": CRITICAL,
    "log.console": True,
    "log.file": True,

    "email.server": "smtp.qq.com",
    "email.port": 465,
    "email.username": "",
    "email.password": "",
    "email.sender": "",
    "email.receiver": "",

    "rqdata.username": "license",
    "rqdata.password": "TItprlYfcJQFiMbMUEq_Voo4_ngiNQr7cV6tobnZinw6bX733Oyfvp7v_jO2V0mmcfBGWQTelO7CTZu7YrYYl4Z3IZhXtglPrML-XMLxiFlcI3JwMgO_Gl7sdQA4gSOxtuTabgXm2dLccsYr-aPqxsneSYh2n0A-7ZIukKnWOF8=GVRwhrddx1qYsrgp57bXdcZvkik_th8S_fp4BARjpKd_MNs4CkuEs1DMUL8mbO83a2MbH-3aD4kxTX6JllfkPQz5Pvn8Yc4p3uYmPOG_5f-T-Iq17jrIbkhSAnAEtDYwCR8B-nKrHtbffDc4AMUIVdzj_NEqH9BXQWfpP_SFHCc=",

    "database.driver": "sqlite",  # see database.Driver
    "database.database": "database.db",  # for sqlite, use this as filepath
    "database.host": "localhost",
    "database.port": 3306,
    "database.user": "root",
    "database.password": "",
    "database.authentication_source": "admin",  # for mongodb
}

# Load global setting from json file.
SETTING_FILENAME = "vt_setting.json"
SETTINGS.update(load_json(SETTING_FILENAME))


def get_settings(prefix: str = ""):
    prefix_length = len(prefix)
    return {k[prefix_length:]: v for k, v in SETTINGS.items() if k.startswith(prefix)}
