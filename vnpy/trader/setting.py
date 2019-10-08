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
    "rqdata.password": "TpGWvd1xClPLrR3laSMdvSKm2actLEQEEdudVl2pnjria7al8xLVgiL2GbZLFyno5oRRQAsuGWy819FVB-nX_Ijqtu9XNPOfjAzMBbSsK6macZIUvhAYFZFxn5uJPA8uI7TPHEsfIDDoVZICRbzJO5GvNv6gVsmeS288wr5GnPU=N1NkuNyT3kfR92q04tVJH7GmIXW1g8N0ph_DIbbGLlTJKLd_fruKVEZqOOPF_FfQdO6W-OblX8pz3ammz-lb3GnoKdaNPa11FaWeLCaFxdJFw841E_8pfDJ8Hc9-QDP45ldiRkXTQoHdc1NeEMhVOnv3yE7YVZhpc9Jp5gctt_g=",

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
