# Setting up global logger
# copied from openbox's logger

import os
import logging
from datetime import datetime

DEFAULT_FORMAT = "(%(levelname)s) %(asctime)s [%(filename)s:%(lineno)d] %(message)s"

logger = None

def init_logger(name="DSE4WSE", level="INFO", stream=True, logdir=None, 
                fmt=DEFAULT_FORMAT, force_init=True):
    global logger

    # only init once if force_init is False
    if logger is not None and not force_init:
        return

    if logdir is None:
        logfile = None
    else:
        os.makedirs(logdir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        logfile = os.path.join(logdir, '%s_%s.log' % (name, timestamp))
        logfile = os.path.abspath(logfile)

    logger = logging.Logger(name, level)
    if logfile:
        plain_formatter = logging.Formatter(fmt)
        file_handler = logging.FileHandler(filename=logfile, mode='a', encoding='utf8')
        file_handler.setFormatter(plain_formatter)
        logger.addHandler(file_handler)
    if stream:
        formatter = logging.Formatter(fmt)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.debug("logger init.")
    if logfile is not None:
        logger.info(f"Logfile {logfile}")

init_logger(name="DSE4WSE", level="DEBUG", stream=True, logdir=None, 
            fmt=DEFAULT_FORMAT, force_init=True)