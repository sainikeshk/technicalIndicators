#!/usr/bin/env python3
# coding: utf-8
# packages
import warnings
warnings.filterwarnings("ignore")
import os
import subprocess
import sys
import re 
import logging
import datetime as dt

def create_logger():
    logger = logging.getLogger(__name__)
    if not os.path.exists('../log'):
        os.makedirs('../log')
    dt_str = str(dt.datetime.now()).replace('-','').replace(':', '').split('.')[0].replace(' ', '.' )
    logging.basicConfig(filename='../log/'+str(dt_str)+'.nse.nsesignals.log', filemode='a', format='%(process)d  %(asctime)s %(levelname)s %(funcName)s %(lineno)d ::: %(message)s', level=logging.INFO)
    logger.info("******************log file created*****************")
    return logger
if __name__ == '__main__':
    logger = create_logger()
