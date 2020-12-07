#!/usr/bin/env python3
# coding: utf-8
# packages
import warnings
warnings.filterwarnings("ignore")
import os
import re 
import logging
import datetime as dt
import logCreate

usedPropertiesKeys = ['sqlconnection', 'inputdbprop','tablename','requirements']
def get_config(logger):
    logger.info("******************confReader.py start executing*****************")
    try:
        configDict=dict()      
        with open('../etc/signalsnse.conf', "rt") as f:
            for line in f:
                if not line.startswith('#'):
                    l = line.strip()
                    key_value = l.split('=')
                    key = key_value[0].strip()
                    key_value = l.replace(' ','').split(key+'=')
                    configDict[key] = ' '.join(key_value[1:]).strip(' "')
                
            configDict = {k:(int(v) if v.isnumeric() else v ) for k,v in configDict.items() if k in usedPropertiesKeys}
            if 'inputdbprop' in configDict and configDict['inputdbprop']:
                configDict['inputdbprop'] = eval(configDict['inputdbprop'])
            
            f.close()
            logger.info("************ confReader.py executed successfully ***************")
            return configDict
    except Exception as e:
        logger.exception('error:: some issue in conf...check etc/financial.conf.... ::%s', e)
        raise e

if __name__ == '__main__':
    logger = logCreate.create_logger()
    try:
        config = get_config(logger)
    except Exception as e:
        logger.exception('ERROR:: some issue in reading the config...check confReader.py script in bin folder....')
        raise e 
    print(config)


