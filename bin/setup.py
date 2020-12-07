#!/usr/bin/env python3
# coding: utf-8
# packages
import warnings
warnings.filterwarnings("ignore")
import logCreate
import packageInstallation
import sqlConnection
import confReader
import getSignals

def main():
	logger = logCreate.create_logger()
	config = confReader.get_config(logger)
	pi=packageInstallation.main()
	sql=sqlConnection.main()
	nsesig=getSignals.main()

if __name__ == "__main__":
		main()
		
