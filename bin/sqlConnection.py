#!/usr/bin/env python3
# coding: utf-8
# packages
import warnings
warnings.filterwarnings("ignore")
# Importing the libraries
import mysql.connector
import confReader
import logCreate

def sqlConnet(user,password,host,logger):
	try:
		cnx = mysql.connector.connect(user=user, password=password,host=host,autocommit=True)
		cur=cnx.cursor()
		logger.info("******************connected to mysql******************")
		return cur,cnx
	except Exception as e:
		logger.info('error:: some issue for connecting to mysql ...check textdata.conf file inputdbprop in etc folder....')
		logger.exception('error :: connection error %s',e)
		raise e
def tableCreation(sqlConnectionPath,dbname,tablename,cur,cnx,logger):
	try:
		with open(sqlConnectionPath, "rt") as f:
			count=0
			for line in f:
				l = line.strip()
				query = l.split(';')
				query=query[0]
				if count <=1:
					query=query.replace("?",dbname)
				if count ==2:
					query=query.replace("?",tablename)
				try:
					cur.execute(query)
					cnx.commit()
					if count ==0:
						logger.info('******************mysql database created***************')
						logger.info('query == %s',query)
					if count ==2:
						logger.info('******************mysql table created******************')
						logger.info('query == %s',query)
				except Exception as e:
					logger.exception('error :: table creation error %s',e)
					logger.warning("warning :: table creation warning == %s", e)
					pass
				count = count+1
		logger.info('******************sqlConnection.py execution completed******************')
	except Exception as e:
		logger.info('error:: some issue for creating table in mysql ...check textdata.conf file inputdbprop,tablename in etc folder and textdata.sql in sql folder....')
		logger.exception('error :: table creation error %s',e)
		raise e

def main():
	logger =logCreate.create_logger()
	logger.info("******************sqlConnection.py start executing*****************")
	try:
		config = confReader.get_config(logger)
		logger.info("config file data :: %s",config)
	except Exception as e:
		logger.exception('error:: some issue in reading the config...check confi_reader.py script in bin folder..%s',e)
		raise e
	
	inputdbprop = config.get("inputdbprop")
	sqlConnectionPath=config.get("sqlconnection")
	dbname=inputdbprop.get("dbname")
	tablename=config.get("tablename")
	cur,cnx = sqlConnet(inputdbprop.get('dbusername'),inputdbprop.get('dbpassword'),inputdbprop.get('hostname'),logger)  
	tablecreation=tableCreation(sqlConnectionPath,dbname,tablename,cur,cnx,logger)
if __name__ == "__main__":
		main()
