#!/usr/bin/env python3
# coding: utf-8
# packages
import warnings
warnings.filterwarnings("ignore")
import subprocess
import sys
import os
import confReader
import logCreate
from pkg_resources import WorkingSet , DistributionNotFound
from setuptools.command.easy_install import main as install

def sys_installed_packages(logger):
	logger.info("******************packageInstallation.py start executing*****************")
	try:
		reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])	
	except:
		try:
			reqs = subprocess.check_output([sys.executable, '-m', 'pip3', 'freeze'])
		except Exception as e:
			logger.info('error::please install python3 in your system')
			logger.exception('error :: python error == %s',e)
			raise e
	installed_packages = [r.decode().split('==')[0] for r in reqs.split()]
	logger.info('******************pip or pip3 is checking already installed packages in system ******************')
	return installed_packages

def sys_install_packages(installed_packages,requirements,logger):
	try:
		packages=[]
		with open(requirements, "rt") as f:
			for line in f:
				l = line.strip()
				package = l.split(',')
				package=package[0]
				packages.append(package)
		logger.info('******************collected necessary packages for the smooth execution of the program from requirements.txt file in lib folder******************')
	except Exception as e:
		logger.exception('error :: requirements == %s',e)

	try:
		for i in packages:
			if i not in installed_packages:
				working_set = WorkingSet()
				try:
					dep = working_set.require('paramiko>=1.0')
				except DistributionNotFound:
					pass
				whoami=os.getlogin()
				if whoami =='root':
					install_package=install([i])
				if whoami !='root':
					try:
						install_package=subprocess.check_call(["pip", "install","--user", i])
					except:
						try:
							install_package=subprocess.check_call(["pip3", "install","--user", i])
						except Exception as e:
							logger.exception('error :: installation error == %s',e)
							logger.warning("warning :: installation warning == %s", e)
							logger.info('******************check whether this user has admin privileges for installing package******************')
	except Exception as e:
		logger.exception('error :: requirements == %s',e)
	logger.info('******************packageInstallation.py completed successfully******************')
def main():
	logger =logCreate.create_logger()
	try:
		config = confReader.get_config(logger)
		logger.info("config file data :: %s",config)
	except Exception as e:
		logger.exception('error:: some issue in reading the config...check confi_reader.py script in bin folder..%s',e)
		raise e
	requirements = config.get("requirements",None)	
	installed_packages= sys_installed_packages(logger)
	install_packages=sys_install_packages(installed_packages,requirements,logger)
if __name__ == "__main__":
		main()