				                 Python package for 
=======================================================================================================================================================
Version             :signals and decisions
Any licenses needed :Nil
Licensed to         :Maveric systems
Developer           :Sai Nikesh
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'signals and decisions' is the packaged file to create nse technical indicators signals and decisions

Prerequisites
-------------------------------------------------------------------------------------------------------------------------------------------------------
Step1 - Download and install python-3.6.5 version and mysql-installer-community-8.0.19.0 
Step2 - Download and extract the zip file of signals and decisions
Step3 - You can change username and password, hostname, of Mysql under etc/signalsnse.conf
Step4 - Make sure you have all admin rights for installing packages
Step5 - nsedailybhavhist and nsesymbols tables of stockanalytics should be present in database before running this package
Folder structure
=======================================================================================================================================================
Step1 - bin folder contains all the .py files for executing
Step2 - etc folder is having 'signalsnse.conf' file, there we can change inputdbprop
Step3 - lib folder contains 
	1.'requirement.txt' file which contain all the packages which is used for this program
Step4 - sql folder is having DDL queries for creating database, using database, and creating table query
Step5 -log folder is for creating log for the program

Program structure
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Step1 - 'confReader.py' is for reading 'signalsnse.conf' file
Step2 - 'packageInstallation.py' is for installing all the necessary packages (this packages will sucessfully install if only you have admin privilege)
Step3 - 'sqlConnection.py' is for executing the query which is there inside sql folder
Step4 - 'getsignals.py' is for inserting records in nsesignals table which contains various technical indicator values and it's respective decisions of latest dates
Step5 - 'logCreate.py' is for creating logs in log folder
Step6- 'setup.py' is for executing all the programs which is mentioned above in one execution
=============================================End of Process=============================================


