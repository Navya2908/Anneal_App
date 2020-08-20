from datetime import datetime
from Training_Module.dsa_validation import Raw_Data_validation
from Training_Module.db_import_export import dBOperation

class train_validation:
    def __init__(self,path,loggerObj):
        self.raw_data = Raw_Data_validation(path,loggerObj)
        #self.dataTransform = dataTransform()
        self.dBOperation = dBOperation(loggerObj)
        #self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.loggerObj = loggerObj

    def training_validation(self):
        try:
            self.loggerObj.logger_log('Start of Validation')
            self.raw_data.validation()

            self.loggerObj.logger_log("Creating Training_Database and importing on the basis of given schema!!!")
            # create database with given name, if present open the connection! Create table with columns given in schema
            self.dBOperation.postData('Training')
            self.loggerObj.logger_log("Posting the data to database Completed!!")

            # Delete the good data folder after loading files in table
            self.raw_data.deleteExistingGoodDataTrainingFolder()
            self.loggerObj.logger_log("Good_Data folder deleted!!!")
            self.loggerObj.logger_log("Moving bad files to Archive and deleting Bad_Data folder!!!")
            # Move the bad files to archive folder
            self.raw_data.moveBadFilesToArchiveBad()
            self.loggerObj.logger_log("Bad files moved to archive!! Bad folder Deleted!!")
            self.loggerObj.logger_log("Validation Operation completed!!")
            self.loggerObj.logger_log("Extracting csv file from table")
            # export data in table to csvfile
            self.dBOperation.selectingDatafromcollectionintocsv('Training')
            self.loggerObj.logger_log("Successfully extracted the training data from database")

        except Exception as e:
            raise e