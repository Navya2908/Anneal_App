from datetime import datetime
from Prediction_Module.dsa_validation import Raw_Data_validation
from Prediction_Module.db_import_export import dBOperation

class predict_validation:
    def __init__(self,loggerObj,path = None):
        self.raw_data = Raw_Data_validation(path,loggerObj)
        self.dBOperation = dBOperation(loggerObj)
        self.loggerObj = loggerObj

    def prediction_validation(self):
        try:
            self.loggerObj.logger_log('Start of Validation on files for prediction!!')
            self.raw_data.validation()
            self.loggerObj.logger_log("Raw Data Validation for prediction Complete!!")

            self.loggerObj.logger_log("Creating Prediction_Database and tables on the basis of given schema!!!")
            # create database with given name, if present open the connection! Create table with columns given in schema
            self.dBOperation.postData('Prediction')
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
            self.dBOperation.selectingDatafromcollectionintocsv('Prediction')
            # Removing previous prediction before starting prediction
            self.raw_data.deletePredictionFile()
            self.loggerObj.logger_log("Successfully extracted the prediction data from database")

        except Exception as e:
            raise e

    def singleRecValidation(self,jsonData):
        try:
            self.loggerObj.logger_log('Start of Validation on files for prediction!!')
            self.raw_data.recValidation(jsonData)
            self.loggerObj.logger_log("Raw Data Validation for prediction Complete!!")

            self.loggerObj.logger_log("Creating Prediction_Database and pushing the data!!!")
            # create database with given name, if present open the connection! Create table with columns given in schema
            self.dBOperation.postDataForSingleRec('Prediction',jsonData)
            self.loggerObj.logger_log("Posting the data to database Completed!!")
        except Exception as e:
            raise e