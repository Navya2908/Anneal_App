import shutil
import sqlite3
from datetime import datetime
from os import listdir
from pymongo import MongoClient
import pandas as pd
import os
import csv
import json
#from application_logging.logger import App_Logger


class dBOperation:
    """
      This class shall be used for handling all the SQL operations.

      Written By: iNeuron Intelligence
      Version: 1.0
      Revisions: None

      """
    def __init__(self,loggerObj):
        self.path = 'Training_Database/'
        self.badFilePath = "Training_Raw_files_validated/Bad_Raw"
        self.goodFilePath = "Training_Raw_files_validated/Good_Raw"
        self.loggerObj = loggerObj
        self.features = ['family', 'product-type', 'steel', 'carbon', 'hardness', 'temper_rolling', 'condition',
                         'formability', 'strength', 'non-ageing',
                         'surface-finish', 'surface-quality', 'enamelability', 'bc', 'bf', 'bt', 'bw/me', 'bl', 'm',
                         'chrom', 'phos', 'cbond', 'marvi', 'exptl',
                         'ferro', 'corr', 'blue/bright/varn/clean', 'lustre', 'jurofm', 's', 'p', 'shape', 'thick',
                         'width', 'len', 'oil', 'bore', 'packing',
                         'classes']


    def dataBaseConnection(self):

        """
                Method Name: dataBaseConnection
                Description: This method creates the database with the given name and if Database already exists then opens the connection to the DB.
                Output: Connection to the DB
                On Failure: Raise ConnectionError

                 Written By: iNeuron Intelligence
                Version: 1.0
                Revisions: None

                """
        try:
            client = MongoClient('mongodb://localhost:27017')
            self.loggerObj.logger_log("Connected to database successfully")
        except ConnectionError:
            self.loggerObj.logger_log("Error while connecting to database: %s" %ConnectionError)
            raise ConnectionError
        return client


    def postData(self,Database):

        """
                               Method Name: insertIntoTableGoodData
                               Description: This method inserts the Good data files from the Good_Raw folder into the
                                            above created table.
                               Output: None
                               On Failure: Raise Exception

                                Written By: iNeuron Intelligence
                               Version: 1.0
                               Revisions: None

        """

        client = self.dataBaseConnection()
        db = client[Database]
        goodFilePath= self.goodFilePath
        badFilePath = self.badFilePath
        onlyfiles = [f for f in listdir(goodFilePath)]
        collection_name = 'ready_to_export'
        db_cm = db[collection_name]
        data = pd.DataFrame()
        for file in onlyfiles:
            data = pd.concat([data, pd.read_csv(self.goodFilePath + "/" + file, header=None, names=self.features)], axis=0)
        try:
            data_json = json.loads(data.to_json(orient='records'))
            db_cm.remove()
            db_cm.insert(data_json)
        except Exception as e:
            self.loggerObj.logger_log("Error while posting the csv file: %s " % e)
            shutil.move(goodFilePath+'/' + file, badFilePath)
            self.loggerObj.logger_log("File Moved Successfully %s" % file)
            client.close()

        client.close()


    def selectingDatafromcollectionintocsv(self,Database):

        """
                               Method Name: selectingDatafromtableintocsv
                               Description: This method exports the data in GoodData table as a CSV file. in a given location.
                                            above created .
                               Output: None
                               On Failure: Raise Exception

                                Written By: iNeuron Intelligence
                               Version: 1.0
                               Revisions: None

        """

        fileFromDb = 'Training_FileFromDB/'
        fileName = 'InputFile.csv'
        try:
            client = self.dataBaseConnection()
            conn = client[Database]
            # Make a query to the specific DB and Collection
            collection_name = 'ready_to_export'
            coll = conn[collection_name]
            # make an API call to the MongoDB server using a Collection object
            cursor = coll.find()
            # Expand the cursor and construct the DataFrame
            df = pd.DataFrame(list(cursor))
            # Delete the _id
            if '_id' in df:
                df.drop(columns=['_id'], inplace = True)
            # Now export to csv
            df.to_csv(fileFromDb + fileName, index = False, header=False)
            self.loggerObj.logger_log("Database data successfully exported to csv")
        except Exception as e:
            self.loggerObj.logger_log("File exporting failed. Error : %s" %e)