from datetime import datetime
from os import listdir
import os
import re
import json
import shutil
import pandas as pd





class Raw_Data_validation:
    """
               This class shall be used for handling all the validation done on the Raw Prediction Data!!.

               Written By: iNeuron Intelligence
               Version: 1.0
               Revisions: None

               """

    def __init__(self,path,loggerObj):
        self.Batch_Directory = path
        self.schema_path = 'Prediction_Module/schema_prediction.json'
        self.loggerObj = loggerObj


    def valuesFromSchema(self):
        """
                                Method Name: valuesFromSchema
                                Description: This method extracts all the relevant information from the pre-defined "Schema" file.
                                Output: LengthOfDateStampInFile, LengthOfTimeStampInFile, column_names, Number of Columns
                                On Failure: Raise ValueError,KeyError,Exception

                                 Written By: iNeuron Intelligence
                                Version: 1.0
                                Revisions: None

                                        """
        try:
            with open(self.schema_path, 'r') as f:
                dic = json.load(f)
                f.close()
            pattern = dic['SampleFileName']
            LengthOfDateStampInFile = dic['LengthOfDateStampInFile']
            LengthOfTimeStampInFile = dic['LengthOfTimeStampInFile']
            column_names = dic['ColName']
            NumberofColumns = dic['NumberofColumns']

            message ="LengthOfDateStampInFile:: %s" %LengthOfDateStampInFile + "\t" + "LengthOfTimeStampInFile:: %s" % LengthOfTimeStampInFile +"\t " + "NumberofColumns:: %s" % NumberofColumns + "\n"
            self.loggerObj.logger_log(message)



        except ValueError:
            self.loggerObj.logger_log("ValueError:Value not found inside schema_training.json")
            raise ValueError

        except KeyError:
            self.loggerObj.logger_log("KeyError:Key value error incorrect key passed")
            raise KeyError

        except Exception as e:
            self.loggerObj.logger_log(str(e))
            raise e

        return LengthOfDateStampInFile, LengthOfTimeStampInFile, column_names, NumberofColumns


    def manualRegexCreation(self):

        """
                                      Method Name: manualRegexCreation
                                      Description: This method contains a manually defined regex based on the "FileName" given in "Schema" file.
                                                  This Regex is used to validate the filename of the prediction data.
                                      Output: Regex pattern
                                      On Failure: None

                                       Written By: iNeuron Intelligence
                                      Version: 1.0
                                      Revisions: None

                                              """
        regex = "['Annealing']+['\_'']+[\d_]+[\d]+\.csv"
        return regex

    def createDirectoryForGoodBadRawData(self):

        """
                                        Method Name: createDirectoryForGoodBadRawData
                                        Description: This method creates directories to store the Good Data and Bad Data
                                                      after validating the prediction data.

                                        Output: None
                                        On Failure: OSError

                                         Written By: iNeuron Intelligence
                                        Version: 1.0
                                        Revisions: None

                                                """
        try:
            path = os.path.join("Prediction_Raw_Files_Validated/", "Good_Raw/")
            if not os.path.isdir(path):
                os.makedirs(path)
            path = os.path.join("Prediction_Raw_Files_Validated/", "Bad_Raw/")
            if not os.path.isdir(path):
                os.makedirs(path)

        except OSError as ex:
            self.loggerObj.logger_log("Error while creating Directory %s:" % ex)
            raise OSError

    def deleteExistingGoodDataTrainingFolder(self):
        """
                                            Method Name: deleteExistingGoodDataTrainingFolder
                                            Description: This method deletes the directory made to store the Good Data
                                                          after loading the data in the table. Once the good files are
                                                          loaded in the DB,deleting the directory ensures space optimization.
                                            Output: None
                                            On Failure: OSError

                                             Written By: iNeuron Intelligence
                                            Version: 1.0
                                            Revisions: None

                                                    """
        try:
            path = 'Prediction_Raw_Files_Validated/'
            if os.path.isdir(path + 'Good_Raw/'):
                shutil.rmtree(path + 'Good_Raw/')
                self.loggerObj.logger_log("GoodRaw directory deleted successfully!!!")
        except OSError as s:
            self.loggerObj.logger_log("Error while Deleting Directory : %s" %s)
            raise OSError
    def deleteExistingBadDataTrainingFolder(self):

        """
                                            Method Name: deleteExistingBadDataTrainingFolder
                                            Description: This method deletes the directory made to store the bad Data.
                                            Output: None
                                            On Failure: OSError

                                             Written By: iNeuron Intelligence
                                            Version: 1.0
                                            Revisions: None

                                                    """

        try:
            path = 'Prediction_Raw_Files_Validated/'
            if os.path.isdir(path + 'Bad_Raw/'):
                shutil.rmtree(path + 'Bad_Raw/')
                self.loggerObj.logger_log("BadRaw directory deleted before starting validation!!!")
        except OSError as s:
            self.loggerObj.logger_log("Error while Deleting Directory : %s" %s)
            raise OSError

    def moveBadFilesToArchiveBad(self):


        """
                                            Method Name: moveBadFilesToArchiveBad
                                            Description: This method deletes the directory made  to store the Bad Data
                                                          after moving the data in an archive folder. We archive the bad
                                                          files to send them back to the client for invalid data issue.
                                            Output: None
                                            On Failure: OSError

                                             Written By: iNeuron Intelligence
                                            Version: 1.0
                                            Revisions: None

                                                    """
        now = datetime.now()
        date = now.date()
        time = now.strftime("%H%M%S")
        try:
            path= "PredictionArchivedBadData"
            if not os.path.isdir(path):
                os.makedirs(path)
            source = 'Prediction_Raw_Files_Validated/Bad_Raw/'
            dest = 'PredictionArchivedBadData/BadData_' + str(date)+"_"+str(time)
            if not os.path.isdir(dest):
                os.makedirs(dest)
            files = os.listdir(source)
            for f in files:
                if f not in os.listdir(dest):
                    shutil.move(source + f, dest)
            self.loggerObj.logger_log("Bad files moved to archive")
            path = 'Prediction_Raw_Files_Validated/'
            if os.path.isdir(path + 'Bad_Raw/'):
                shutil.rmtree(path + 'Bad_Raw/')
            self.loggerObj.logger_log("Bad Raw Data Folder Deleted successfully!!")
        except OSError as e:
            self.loggerObj.logger_log("Error while moving bad files to archive:: %s" % e)
            raise OSError




    def validationFileNameRaw(self,regex,LengthOfDateStampInFile,LengthOfTimeStampInFile):
        """
            Method Name: validationFileNameRaw
            Description: This function validates the name of the prediction csv file as per given name in the schema!
                         Regex pattern is used to do the validation.If name format do not match the file is moved
                         to Bad Raw Data folder else in Good raw data.
            Output: None
            On Failure: Exception

             Written By: iNeuron Intelligence
            Version: 1.0
            Revisions: None

        """
        # delete the directories for good and bad data in case last run was unsuccessful and folders were not deleted.
        self.deleteExistingBadDataTrainingFolder()
        self.deleteExistingGoodDataTrainingFolder()
        self.createDirectoryForGoodBadRawData()
        onlyfiles = [f for f in listdir(self.Batch_Directory)]
        try:
            for filename in onlyfiles:
                if (re.match(regex, filename)):
                    splitAtDot = re.split('.csv', filename)
                    splitAtDot = (re.split('_', splitAtDot[0]))
                    if len(splitAtDot[1]) == LengthOfDateStampInFile:
                        if len(splitAtDot[2]) == LengthOfTimeStampInFile:
                            shutil.copy("Prediction_Batch_files/" + filename, "Prediction_Raw_Files_Validated/Good_Raw")
                            self.loggerObj.logger_log("Valid File name!! File moved to GoodRaw Folder :: %s" % filename)

                        else:
                            shutil.copy("Prediction_Batch_files/" + filename, "Prediction_Raw_Files_Validated/Bad_Raw")
                            self.loggerObj.logger_log("Invalid File Name!! File moved to Bad Raw Folder :: %s" % filename)
                    else:
                        shutil.copy("Prediction_Batch_files/" + filename, "Prediction_Raw_Files_Validated/Bad_Raw")
                        self.loggerObj.logger_log("Invalid File Name!! File moved to Bad Raw Folder :: %s" % filename)
                else:
                    shutil.copy("Prediction_Batch_files/" + filename, "Prediction_Raw_Files_Validated/Bad_Raw")
                    self.loggerObj.logger_log("Invalid File Name!! File moved to Bad Raw Folder :: %s" % filename)

        except Exception as e:
            self.loggerObj.logger_log("Error occured while validating FileName %s" % e)
            raise e




    def validateColumnCount(self,NumberofColumns):
        """
                    Method Name: validateColumnLength
                    Description: This function validates the number of columns in the csv files.
                                 It is should be same as given in the schema file.
                                 If not same file is not suitable for processing and thus is moved to Bad Raw Data folder.
                                 If the column number matches, file is kept in Good Raw Data for processing.
                                The csv file is missing the first column name, this function changes the missing name to "Wafer".
                    Output: None
                    On Failure: Exception

                     Written By: iNeuron Intelligence
                    Version: 1.0
                    Revisions: None

             """
        try:
            self.loggerObj.logger_log("Column Length Validation Started!!")
            for file in listdir('Prediction_Raw_Files_Validated/Good_Raw/'):
                csv = pd.read_csv("Prediction_Raw_Files_Validated/Good_Raw/" + file)
                if csv.shape[1] == NumberofColumns:
                    pass
                else:
                    shutil.move("Prediction_Raw_Files_Validated/Good_Raw/" + file, "Prediction_Raw_Files_Validated/Bad_Raw")
                    self.loggerObj.logger_log("Invalid Column Length for the file!! File moved to Bad Raw Folder :: %s" % file)

            self.loggerObj.logger_log("Column Length Validation Completed!!")
        except OSError:
            self.loggerObj.logger_log("Error Occured while moving the file :: %s" % OSError)
            raise OSError
        except Exception as e:
            self.loggerObj.logger_log("Error Occured:: %s" % e)
            raise e

    def validateColumnCountForRec(self, jsonData, NumberofColumns):
        """
                    Method Name: validateColumnCountForRec
                    Description: This function validates the number of columns in the single record.
                                 It is should be same as given in the schema file.
                                 If not same file is not suitable for processing and thus is moved to Bad Raw Data folder.
                                 If the column number matches, file is kept in Good Raw Data for processing.
                                The csv file is missing the first column name, this function changes the missing name to "Wafer".
                    Output: None
                    On Failure: Exception

                     Written By: iNeuron Intelligence
                    Version: 1.0
                    Revisions: None

             """
        try:
            self.loggerObj.logger_log("Column Length for single record Validation Started!!")

            df = pd.DataFrame([jsonData])
            if df.shape[1] == NumberofColumns:
                pass
            else:
                raise Exception("Total number of  variables does not match" , df.shape[1])

            self.loggerObj.logger_log("Column Length Validation for single record Completed!!")
        except OSError:
            self.loggerObj.logger_log("Error Occured while moving the file :: %s" % OSError)
            raise OSError
        except Exception as e:
            self.loggerObj.logger_log("Error Occured:: %s" % e)
            raise e

    def deletePredictionFile(self):

        if os.path.exists('Prediction_Output_File/Predictions.csv'):
            os.remove('Prediction_Output_File/Predictions.csv')

    def validateColumnLength(self, NumberofColumns):
        """
                    Method Name: validateColumnLength
                    Description: This function validates the number of columns in the csv files.
                                 It is should be same as given in the schema file.
                                 If not same file is not suitable for processing and thus is moved to Bad Raw Data folder.
                                 If the column number matches, file is kept in Good Raw Data for processing.
                                The csv file is missing the first column name, this function changes the missing name to "Wafer".
                    Output: None
                    On Failure: Exception

                     Written By: iNeuron Intelligence
                    Version: 1.0
                    Revisions: None

             """
        try:
            self.loggerObj.logger_log("Column Length Validation Started!!")
            for file in listdir('Prediction_Raw_Files_Validated/Good_Raw/'):
                csv = pd.read_csv("Prediction_Raw_Files_Validated/Good_Raw/" + file)
                if csv.shape[1] == NumberofColumns:
                    csv.rename(columns={"Unnamed: 0": "Wafer"}, inplace=True)
                    csv.to_csv("Prediction_Raw_Files_Validated/Good_Raw/" + file, index=None, header=True)
                else:
                    shutil.move("Prediction_Raw_Files_Validated/Good_Raw/" + file,
                                "Prediction_Raw_Files_Validated/Bad_Raw")
                    self.loggerObj.logger_log("Invalid Column Length for the file!! File moved to Bad Raw Folder :: %s" % file)

            self.loggerObj.logger_log("Column Length Validation Completed!!")
        except OSError:
            self.loggerObj.logger_log("Error Occured while moving the file :: %s" % OSError)
            raise OSError
        except Exception as e:
            self.loggerObj.logger_log("Error Occured:: %s" % e)
            raise e

    def validateMissingValuesInWholeColumn(self):
        """
                                  Method Name: validateMissingValuesInWholeColumn
                                  Description: This function validates if any column in the csv file has all values missing.
                                               If all the values are missing, the file is not suitable for processing.
                                               SUch files are moved to bad raw data.
                                  Output: None
                                  On Failure: Exception

                                   Written By: iNeuron Intelligence
                                  Version: 1.0
                                  Revisions: None

                              """
        try:
            f = open("Prediction_Logs/missingValuesInColumn.txt", 'a+')
            self.loggerObj.logger_log("Missing Values Validation Started!!")

            for file in listdir('Prediction_Raw_Files_Validated/Good_Raw/'):
                csv = pd.read_csv("Prediction_Raw_Files_Validated/Good_Raw/" + file)
                count = 0
                for columns in csv:
                    if (len(csv[columns]) - csv[columns].count()) == len(csv[columns]):
                        count+=1
                        shutil.move("Prediction_Raw_Files_Validated/Good_Raw/" + file,
                                    "Prediction_Raw_Files_Validated/Bad_Raw")
                        self.loggerObj.logger_log("Invalid Column Length for the file!! File moved to Bad Raw Folder :: %s" % file)
                        break
                if count==0:
                    csv.rename(columns={"Unnamed: 0": "Wafer"}, inplace=True)
                    csv.to_csv("Prediction_Raw_Files_Validated/Good_Raw/" + file, index=None, header=True)
        except OSError:
            self.loggerObj.logger_log("Error Occured while moving the file :: %s" % OSError)
            raise OSError
        except Exception as e:
            self.loggerObj.logger_log("Error Occured:: %s" % e)
            raise e

    def removeFilesFromPredictionBatchFiles(self):
        """
                                            Method Name: removeFilesFromPredictionBatchFiles
                                            Description: This method deletes the directory made to store the Good Data
                                                          after loading the data in the table. Once the good files are
                                                          loaded in the DB,deleting the directory ensures space optimization.
                                            Output: None
                                            On Failure: OSError

                                             Written By: iNeuron Intelligence
                                            Version: 1.0
                                            Revisions: None

                                                    """
        try:
            shutil.rmtree('Prediction_Batch_Files' + '/')
            os.mkdir('Prediction_Batch_Files')
            self.loggerObj.logger_log("Removed validated files successfully!!!")
        except OSError as s:
            self.loggerObj.logger_log("Error while removing files from prediction_batch_files directory : %s" %s)
            raise OSError

    def isFilesPresentInBatchDirectory(self):
        """
                                            Method Name: isFilesPresentInBatchDirectory
                                            Description: This method deletes the directory made to store the Good Data
                                                          after loading the data in the table. Once the good files are
                                                          loaded in the DB,deleting the directory ensures space optimization.
                                            Output: None
                                            On Failure: OSError

                                             Written By: iNeuron Intelligence
                                            Version: 1.0
                                            Revisions: None

                                                    """
        try:
            onlyfiles = [f for f in listdir(self.Batch_Directory)]
            if len(onlyfiles) == 0:
                raise Exception("There are no files present to predict")
            self.loggerObj.logger_log("Removed validated files successfully!!!")
        except Exception as e:
            self.loggerObj.logger_log("There are no input files to train")
            raise e

    def validation(self):
        # check files present in source directory
        self.isFilesPresentInBatchDirectory()
        # extracting values from prediction schema
        LengthOfDateStampInFile, LengthOfTimeStampInFile, column_names, noofcolumns = self.valuesFromSchema()
        # getting the regex defined to validate filename
        regex = self.manualRegexCreation()
        # validating filename of training files
        self.validationFileNameRaw(regex, LengthOfDateStampInFile, LengthOfTimeStampInFile)
        # validating column length in the file
        self.validateColumnCount(noofcolumns)
        # once validation is successful remove the input files from the directory
        self.removeFilesFromPredictionBatchFiles()
        self.loggerObj.logger_log("Raw Data Validation Complete!!")

    def recValidation(self,jsonData):
        # extracting values from prediction schema
        LengthOfDateStampInFile, LengthOfTimeStampInFile, column_names, noofcolumns = self.valuesFromSchema()
        # validating column length in the file
        self.loggerObj.logger_log("Validation on number of variables!!")
        self.validateColumnCountForRec(jsonData,noofcolumns)
        # validating if any column has all values missing
        #self.validateMissingValuesInWholeColumn()  Need to analyse more in this function"""
        self.loggerObj.logger_log("Single record validation complete Validation Complete!!")