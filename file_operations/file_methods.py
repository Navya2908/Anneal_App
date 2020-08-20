import pickle
import os
import shutil


class File_Operation:
    """
                This class shall be used to save the model after training
                and load the saved model for prediction.

                Written By: iNeuron Intelligence
                Version: 1.0
                Revisions: None

                """
    def __init__(self,loggerObj):
        self.loggerObj = loggerObj
        self.model_directory='model'

    def save_model(self,model,filename):
        """
            Method Name: save_model
            Description: Save the model file to directory
            Outcome: File gets saved
            On Failure: Raise Exception

            Written By: iNeuron Intelligence
            Version: 1.0
            Revisions: None
"""
        self.loggerObj.logger_log( 'Entered the save_model method of the File_Operation class')
        try:
            # Before saving model remove the old files
            shutil.rmtree('model' + '/')
            os.mkdir('model')
            with open(self.model_directory +'/' + filename+'.sav', 'wb') as f:
                pickle.dump(model, f) # save the model to file
            self.loggerObj.logger_log('Model File '+filename+' saved. Exited the save_model method of the Model_Finder class')

            return 'success'
        except Exception as e:
            self.loggerObj.logger_log('Exception occured in save_model method of the Model_Finder class. Exception message:  ' + str(e))
            self.loggerObj.logger_log('Model File '+filename+' could not be saved. Exited the save_model method of the Model_Finder class')
            raise Exception()

    def load_model(self,filename):
        """
                    Method Name: load_model
                    Description: load the model file to memory
                    Output: The Model file loaded in memory
                    On Failure: Raise Exception

                    Written By: iNeuron Intelligence
                    Version: 1.0
                    Revisions: None
        """
        self.loggerObj.logger_log('Entered the load_model method of the File_Operation class')
        try:
            with open(self.model_directory + '/' + filename,'rb') as f:
                self.loggerObj.logger_log('Model File ' + filename + ' loaded. Exited the load_model method of the Model_Finder class')
                return pickle.load(f)
        except Exception as e:
            self.loggerObj.logger_log('Exception occured in load_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.loggerObj.logger_log('Model File ' + filename + ' could not be saved. Exited the load_model method of the Model_Finder class')
            raise Exception()
    def find_model_file(self):
        """
                            Method Name: find_correct_model_file
                            Description: Select the correct model based on cluster number
                            Output: The Model file
                            On Failure: Raise Exception

                            Written By: iNeuron Intelligence
                            Version: 1.0
                            Revisions: None
                """
        self.loggerObj.logger_log('Entered the find_model_file method of the File_Operation class')
        try:
            folder_name=self.model_directory
            model_name = os.listdir(folder_name)[0]
            return model_name
        except Exception as e:
            self.loggerObj.logger_log('Exception occured in find_correct_model_file method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.loggerObj.logger_log( 'Exited the find_correct_model_file method of the Model_Finder class with Failure')
            raise Exception()