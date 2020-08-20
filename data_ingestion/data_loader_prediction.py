import pandas as pd

class Data_Getter_Pred:
    """
    This class shall  be used for obtaining the data from the source for prediction.

    Written By: iNeuron Intelligence
    Version: 1.0
    Revisions: None

    """
    def __init__(self, loggerObj):
        self.prediction_file='Prediction_FileFromDB/InputFile.csv'
        self.loggerObj = loggerObj
        self.features = ['family', 'product-type', 'steel', 'carbon', 'hardness', 'temper_rolling', 'condition',
                         'formability', 'strength', 'non-ageing',
                         'surface-finish', 'surface-quality', 'enamelability', 'bc', 'bf', 'bt', 'bw/me', 'bl', 'm',
                         'chrom', 'phos', 'cbond', 'marvi', 'exptl',
                         'ferro', 'corr', 'blue/bright/varn/clean', 'lustre', 'jurofm', 's', 'p', 'shape', 'thick',
                         'width', 'len', 'oil', 'bore', 'packing']

    def get_data(self):
        """
        Method Name: get_data
        Description: This method reads the data from source.
        Output: A pandas DataFrame.
        On Failure: Raise Exception

         Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

        """
        self.loggerObj.logger_log('Entered the get_data method of the Data_Getter class')
        try:
            self.data= pd.read_csv(self.prediction_file, header = None, names = self.features) # reading the data file
            self.loggerObj.logger_log('Data Load Successful.Exited the get_data method of the Data_Getter class')
            return self.data
        except Exception as e:
            self.loggerObj.logger_log('Exception occured in get_data method of the Data_Getter class. Exception message: '+str(e))
            self.loggerObj.logger_log('Data Load Unsuccessful.Exited the get_data method of the Data_Getter class')
            raise Exception()

    def get_data_for_rec(self,jsondata):
        """
        Method Name: get_data
        Description: This method reads the data from source.
        Output: A pandas DataFrame.
        On Failure: Raise Exception

         Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

        """
        self.loggerObj.logger_log('Entered the get_data_for_rec method of the Data_Getter class')
        try:
            data = pd.DataFrame([jsondata]) # reading the data file
            self.loggerObj.logger_log('Data Load Successful.Exited the get_data method of the Data_Getter class')
            return data
        except Exception as e:
            self.loggerObj.logger_log('Exception occured in get_data method of the Data_Getter class. Exception message: '+str(e))
            self.loggerObj.logger_log('Data Load Unsuccessful.Exited the get_data method of the Data_Getter class')
            raise Exception()