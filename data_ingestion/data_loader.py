import pandas as pd
from os import walk

class Data_Getter:
    """
    This class shall  be used for obtaining the data from the source for training.

    Written By: iNeuron Intelligence
    Version: 1.0
    Revisions: None

    """
    def __init__(self, loggerObj):
        self.training_path = 'Training_FileFromDB'
        self.loggerObj=loggerObj
        self.features = ['family','product-type','steel','carbon','hardness','temper_rolling','condition','formability','strength','non-ageing',
                         'surface-finish','surface-quality','enamelability','bc','bf','bt','bw/me','bl','m','chrom','phos','cbond','marvi','exptl',
                         'ferro','corr','blue/bright/varn/clean','lustre','jurofm','s','p','shape','thick','width','len','oil','bore','packing',
                         'classes']

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
            training_files = []
            # store all the input files in the directory to list
            for (dir_path, dir_name, file_names) in walk(self.training_path):
                training_files.extend(file_names)
                break
            data = pd.DataFrame()
            # Load all the input files into single data frame
            for file in training_files:
                data = pd.concat([data, pd.read_csv(self.training_path + '/' + file, header=None, names = self.features)], axis=0)
            self.loggerObj.logger_log('Data Load Successful.Exited the get_data method of the Data_Getter class')
            return data
        except Exception as e:
            self.loggerObj.logger_log('Exception occured in get_data method of the Data_Getter class. Exception message: '+str(e))
            self.loggerObj.logger_log('Data Load Unsuccessful.Exited the get_data method of the Data_Getter class')
            raise Exception()


