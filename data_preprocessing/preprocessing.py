import pandas as pd
import numpy as np
#from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import pickle
#from imblearn.over_sampling import RandomOverSampler


class Preprocessor:
    """
        This class shall  be used to clean and transform the data before training.

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

        """

    def __init__(self, loggerObj):
        self.loggerObj = loggerObj

    def remove_columns(self,data,columns):
        """
                Method Name: remove_columns
                Description: This method removes the given columns from a pandas dataframe.
                Output: A pandas DataFrame after removing the specified columns.
                On Failure: Raise Exception

                Written By: iNeuron Intelligence
                Version: 1.0
                Revisions: None

        """
        self.loggerObj.logger_log('Entered the remove_columns method of the Preprocessor class')
        try:
            useful_data=data.drop(labels=columns, axis=1) # drop the labels specified in the columns
            self.loggerObj.logger_log('Column removal Successful.Exited the remove_columns method of the Preprocessor class')
            return useful_data
        except Exception as e:
            self.loggerObj.logger_log('Exception occured in remove_columns method of the Preprocessor class. Exception message:  '+str(e))
            self.loggerObj.logger_log('Column removal Unsuccessful. Exited the remove_columns method of the Preprocessor class')
            raise Exception()

    def separate_label_feature(self, data, label_column_name):
        """
                        Method Name: separate_label_feature
                        Description: This method separates the features and a Label Coulmns.
                        Output: Returns two separate Dataframes, one containing features and the other containing Labels .
                        On Failure: Raise Exception

                        Written By: iNeuron Intelligence
                        Version: 1.0
                        Revisions: None

                """
        self.loggerObj.logger_log('Entered the separate_label_feature method of the Preprocessor class')
        try:
            self.X=data.drop(labels=label_column_name,axis=1) # drop the columns specified and separate the feature columns
            self.Y=data[label_column_name] # Filter the Label columns
            self.loggerObj.logger_log('Label Separation Successful. Exited the separate_label_feature method of the Preprocessor class')
            return self.X,self.Y
        except Exception as e:
            self.loggerObj.logger_log('Exception occured in separate_label_feature method of the Preprocessor class. Exception message:  ' + str(e))
            self.loggerObj.logger_log('Label Separation Unsuccessful. Exited the separate_label_feature method of the Preprocessor class')
            raise Exception()

    def dropUnnecessaryColumns(self,data):
        """
                        Method Name: is_null_present
                        Description: This method drops the unwanted columns as discussed in EDA section.

                        Written By: iNeuron Intelligence
                        Version: 1.0
                        Revisions: None

                                """
        self.loggerObj.logger_log('Entered the dropUnnecessary method of the Preprocessor class')
        try:
            # Finding features which are having missing values more than 50% of data
            mis_columns = []
            for column in data.columns:
                count = data[column].isnull().sum()
                rec = data.shape[0] * 0.5
                if (count > rec):
                    mis_columns.append(column)
            if len(mis_columns) > 0:
                # Drop the features which have missing values more than 50%
                data.drop(columns=mis_columns, inplace=True)
                with open('data_preprocessing/UnnecessaryColumns.pickle', 'wb') as file:
                    pickle.dump(mis_columns, file)
            # Drop variable 'product-type', because it have only one value.So SD becomes zero
            data = data.drop(columns=['product-type'])
            return data
        except Exception as e:
            self.loggerObj.logger_log('Exception occured in dropUnnecessary method of the Preprocessor class. Exception message:  ' + str(e))
            self.loggerObj.logger_log('Dropping unnecessary columns failed. Exited the dropUnnecessary method of the Preprocessor class')
            raise Exception()

    def dropUnnecessaryColumnsForPrediction(self,data):
        """
                        Method Name: is_null_present
                        Description: This method drops the unwanted columns as discussed in EDA section.

                        Written By: iNeuron Intelligence
                        Version: 1.0
                        Revisions: None

                                """
        self.loggerObj.logger_log('Entered the dropUnnecessary method of the Preprocessor class')
        try:
            # Finding features which are having missing values more than 50% of data
            with open('data_preprocessing/UnnecessaryColumns.pickle', 'rb') as file:
                mis_columns = pickle.load(file)
            if len(mis_columns) > 0:
                # Drop the features which have missing values more than 50%
                data.drop(columns=mis_columns, inplace=True)
            # Drop variable 'product-type', because it have only one value.So SD becomes zero
            data = data.drop(columns = ['product-type'])
            return data
        except Exception as e:
            self.loggerObj.logger_log('Exception occured in dropUnnecessary method of the Preprocessor class. Exception message:  ' + str(e))
            self.loggerObj.logger_log('Dropping unnecessary columns failed. Exited the dropUnnecessary method of the Preprocessor class')
            raise Exception()


    def replaceInvalidValuesWithNull(self,data):

        """
                               Method Name: is_null_present
                               Description: This method replaces invalid values i.e. '?' with null, as discussed in EDA.

                               Written By: iNeuron Intelligence
                               Version: 1.0
                               Revisions: None

                                       """

        for column in data.columns:
            count = data[column][data[column] == '?'].count()
            if count != 0:
                data[column] = data[column].replace('?', np.nan)
        return data

    def is_null_present(self,data):
        """
                                Method Name: is_null_present
                                Description: This method checks whether there are null values present in the pandas Dataframe or not.
                                Output: Returns a Boolean Value. True if null values are present in the DataFrame, False if they are not present.
                                On Failure: Raise Exception

                                Written By: iNeuron Intelligence
                                Version: 1.0
                                Revisions: None

                        """
        self.loggerObj.logger_log('Entered the is_null_present method of the Preprocessor class')
        self.null_present = False
        try:
            self.null_counts=data.isna().sum() # check for the count of null values per column
            for i in self.null_counts:
                if i>0:
                    self.loggerObj.logger_log('There are null values present in the data')
                    self.null_present=True
                    break
            self.loggerObj.logger_log('Finding missing values is a success.')
            return self.null_present
        except Exception as e:
            self.loggerObj.logger_log('Exception occured in is_null_present method of the Preprocessor class. Exception message:  ' + str(e))
            self.loggerObj.logger_log('Finding missing values failed. Exited the is_null_present method of the Preprocessor class')
            raise Exception()

    def encodeCategoricalValues(self,X,y):
        """
                                        Method Name: encodeCategoricalValues
                                        Description: This method encodes all the categorical values in the training set.
                                        Output: A Dataframe which has all the categorical values encoded.
                                        On Failure: Raise Exception

                                        Written By: iNeuron Intelligence
                                        Version: 1.0
                                        Revisions: None
                     """

        ct = ColumnTransformer(transformers=[('encoding',OneHotEncoder(),[0,3,4,6,7])],remainder = 'passthrough')
        ct.fit(X)
        X = np.array(ct.transform(X))
        with open('EncoderPickle/onehotenc.pickle', 'wb') as file:
            pickle.dump(ct, file)


        encode = LabelEncoder().fit(y.values)
        y = encode.transform(y.values)

        # we will save the encoder as pickle to use when we do the prediction. We will need to decode the predcited values
        # back to original
        with open('EncoderPickle/enc.pickle', 'wb') as file:
            pickle.dump(encode, file)

        return X, y


    def encodeCategoricalValuesPrediction(self,data):
        """
                                               Method Name: encodeCategoricalValuesPrediction
                                               Description: This method encodes all the categorical values in the prediction set.
                                               Output: A Dataframe which has all the categorical values encoded.
                                               On Failure: Raise Exception

                                               Written By: iNeuron Intelligence
                                               Version: 1.0
                                               Revisions: None
                            """

        # Get the encoder
        with open('EncoderPickle/onehotenc.pickle', 'rb') as file:
            ct = pickle.load(file)
        data = np.array(ct.transform(data))

        return data

    def handleImbalanceDataset(self,X,Y):
        """
                                                      Method Name: handleImbalanceDataset
                                                      Description: This method handles the imbalance in the dataset by oversampling.
                                                      Output: A Dataframe which is balanced now.
                                                      On Failure: Raise Exception

                                                      Written By: iNeuron Intelligence
                                                      Version: 1.0
                                                      Revisions: None
                                   """

        self.loggerObj.logger_log('Entered the handleImbalanceDataset method of the Preprocessor class')
        try:
            oversample = SMOTE()
            x_sampled, y_sampled = oversample.fit_resample(X, Y)
            self.loggerObj.logger_log('successfully completed handleImbalanceDataset')
            return x_sampled, y_sampled
        except Exception as e:
            self.loggerObj.logger_log('Exception occured in handleImbalanceDataset method of the Preprocessor class. Exception message:  ' + str(e))
            raise Exception()

    def impute_missing_values(self, data):
        """
                                        Method Name: impute_missing_values
                                        Description: This method replaces all the missing values in the Dataframe using KNN Imputer.
                                        Output: A Dataframe which has all the missing values imputed.
                                        On Failure: Raise Exception

                                        Written By: iNeuron Intelligence
                                        Version: 1.0
                                        Revisions: None
                     """
        self.loggerObj.logger_log('Entered the impute_missing_values method of the Preprocessor class')
        try:
            # According to the feature discription there is one more category in all the below features. So considering missing values as new category
            data['steel'].fillna(value='U', inplace=True)
            data['condition'].fillna(value='X', inplace=True)
            data['formability'].fillna(value='4', inplace=True)
            data['surface-quality'].fillna(value='H', inplace=True)
            self.loggerObj.logger_log( 'Imputing missing values Successful. Exited the impute_missing_values method of the Preprocessor class')
            return data
        except Exception as e:
            self.loggerObj.logger_log('Exception occured in impute_missing_values method of the Preprocessor class. Exception message:  ' + str(e))
            self.loggerObj.logger_log('Imputing missing values failed. Exited the impute_missing_values method of the Preprocessor class')
            raise Exception()

    def get_columns_with_zero_std_deviation(self,data):
        """
                                                Method Name: get_columns_with_zero_std_deviation
                                                Description: This method finds out the columns which have a standard deviation of zero.
                                                Output: List of the columns with standard deviation of zero
                                                On Failure: Raise Exception

                                                Written By: iNeuron Intelligence
                                                Version: 1.0
                                                Revisions: None
                             """
        self.loggerObj.logger_log( 'Entered the get_columns_with_zero_std_deviation method of the Preprocessor class')
        self.columns=data.columns
        self.data_n = data.describe()
        self.col_to_drop=[]
        try:
            for x in self.columns:
                if (self.data_n[x]['std'] == 0): # check if standard deviation is zero
                    self.col_to_drop.append(x)  # prepare the list of columns with standard deviation zero
            self.loggerObj.logger_log('Column search for Standard Deviation of Zero Successful. Exited the get_columns_with_zero_std_deviation method of the Preprocessor class')
            return self.col_to_drop

        except Exception as e:
            self.loggerObj.logger_log('Exception occured in get_columns_with_zero_std_deviation method of the Preprocessor class. Exception message:  ' + str(e))
            self.loggerObj.logger_log('Column search for Standard Deviation of Zero Failed. Exited the get_columns_with_zero_std_deviation method of the Preprocessor class')
            raise Exception()

    def data_standardisation(self,data):
        """
                                                Method Name: get_columns_with_zero_std_deviation
                                                Description: This method finds out the columns which have a standard deviation of zero.
                                                Output: List of the columns with standard deviation of zero
                                                On Failure: Raise Exception

                                                Written By: iNeuron Intelligence
                                                Version: 1.0
                                                Revisions: None
                             """
        self.loggerObj.logger_log('Entered the do standardisation method of the Preprocessor class')
        try:
            scaler = StandardScaler()
            scaler.fit(data[:,23:30])
            data[:,23:30] = scaler.transform(data[:,23:30])

            # we will save the scaler as pickle to use when we do the prediction. We will need to scale the predcited values
            # by using same scaler
            with open('ScalerPickle/scaler.pickle', 'wb') as file:
                pickle.dump(scaler, file)

            return data
        except Exception as e:
            self.loggerObj.logger_log('Exception occured in get_columns_with_zero_std_deviation method of the Preprocessor class. Exception message:  ' + str(e))
            self.loggerObj.logger_log('Column search for Standard Deviation of Zero Failed. Exited the get_columns_with_zero_std_deviation method of the Preprocessor class')
            raise Exception()

    def prediction_data_standardisation(self,data):
        """
                                                Method Name: get_columns_with_zero_std_deviation
                                                Description: This method finds out the columns which have a standard deviation of zero.
                                                Output: List of the columns with standard deviation of zero
                                                On Failure: Raise Exception

                                                Written By: iNeuron Intelligence
                                                Version: 1.0
                                                Revisions: None
                             """
        self.loggerObj.logger_log('Entered the prediction_data_standardisation method of the Preprocessor class')
        try:
            with open('ScalerPickle/scaler.pickle', 'rb') as file:
                scaler = pickle.load(file)
            data[:,23:30] = scaler.transform(data[:,23:30])

            return data
        except Exception as e:
            self.loggerObj.logger_log('Exception occured in get_columns_with_zero_std_deviation method of the Preprocessor class. Exception message:  ' + str(e))
            self.loggerObj.logger_log('Column search for Standard Deviation of Zero Failed. Exited the get_columns_with_zero_std_deviation method of the Preprocessor class')
            raise Exception()