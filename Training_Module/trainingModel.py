# Doing necessary imports
from application_logging import logger
from data_ingestion import data_loader
from data_preprocessing import preprocessing
from sklearn.model_selection import train_test_split
from best_model_finder import tuner
from file_operations import file_methods



class trainModel:

    def __init__(self,loggerObj):
        self.loggerObj = loggerObj

    def trainingModel(self):
        # Logging start of the training
        self.loggerObj.logger_log("Start of TrainingModel")
        try:
            # Getting the data from the source
            data_getter = data_loader.Data_Getter(self.loggerObj)
            data = data_getter.get_data()

            """doing the data preprocessing"""

            preprocessor = preprocessing.Preprocessor(self.loggerObj)

            # repalcing '?' values with np.nan as discussed in the EDA part
            data = preprocessor.replaceInvalidValuesWithNull(data)

            # Drop the columns which are having missing values more than 50% of total observations
            data = preprocessor.dropUnnecessaryColumns(data)

            # check if missing values are present in the dataset
            is_null_present = preprocessor.is_null_present(data)

            # if missing values are there, impute them appropriately.
            if (is_null_present):
                data = preprocessor.impute_missing_values(data)  # missing value imputation

            # create separate features and labels
            X, y = preprocessor.separate_label_feature(data, label_column_name='classes')

            # Categorical encoding
            X, y = preprocessor.encodeCategoricalValues(X,y)

            # Handling imbalance dataset using SMOTE
            #X, y = preprocessor.handleImbalanceDataset(X, y)

            # splitting the data into training and test set
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=44)

            # Standardisation of X_train data
            X_train = preprocessor.data_standardisation(X_train)

            # applying same standardisation object on X_test data
            X_test = preprocessor.prediction_data_standardisation(X_test)

            model_finder = tuner.Model_Finder(self.loggerObj)  # object initialization

            # getting the best model for each of the clusters
            best_model_name, best_model = model_finder.get_best_model(X_train, y_train, X_test, y_test)

            # saving the best model to the directory.
            file_op = file_methods.File_Operation(self.loggerObj)
            save_model = file_op.save_model(best_model, best_model_name)

        except Exception:
            # Logging the unsuccessful training
            self.loggerObj.logger_log("Unsuccessful end of training")
            raise Exception



