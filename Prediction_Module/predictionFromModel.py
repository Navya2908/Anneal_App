import pandas
from file_operations import file_methods
from data_preprocessing import preprocessing
from data_ingestion import data_loader_prediction
from file_operations import file_methods
import pickle


class prediction:

    def __init__(self,loggerObj):
        self.loggerObj = loggerObj

    def predictionFromModel(self,singlerecdata = None):

        try:
            self.loggerObj.logger_log('Start of Prediction')
            data_getter = data_loader_prediction.Data_Getter_Pred(self.loggerObj)
            if singlerecdata is None:
                data = data_getter.get_data()
            else:
                data = data_getter.get_data_for_rec(singlerecdata)

            preprocessor=preprocessing.Preprocessor(self.loggerObj)

            # repalcing '?' values with np.nan as discussed in the EDA part
            data = preprocessor.replaceInvalidValuesWithNull(data)

            data = preprocessor.dropUnnecessaryColumnsForPrediction(data)

            # check if missing values are present in the dataset
            is_null_present = preprocessor.is_null_present(data)
            if (is_null_present):
                data = preprocessor.impute_missing_values(data)

            data = preprocessor.encodeCategoricalValuesPrediction(data)

            data = preprocessor.prediction_data_standardisation(data)

            file_loader = file_methods.File_Operation(self.loggerObj)
            model_name = file_loader.find_model_file()
            model = file_loader.load_model(model_name)

            result = []  # initialize balnk list for storing predicitons
            with open('EncoderPickle/enc.pickle',
                      'rb') as file:  # let's load the encoder pickle file to decode the values
                encoder = pickle.load(file)
            if singlerecdata is None:
                for val in (encoder.inverse_transform(model.predict(data))):
                    result.append(val)
                result = pandas.DataFrame(result, columns=['Predictions'])
                path = "Prediction_Output_File/Predictions.csv"
                result.to_csv("Prediction_Output_File/Predictions.csv",
                              header=True)  # appends result to prediction file
                self.loggerObj.logger_log('End of Prediction')
                return path
            else:
                val = encoder.inverse_transform(model.predict(data))
                self.loggerObj.logger_log('End of Prediction')
                return val

        except Exception as ex:
            self.loggerObj.logger_log('Error occured while running the prediction!! Error:: %s' % ex)
            raise ex




