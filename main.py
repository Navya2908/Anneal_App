from flask import Flask, request, render_template
from flask import Response
import os
import json
from datetime import datetime
import pandas as pd
from flask_cors import CORS, cross_origin



from Training_Module.trainingModel import trainModel
from Training_Module.train_data_validation import train_validation
from Prediction_Module.predict_data_validation import predict_validation
from Prediction_Module.predictionFromModel import prediction
#import flask_monitoringdashboard as dashboard
from application_logging.logger import App_Logger


os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
#dashboard.bind(app)
CORS(app)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRouteClient():
    loggerObj = App_Logger()
    try:
        if request.json is not None:
            content = json.loads(request.get_json())
            pred_valObj = predict_validation(loggerObj) # object initialization
            pred_valObj.singleRecValidation(content)  # calling the prediction_validation function
            pred_obj = prediction(loggerObj)  # object initialization
            result = pred_obj.predictionFromModel(content)  # calling the function to predict the data
            return Response("Predicted class is %s!!!" % result)
        elif request.form is not None:
            path = request.form['filepath']

            pred_valObj = predict_validation(loggerObj,path) # object initialization
            pred_valObj.prediction_validation() #calling the prediction_validation function
            pred_obj = prediction(loggerObj) # object initialization
            path = pred_obj.predictionFromModel() # calling the function to predict the data
            return Response("Prediction File created at %s!!!" % path)

    except ValueError:
        print("Error Occurred! " +str(ValueError))
        return Response("Error Occurred! %s" %str(ValueError))
    except KeyError:
        print("Error Occurred! " + str(KeyError))
        return Response("Error Occurred! %s" %KeyError)
    except Exception as e:
        print("Error Occurred! " +str(e))
        return Response("Error Occurred! %s" %e)

@app.route("/predict_csv", methods=['POST'])
@cross_origin()
def predictInputFile():
    filedir = 'Prediction_Batch_Files'
    loggerObj = App_Logger()
    now = datetime.now()
    df = pd.DataFrame()
    currdate = now.strftime("%d%m%Y")
    currtime = now.strftime("%H%M%S%f")
    features = ['family', 'product-type', 'steel', 'carbon', 'hardness', 'temper_rolling', 'condition', 'formability',
                'strength', 'non-ageing',
                'surface-finish', 'surface-quality', 'enamelability', 'bc', 'bf', 'bt', 'bw/me', 'bl', 'm', 'chrom',
                'phos', 'cbond', 'marvi', 'exptl',
                'ferro', 'corr', 'blue/bright/varn/clean', 'lustre', 'jurofm', 's', 'p', 'shape', 'thick', 'width',
                'len', 'oil', 'bore', 'packing']
    try:
        if request.form is not None:
            df = pd.read_csv(request.files.get("CsvDoc"), header=None, names=features)
        # Now export to csv
        df.to_csv(filedir + '/' + 'Annealing_' + currdate + '_' + currtime + '.csv', index=False, header=False)

        pred_valObj = predict_validation(loggerObj, filedir)  # object initialization
        pred_valObj.prediction_validation()  # calling the prediction_validation function
        pred_obj = prediction(loggerObj)  # object initialization
        path = pred_obj.predictionFromModel()  # calling the function to predict the data
        return Response("Prediction File created at %s!!!" % path)
    except ValueError:
        print("Error Occurred! " + str(ValueError))
        return Response("Error Occurred! %s" % str(ValueError))
    except KeyError:
        print("Error Occurred! " + str(KeyError))
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        print("Error Occurred! " + str(e))
        return Response("Error Occurred! %s" % e)


@app.route("/train", methods=['POST'])
@cross_origin()
def trainRouteClient():
    loggerObj = App_Logger()
    try:
        if request.form is not None:
            path = request.form['filepath']
            valObj = train_validation(path, loggerObj)  # object initialization
            valObj.training_validation()  # calling the training_validation function

            train_obj = trainModel(loggerObj)  # object initialization
            train_obj.trainingModel()  # training the model for the files in the table

    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)
    return Response("Training successfull!!")


if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
    app.run(debug=True)