from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics  import roc_auc_score,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import numpy as np

class Model_Finder:
    """
                This class shall  be used to find the model with best accuracy and AUC score.
                Written By: iNeuron Intelligence
                Version: 1.0
                Revisions: None

                """

    def __init__(self,loggerObj):
        self.loggerObj = loggerObj
        self.clf = RandomForestClassifier()
        self.knn = KNeighborsClassifier()
        self.xgb = XGBClassifier()

    def get_best_params_for_random_forest(self,train_x,train_y):
        """
                                Method Name: get_best_params_for_random_forest
                                Description: get the parameters for Random Forest Algorithm which give the best accuracy.
                                             Use Hyper Parameter Tuning.
                                Output: The model with the best parameters
                                On Failure: Raise Exception

                                Written By: iNeuron Intelligence
                                Version: 1.0
                                Revisions: None

                        """
        self.loggerObj.logger_log('Entered the get_best_params_for_random_forest method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid = {"n_estimators": [10, 50, 100, 130], "criterion": ['gini', 'entropy'],
                               "max_depth": range(2, 4, 1), "max_features": ['auto', 'log2']}

            #Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.clf, param_grid=self.param_grid, cv=5,  verbose=3)
            #finding the best parameters
            self.grid.fit(train_x, train_y)

            #extracting the best parameters
            self.criterion = self.grid.best_params_['criterion']
            self.max_depth = self.grid.best_params_['max_depth']
            self.max_features = self.grid.best_params_['max_features']
            self.n_estimators = self.grid.best_params_['n_estimators']

            #creating a new model with the best parameters
            self.clf = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion,
                                              max_depth=self.max_depth, max_features=self.max_features)
            # training the mew model
            self.clf.fit(train_x, train_y)
            self.loggerObj.logger_log('Random Forest best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_random_forest method of the Model_Finder class')

            return self.clf
        except Exception as e:
            self.loggerObj.logger_log('Exception occured in get_best_params_for_random_forest method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.loggerObj.logger_log('Random Forest Parameter tuning  failed. Exited the get_best_params_for_random_forest method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_xgboost(self,train_x,train_y):
        """
                                Method Name: get_best_params_for_xgboost
                                Description: get the parameters for XG Boost Algorithm which give the best accuracy.
                                             Use Hyper Parameter Tuning.
                                Output: The model with the best parameters
                                On Failure: Raise Exception

                                Written By: iNeuron Intelligence
                                Version: 1.0
                                Revisions: None

                        """
        self.loggerObj.logger_log('Entered the get_best_params_for_xgboost method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid = {'min_child_weight': [1, 5, 10], 'gamma': [0.5, 1, 1.5, 2, 5],
                               'subsample': [0.6, 0.8, 1.0], 'colsample_bytree': [0.6, 0.8, 1.0],
                               'max_depth': [3, 4, 5]}

            #Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.xgb, param_grid=self.param_grid, cv=5,  verbose=3)
            #finding the best parameters
            self.grid.fit(train_x, train_y)

            #extracting the best parameters
            self.colsample_bytree = self.grid.best_params_['colsample_bytree']
            self.gamma = self.grid.best_params_['gamma']
            self.min_child_weight = self.grid.best_params_['min_child_weight']
            self.subsample = self.grid.best_params_['subsample']
            self.max_depth = self.grid.best_params_['max_depth']

            #creating a new model with the best parameters
            self.xgb = XGBClassifier(learning_rate=0.01,colsample_bytree=self.colsample_bytree, gamma=self.gamma,
                                     min_child_weight=self.min_child_weight,subsample=self.subsample,
                                     max_depth=self.max_depth)
            # training the mew model
            self.xgb.fit(train_x, train_y)
            self.loggerObj.logger_log('XGB best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_random_forest method of the Model_Finder class')

            return self.xgb
        except Exception as e:
            self.loggerObj.logger_log('Exception occured in get_best_params_for_xg boost method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.loggerObj.logger_log('XG Boost tuning  failed. Exited the get_best_params_for_random_forest method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_KNN(self, train_x, train_y):
        """
                                                Method Name: get_best_params_for_KNN
                                                Description: get the parameters for KNN Algorithm which give the best accuracy.
                                                             Use Hyper Parameter Tuning.
                                                Output: The model with the best parameters
                                                On Failure: Raise Exception

                                                Written By: iNeuron Intelligence
                                                Version: 1.0
                                                Revisions: None

                                        """
        self.loggerObj.logger_log('Entered the get_best_params_for_Ensembled_KNN method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid_knn = {
                'algorithm' : ['ball_tree', 'kd_tree', 'brute'],
                'leaf_size' : [10,17,24,28,30,35],
                'n_neighbors':[4,5,8,10,11],
                'p':[1,2]
            }

            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(self.knn, self.param_grid_knn, verbose=3,
                                     cv=5)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.algorithm = self.grid.best_params_['algorithm']
            self.leaf_size = self.grid.best_params_['leaf_size']
            self.n_neighbors = self.grid.best_params_['n_neighbors']
            self.p  = self.grid.best_params_['p']

            # creating a new model with the best parameters
            self.knn = KNeighborsClassifier(algorithm=self.algorithm, leaf_size=self.leaf_size, n_neighbors=self.n_neighbors,p=self.p,n_jobs=-1)
            # training the mew model
            self.knn.fit(train_x, train_y)
            self.loggerObj.logger_log('KNN best params: ' + str(
                                       self.grid.best_params_) + '. Exited the KNN method of the Model_Finder class')
            return self.knn
        except Exception as e:
            self.loggerObj.logger_log('Exception occured in knn method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.loggerObj.logger_log( 'knn Parameter tuning  failed. Exited the knn method of the Model_Finder class')
            raise Exception()


    def get_best_model(self,train_x,train_y,test_x,test_y):
        """
                                                Method Name: get_best_model
                                                Description: Find out the Model which has the best AUC score.
                                                Output: The best model name and the model object
                                                On Failure: Raise Exception

                                                Written By: iNeuron Intelligence
                                                Version: 1.0
                                                Revisions: None

                                        """
        self.loggerObj.logger_log('Entered the get_best_model method of the Model_Finder class')
        # create best model for KNN
        try:
            self.knn= self.get_best_params_for_KNN(train_x,train_y)
            self.prediction_knn = self.knn.predict_proba(test_x) # Predictions using the KNN Model

            if len(np.unique(test_y)) == 1: #if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.knn_score = accuracy_score(test_y, self.prediction_knn)
                self.loggerObj.logger_log('Accuracy for knn:' + str(self.knn_score))  # Log AUC
            else:
                self.knn_score = roc_auc_score(test_y, self.prediction_knn, multi_class='ovr') # AUC for KNN
                self.loggerObj.logger_log('AUC for knn:' + str(self.knn_score)) # Log AUC

            # create best model for Random Forest
            self.random_forest=self.get_best_params_for_random_forest(train_x,train_y)
            self.prediction_random_forest=self.random_forest.predict_proba(test_x) # prediction using the Random Forest Algorithm

            if len(np.unique(test_y)) == 1:#if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.random_forest_score = accuracy_score((test_y),self.prediction_random_forest)
                self.loggerObj.logger_log('Accuracy for RF:' + str(self.random_forest_score))
            else:
                self.random_forest_score = roc_auc_score((test_y), self.prediction_random_forest,multi_class='ovr') # AUC for Random Forest
                self.loggerObj.logger_log('AUC for RF:' + str(self.random_forest_score))

            # create best model for XG Boost
            self.xg_boost = self.get_best_params_for_xgboost(train_x, train_y)
            self.prediction_xg_boost = self.xg_boost.predict_proba(test_x)  # prediction using the XG boost

            if len(np.unique(test_y)) == 1:#if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.xg_boost_score = accuracy_score((test_y),self.prediction_xg_boost)
                self.loggerObj.logger_log('Accuracy for XGB:' + str(self.xg_boost_score))
            else:
                self.xg_boost_score = roc_auc_score((test_y), self.prediction_xg_boost,multi_class='ovr') # AUC for XG boost
                self.loggerObj.logger_log('AUC for XGB:' + str(self.xg_boost_score))
            print('KNN score: ',self.knn_score)
            print('Random forest score: ',self.random_forest_score)
            print('XG Boost score: ',self.xg_boost_score)

            #comparing the two models
            if(self.random_forest_score <  self.knn_score):
                if (self.xg_boost_score < self.knn_score):
                    return 'KNN',self.knn
                else:
                    return 'XGBoost',self.xg_boost
            else:
                if (self.xg_boost_score < self.random_forest_score):
                    return 'RandomForest', self.random_forest
                else:
                    return 'XGBoost', self.xg_boost


        except Exception as e:
            self.loggerObj.logger_log('Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.loggerObj.logger_log('Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()