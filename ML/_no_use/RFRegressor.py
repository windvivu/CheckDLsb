from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,  r2_score
from sklearn.model_selection import GridSearchCV
import pandas as pd

class ModelRFRegression:
    def __init__(self, optionScaler=False):
        self.n_estimators=100
        self.max_depth=10
        self.min_samples_split=2
        self.min_samples_leaf=1
        self.max_features=1

        self.optionScaler = optionScaler

        self.feature_importances = None
        self.model = None
        self.mse = None
        self.r2 = None
        self.mse_new = None
        self.r2_new = None
    
    def trainModel(self, X, Y):
        '''
        Chạy trainModel với tham số mặc định.
        Có thể chạy trainModelGridSearch trước để có best model, và bestparams, sau đó chạy trainModel nếu muốn
        training tiếp theo best params
        trainModelGridSearch thường mất thời gian
        '''
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Chuẩn hóa dữ liệu
        if self.optionScaler:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        # Khởi tạo mô hình
        self.model = RandomForestRegressor(n_estimators=self.n_estimators, 
                                             max_depth=self.max_depth,
                                             min_samples_split=self.min_samples_split,
                                             min_samples_leaf=self.min_samples_leaf,
                                             max_features=self.max_features,
                                           random_state=42)
        
        self.model.fit(X_train, y_train)

        ypredict = self.model.predict(X_test)
        self.mse = mean_squared_error(y_test, ypredict)
        self.r2 = r2_score(y_test, ypredict)
        self.feature_importances = self.model.feature_importances_


    def trainModelGridSearch(self, X, Y, showProgess=True):
        '''
        Chạy trainModel với tham số mặc định.
        Có thể chạy trainModelGridSearch trước để có best model, và bestparams, sau đó chạy trainModel nếu muốn
        training tiếp theo best params
        trainModelGridSearch thường mất thời gian
        '''
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Chuẩn hóa dữ liệu
        if self.optionScaler:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt']
        }

        # Khởi tạo mô hình
        rf = RandomForestRegressor(random_state=42)
        if showProgess:
            grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=2, verbose=1)
        else:
            grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=2, verbose=0)
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        self.n_estimators = best_params['n_estimators']
        self.max_depth = best_params['max_depth']
        self.min_samples_split = best_params['min_samples_split']
        self.min_samples_leaf = best_params['min_samples_leaf']
        self.max_features = best_params['max_features']
    
        self.model = grid_search.best_estimator_
        
        ypredict = self.model.predict(X_test)
        self.mse = mean_squared_error(y_test, ypredict)
        self.r2 = r2_score(y_test, ypredict)
        self.feature_importances = self.model.feature_importances_

    
    def printImportantFeatures(self, inputCol=None):
        '''
        Xem tầm quan trọng của các đặc trưng

        inputCol đưa vào phải tường ững với các feature đã đưa vào để fit training
        '''
        if inputCol is None:
            feature_importance = pd.DataFrame({'importance': self.feature_importances})
        else:
            feature_importance = pd.DataFrame({'feature': inputCol, 'importance': self.feature_importances})
        print(feature_importance.sort_values('importance', ascending=False))

    def exportParams(self):
        param_grid = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features
        }
        return param_grid
    
    def importParams(self, param_grid):
        self.n_estimators = param_grid['n_estimators']
        self.max_depth = param_grid['max_depth']
        self.min_samples_split = param_grid['min_samples_split']
        self.min_samples_leaf = param_grid['min_samples_leaf']
        self.max_features = param_grid['max_features']
    
    def predict(self, Xnew, Ytruenew=None):
        if self.optionScaler:
            Xnew = self.scaler.transform(Xnew)
        ypredict = self.model.predict(Xnew)
        if Ytruenew is not None:
            self.mse_new = mean_squared_error(Ytruenew, ypredict)
            self.r2_new = r2_score(Ytruenew, ypredict)
        return ypredict
    

