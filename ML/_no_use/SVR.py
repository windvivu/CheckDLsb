from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

class ModelSVR:
    def __init__(self):
        self.kernel='rbf'
        self.C=100
        self.epsilon=0.1
        
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
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # Khởi tạo mô hình
        self.model = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon)
        
        self.model.fit(X_train, y_train)

        ypredict = self.model.predict(X_test)
        self.mse = mean_squared_error(y_test, ypredict)
        self.r2 = r2_score(y_test, ypredict)


    def trainModelGridSearch(self, X, Y, showProgess=True):
        '''
        Chạy trainModel với tham số mặc định.
        Có thể chạy trainModelGridSearch trước để có best model, và bestparams, sau đó chạy trainModel nếu muốn
        training tiếp theo best params
        trainModelGridSearch thường mất thời gian
        '''
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Chuẩn hóa dữ liệu
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.1, 0.2, 0.3],
            'kernel': ['rbf', 'linear, poly']
        }

        # Khởi tạo mô hình
        svr = SVR()
        if showProgess:
            grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=2, verbose=1)
        else:
            grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=2, verbose=0)
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        self.kernel = best_params['kernel']
        self.C = best_params['C']
        self.epsilon = best_params['epsilon']
    
        self.model = grid_search.best_estimator_
        
        ypredict = self.model.predict(X_test)
        self.mse = mean_squared_error(y_test, ypredict)
        self.r2 = r2_score(y_test, ypredict)


    def exportParams(self):
        param_grid = {
            'kernel': self.kernel,
            'C': self.C,
            'epsilon': self.epsilon,
        }
        return param_grid
    
    def importParams(self, param_grid):
        self.kernel = param_grid['kernel']
        self.C = param_grid['C']
        self.epsilon = param_grid['epsilon']

    
    def predict(self, Xnew, Ytruenew=None):
        Xnew = self.scaler.transform(Xnew)
        ypredict = self.model.predict(Xnew)
        if Ytruenew is not None:
            self.mse_new = mean_squared_error(Ytruenew, ypredict)
            self.r2_new = r2_score(Ytruenew, ypredict)
        return ypredict
    

