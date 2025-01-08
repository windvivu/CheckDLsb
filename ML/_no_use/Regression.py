from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,  r2_score

class ModelRegression:
    def __init__(self, optionRegresion:int=0, optionScaler=False):
        '''
        0: mặc định sẽ dùng LinearRegression
        
        2: Hồi quy Ridge (L2 Regularization)
        
        1: Hồi quy Lasso (L1 Regularization)
        
        3: ElasticNet (L1 + L2 Regularization)
        '''
        self.optionRegresion = optionRegresion
        self.optionScaler = optionScaler
        self.model = None
        self.mse = None
        self.r2 = None
        self.mse_new = None
        self.r2_new = None
    
    def trainModel(self, X, Y):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Chuẩn hóa dữ liệu
        if self.optionScaler:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        # Khởi tạo mô hình
        if self.optionRegresion == 1:
            self.model = Lasso(alpha=0.1, max_iter=5000)
        elif self.optionRegresion == 2:
            self.model = Ridge(alpha=1.0)
        elif self.optionRegresion == 3:
            self.model = ElasticNet(alpha=0.1, l1_ratio=0.5)
        else:
            self.model = LinearRegression()
        
        self.model.fit(X_train, y_train)

        ypredict = self.model.predict(X_test)
        self.mse = mean_squared_error(y_test, ypredict)
        self.r2 = r2_score(y_test, ypredict)
    
    
    def predict(self, Xnew, Ytruenew=None):
        if self.optionScaler:
            Xnew = self.scaler.transform(Xnew)
        ypredict = self.model.predict(Xnew)
        if Ytruenew is not None:
            self.mse_new = mean_squared_error(Ytruenew, ypredict)
            self.r2_new = r2_score(Ytruenew, ypredict)
        return ypredict