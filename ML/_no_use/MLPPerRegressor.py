from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

class ModelMLPerRegressor:
    def __init__(self):
        '''
        - Cấu trúc mạng: Trong ví dụ này, chúng ta sử dụng một mạng với hai lớp ẩn (100 và 50 nút). Bạn có thể điều chỉnh cấu trúc này tùy theo độ phức tạp của vấn đề.
        
        - Tham số mô hình: 'relu' là hàm kích hoạt, 'adam' là bộ tối ưu hóa. Bạn có thể thử nghiệm với các giá trị khác nhau.
        
        - Số lần lặp tối đa: Được đặt là 1000, nhưng có thể cần điều chỉnh tùy thuộc vào dữ liệu của bạn.
        
        - Overfitting: Mạng neural có thể dễ bị overfitting. Bạn có thể xem xét sử dụng validation set hoặc cross-validation để kiểm soát điều này.
        '''
        self.hidden_layer_sizes = (100, 50)
        self.activation = 'relu'
        self.solver = 'adam'
        self.max_iter = 1000
    
        self.model = None
        self.mse = None
        self.r2 = None
        self.mse_new = None
        self.r2_new = None
    
    def trainModel(self, X, Y):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Chuẩn hóa dữ liệu
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # Khởi tạo mô hình
        # Khởi tạo và huấn luyện mô hình
        self.model = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes, activation=self.activation, 
                                  solver=self.solver, max_iter=self.max_iter, random_state=42)
        
        self.model.fit(X_train, y_train)

        ypredict = self.model.predict(X_test)
        self.mse = mean_squared_error(y_test, ypredict)
        self.r2 = r2_score(y_test, ypredict)
    
    
    def predict(self, Xnew, Ytruenew=None):
        Xnew = self.scaler.transform(Xnew)
        ypredict = self.model.predict(Xnew)
        if Ytruenew is not None:
            self.mse_new = mean_squared_error(Ytruenew, ypredict)
            self.r2_new = r2_score(Ytruenew, ypredict)
        return ypredict