from ML._MakeSample import SampleXYtseries
from ML._ModelClassifier import ModelClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

class ModelSVC(ModelClassifier):
    def _note_(self):
        self._note = '''

        Data đầu vào phải là object SampleXY. Object này đã được dựng trước đó, quy định các feature, label, thresoldDiff từ data gôc.
         Ex: SampleXY(data, features=['close','MA5','MA10'], label='close', pastCandle=14, foreCast=5, thresoldDiff=0.02)
        
        Kernel:
            linear: Phù hợp cho dữ liệu tuyến tính.
            rbf (Radial Basis Function): Phổ biến nhất, hiệu quả cho nhiều loại dữ liệu.
            poly: Cho dữ liệu phi tuyến tính.
            sigmoid: Ít được sử dụng hơn.
        C (Regularization parameter):
            Kiểm soát trade-off giữa sai số phân loại và độ phức tạp của mô hình.
            C lớn: Mô hình phức tạp hơn, có thể overfitting.
            C nhỏ: Mô hình đơn giản hơn, có thể underfitting.
        gamma (Kernel coefficient cho 'rbf', 'poly' và 'sigmoid'):
            Xác định ảnh hưởng của một mẫu huấn luyện.
            gamma lớn: Mô hình phức tạp, có thể overfitting.
            gamma nhỏ: Mô hình đơn giản, có thể underfitting.
        degree (Cho kernel 'poly'):
            Bậc của hàm đa thức.
        '''
    def __init__(self, DataObject:SampleXYtseries, kernel='rbf', C=100, gamma='scale', degree=3, class_weight='balanced', verbose=0, internalInfo=True) -> None:
        super().__init__(DataObject, class_weight=class_weight, verbose=verbose, internalInfo=internalInfo, )
        
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.class_weight = class_weight
        # set_param_grid_poly=True thì gridsearch rất lâu, chưa rõ lý do
        self.set_param_grid_poly = False
               
    
    def trainModel(self):
        if self.X_train is None:
            print('Data is not prepared yet. Prepare data first.')
            return
        
        title = 'SVM Classifier'       
        if self.internalInfo: print('Train model ' + title + '...')

        self.model = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma, class_weight=self.class_weight, random_state=42)        
        
        self.model.fit(self.X_train, self.y_train)

        if self.internalInfo: 
            self._assessment()
    
    def trainModelGridSearch(self, showProgess=True):
        '''
        Chạy trainModel với tham số mặc định.
        Có thể chạy trainModelGridSearch trước để có best model, và bestparams, sau đó chạy trainModel nếu muốn
        training tiếp theo best params
        trainModelGridSearch thường mất thời gian
        '''

        if self.X_train is None:
            print('Data is not prepared yet. Prepare data first.')
            return

        param_grid = {
            'kernel': ['rbf', 'linear'],
            'C': [0.1, 1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001],
        }

        param_grid_poly = {
            'kernel': ['poly'],
            'C': [0.1, 1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001],
            'degree': [2, 3, 4]  # Chỉ có tác dụng khi kernel='poly'
        }
            
        svc = SVC(class_weight=self.class_weight, random_state=42)
        if showProgess:
            verbose=2
        else:
            verbose=0

        # thật ra self.set_param_grid_poly luôn set False vì True sẽ bị đứng chưa rõ lý do    
        if self.set_param_grid_poly:
            grid_search = GridSearchCV(svc, param_grid_poly, cv=5, scoring='accuracy', n_jobs=-1, verbose=verbose)
        else:
            grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=verbose)

        grid_search.fit(self.X_train, self.y_train)

        self.bestParas = grid_search.best_params_
        
        self.kernel = self.bestParas['kernel']
        self.C = self.bestParas['C'],
        self.gamma = self.bestParas['gamma'],
        if self.set_param_grid_poly: self.degree = self.bestParas['degree']

        print(f'Best parameters: {self.bestParas}')

        self.model = grid_search.best_estimator_

        self._assessment()

        return self.model
    