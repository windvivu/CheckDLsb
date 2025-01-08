from ML._MakeSample import SampleXY
from ML._ModelClassifier import ModelClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV

class ModelXGBClassifier(ModelClassifier):
    def _note_(self):
        self._note = '''
        Data đầu vào phải là object SampleXY. Object này đã được dựng trước đó, quy định các feature, label, thresoldDiff từ data gôc.
         Ex: SampleXY(data, features=['close','MA5','MA10'], label='close', pastCandle=14, foreCast=5, thresoldDiff=0.02)
        
        '''

    def __init__(self, DataObject:SampleXY, n_estimators=100, max_depth=6, learning_rate=0.1, subsample=1, colsample_bytree=1, gamma=0, useSMOTE=False, verbose=0, internalInfo=True, clipFeatures:list=[], pca_components=None) -> None:
        '''
        pca_components: khai báo nếu muốn giảm chiều dữ liệu
        \nclipFeatures: list collumn muốn cắt bơt sau khi đã make Sample
        '''
        
        super().__init__(DataObject, verbose=verbose, internalInfo=internalInfo, clipFeatures=clipFeatures, pca_components=pca_components)

        self.useSMOTE = useSMOTE
        self.n_estimators = n_estimators # Số lượng cây
        self.max_depth = max_depth # Độ sâu của cây
        self.learning_rate = learning_rate # Tốc độ học
        self.subsample = subsample # Tỷ lệ mẫu được chọn ngẫu nhiên để xây dựng mỗi cây
        self.colsample_bytree = colsample_bytree # Tỷ lệ mẫu được chọn ngẫu nhiên của các cột khi xây dựng mỗi cây
        self.gamma = gamma # Giảm loss function khi tăng độ sâu của cây
    
    def trainModel(self):
        if self.internalInfo: print('Train model XGBoost Classifier...')
        if self.internalInfo: print('\tPrepare data for training/testing...')

        # chuyển đổi nhãn -1, 0, 1 thành 0, 1, 2 (XGBoost yêu cầu nhãn là số nguyên không âm)
        X_train, y_train = self._prepareData(labelEncode=True)

        if self.useSMOTE:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

        
        if self.internalInfo: print('\tTraining model...')

        # multi:softprob => nếu muốn dự đoán xác suất cho từng lớp
        self.model = XGBClassifier(
                                       objective='multi:softmax',  # Mục tiêu phân loại đa lớp
                                       eval_metric = 'mlogloss',    # Đánh giá mô hình
                                       num_class=3,                # Số lượng lớp
                                       n_estimators = self.n_estimators, # Số lượng cây
                                       max_depth = self.max_depth, # Độ sâu của cây
                                       learning_rate = self.learning_rate, # Tốc độ học
                                       subsample = self.subsample, # Tỷ lệ mẫu được chọn ngẫu nhiên để xây dựng mỗi cây
                                       colsample_bytree = self.colsample_bytree, # Tỷ lệ mẫu được chọn ngẫu nhiên của các cột khi xây dựng mỗi cây
                                       gamma = self.gamma, # Giảm loss function khi tăng độ sâu của cây
                                       random_state=42)


        self.model.fit(X_train, y_train)

        self._assessment()
        return self.model
    
    def trainModelGridSearch(self, showProgess=True, useSMOTE=False):
        '''
        ** Lưu ý use SMOTE với dữ liệu không cân bằng
        Chạy trainModel với tham số mặc định.
        Có thể chạy trainModelGridSearch trước để có best model, và bestparams, sau đó chạy trainModel nếu muốn
        training tiếp theo best params
        trainModelGridSearch thường mất thời gian
        '''
        X_train, y_train = self._prepareData(labelEncode=True)

        if useSMOTE:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 3, 4, 5, 6],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2],
        }

         # Khởi tạo mô hình
        xgbc = XGBClassifier(objective='multi:softmax', num_class=3, random_state=42)
        if showProgess:
            grid_search = GridSearchCV(xgbc, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
        else:
            grid_search = GridSearchCV(xgbc, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        self.n_estimators = best_params['n_estimators']
        self.max_depth = best_params['max_depth']
        self.learning_rate = best_params['learning_rate']
        self.subsample = best_params['subsample']
        self.colsample_bytree = best_params['colsample_bytree']
        self.gamma = best_params['gamma']

        print('n_estimators:', self.n_estimators)
        print('max_depth:', self.max_depth)
        print('learning_rate:', self.learning_rate)
        print('subsample:', self.subsample)
        print('colsample_bytree:', self.colsample_bytree)
        print('gamma:', self.gamma)

        self.model = grid_search.best_estimator_

        self._assessment()

        return self.model