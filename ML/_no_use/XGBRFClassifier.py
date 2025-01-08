from ML._MakeSample import SampleXY
from ML._ModelClassifier import ModelClassifier
from xgboost import XGBRFClassifier
from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.model_selection import GridSearchCV

class ModelXGBRFClassifier(ModelClassifier):
    def _note_(self):
        self._note = '''
        Data đầu vào phải là object SampleXY. Object này đã được dựng trước đó, quy định các feature, label, thresoldDiff từ data gôc.
         Ex: SampleXY(data, features=['close','MA5','MA10'], label='close', pastCandle=14, foreCast=5, thresoldDiff=0.02)
        
        '''

    def __init__(self, DataObject:SampleXY, n_estimators=100, max_depth=None, useSMOTE=False, verbose=0, internalInfo=True, clipFeatures:list=[], pca_components=None) -> None:
        '''
        pca_components: khai báo nếu muốn giảm chiều dữ liệu
        \nclipFeatures: list collumn muốn cắt bơt sau khi đã make Sample
        '''
        
        super().__init__(DataObject, verbose=verbose, internalInfo=internalInfo, clipFeatures=clipFeatures, pca_components=pca_components)

        self.useSMOTE = useSMOTE
        self.n_estimators = n_estimators # Số lượng cây
        self.max_depth = max_depth # Độ sâu của cây
    
    def trainModel(self):
        if self.internalInfo: print('Train model XGBoost Random Forest Classifier...')
        if self.internalInfo: print('\tPrepare data for training/testing...')

        # chuyển đổi nhãn -1, 0, 1 thành 0, 1, 2 (XGBoost yêu cầu nhãn là số nguyên không âm)
        X_train, y_train = self._prepareData(labelEncode=True)

        if self.useSMOTE:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

        
        if self.internalInfo: print('\tTraining model...')

        # multi:softprob => nếu muốn dự đoán xác suất cho từng lớp
        self.model = XGBRFClassifier  (
                                       objective='multi:softmax',  # Mục tiêu phân loại đa lớp
                                       eval_metric = 'mlogloss',    # Đánh giá mô hình
                                       num_class=3,                # Số lượng lớp
                                       n_estimators = self.n_estimators, # Số lượng cây
                                       max_depth = self.max_depth, # Độ sâu của cây
                                       random_state=42)


        self.model.fit(X_train, y_train)

        self._assessment()
        return self.model

    def feature_importance(self, returnDf=True, printOut=True):
        if self.model is None:
            print('Model is not trained yet. Train model first.')
            return
        
        feature_importance = pd.DataFrame({'feature': self.features, 'importance': self.model.feature_importances_})
        feature_importance = feature_importance.sort_values('importance', ascending=False).reset_index(drop=True)
        if printOut: print(feature_importance)
        if returnDf: return feature_importance
    
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
        }

         # Khởi tạo mô hình
        xgbc = XGBRFClassifier(objective='multi:softmax', num_class=3, random_state=42)
        if showProgess:
            grid_search = GridSearchCV(xgbc, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
        else:
            grid_search = GridSearchCV(xgbc, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        self.n_estimators = best_params['n_estimators']
        self.max_depth = best_params['max_depth']
       
        print('n_estimators:', self.n_estimators)
        print('max_depth:', self.max_depth)
       
        self.model = grid_search.best_estimator_

        self._assessment()

        return self.model