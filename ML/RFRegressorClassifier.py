from ML._MakeSample import SampleXYtseries
from ML._ModelClassifier import ModelClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
import pandas as pd
from sklearn.model_selection import GridSearchCV

class ModelRFRegressorClassifier(ModelClassifier):
    def _note_(self):
        self._note = '''
        Đã test với dữ liệu BNB/USDT sport khung 4h
        Nhận thấy: RSI có tác dụng tốt
                   EMA không có tác dụng
        Ổn định: 'close','MA50','MA200','RSI','ATR','ADX'
        md = ModelRFRegressorClassifier(sb, 
        n_estimators=300, max_depth = None, min_samples_split = 2, min_samples_leaf = 1, max_features = 'log2')

        '''

    def __init__(self, DataObject:SampleXYtseries, n_estimators=100, max_depth=None, max_features='sqrt', min_samples_split=2, min_samples_leaf=1, class_weight='balanced', verbose=0, internalInfo=True) -> None:      
        super().__init__(DataObject, class_weight=class_weight, verbose=verbose, internalInfo=internalInfo)

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf


    def trainModel(self):
        if self.X_train is None:
            print('Data is not prepared yet. Prepare data first.')
            return

        title = 'Random Forest R Classifier'

        if self.internalInfo: print('Train model ' + title + '...')


        self.model = BalancedRandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, max_features=self.max_features, min_samples_split = self.min_samples_split, min_samples_leaf=self.min_samples_leaf,
                                                    class_weight=self.class_weight, 
                                                    sampling_strategy='all', replacement=True, bootstrap=False, random_state=42)
        
        self.model.fit(self.X_train, self.y_train)

        if self.internalInfo:  print('\tAssessing model...')
        
        self._assessment()

        if self.internalInfo: self.printOutAssessment(ispredict=False)
    
    def TrainHardModel(self):
        if self.X_train_hard is None:
            print('Data is not HARD prepared yet. Prepare HARD data first.')
            return

        title = 'Random Forest R Classifier'
        if self.internalInfo: print('Train model ' + title + ' with hard data...')
        
        self.model = BalancedRandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, max_features=self.max_features, min_samples_split = self.min_samples_split, min_samples_leaf=self.min_samples_leaf,
                                                        class_weight=self.class_weight, 
                                                        sampling_strategy='all', replacement=True, bootstrap=False, random_state=42)
        
        self.model.fit(self.X_train_hard, self.y_train_hard)
    
    def TestTrainAndPredictHardModel(self, cuttail=0): # fortest
        import numpy as np
        import random
        from tqdm import tqdm
        if self.X_train_hard is None:
            print('Data is not HARD prepared yet. Prepare HARD data first.')
            return
        
        title = 'Random Forest R Classifier'
        if self.internalInfo: print('Train model ' + title + ' with hard data...')

        
        self.y_pred_hard = np.array([], dtype=int)
        prevpredict = None
        for i in tqdm(range(len(self.y_test_hard))):
            # if i > 0:
            #     random_index = random.randint(0, len(self.X_train_hard))
            #     rowx = self.X_test_hard.iloc[[i-1]]
            #     self.X_train_hard = pd.concat([self.X_train_hard.iloc[:random_index], rowx, self.X_train_hard.iloc[random_index:]], ignore_index=True)
            #     rowy = self.y_test_hard[i-1]
            #     self.y_train_hard = np.insert(self.y_train_hard, random_index, rowy)

        
            if i > 0:  
                self.X_train_hard = pd.concat([self.X_train_hard, self.X_test_hard.iloc[[i-1]]], ignore_index=True)
                self.y_train_hard = np.append(self.y_train_hard, prevpredict)
                if i > cuttail:
                    print(f'\n Try predict sample {self.y_train_hard[0-cuttail-1]} - {self.y_test_hard[i-cuttail-1]}')
                    self.y_train_hard[0-cuttail-1] = self.y_test_hard[i-cuttail-1]
           

            self.model = BalancedRandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, max_features=self.max_features, min_samples_split = self.min_samples_split, min_samples_leaf=self.min_samples_leaf,
                                                        class_weight=self.class_weight, 
                                                        sampling_strategy='all', replacement=True, bootstrap=False, random_state=42)
            
            self.model.fit(self.X_train_hard, self.y_train_hard)

            predict =  (self.model.predict(self.X_test_hard.iloc[[i]]))
            prevpredict = predict[0]

            self.y_pred_hard = np.append(self.y_pred_hard, predict)

            print(f"\n len:{len(self.X_train_hard)} TS:{self.tss.iloc[i]} {'True' if predict==self.y_test_hard[i] else 'False'} - (Predict:{predict[0]}/Real:{self.y_test_hard[i]})")

        self._hard_assessment()

        if self.internalInfo: self.printOutAssessment(ispredict=False)
    

    def printOutFeatureImportance(self, returnDf=False, printOut=True):
        if self.model is None:
            print('Model is not trained yet. Train model first.')
            return
        
        feature_importance = pd.DataFrame({'feature': self.model.feature_names_in_, 'importance': self.model.feature_importances_})
        feature_importance = feature_importance.sort_values('importance', ascending=False).reset_index(drop=True)
        if printOut: print(feature_importance)
        if returnDf: return feature_importance

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
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['log2', 'sqrt']
        }

         # Khởi tạo mô hình
        rf = BalancedRandomForestClassifier(class_weight=self.class_weight, sampling_strategy='all', replacement=True, bootstrap=False, random_state=42)
        # sampling_strateg: 'all': Tất cả các lớp sẽ được cân bằng sao cho mỗi lớp có số lượng mẫu giống nhau.
        #                   'minority': Chỉ lớp thiểu số sẽ được tăng cường số lượng mẫu cho bằng với lớp đa số
        #                   'not minority': Tất cả các lớp ngoại trừ lớp thiểu số sẽ được cân bằng.
        #                   'auto': Cân bằng tất cả các lớp sao cho mỗi lớp có số lượng mẫu tương đương với lớp có số lượng mẫu lớn nhất.
        # replacement: Quyết định xem các mẫu có được lấy mẫu lại (sampling) với phép thay thế hay không.
        #              True: Lấy mẫu lại với phép thay thế (có thể chọn lại một mẫu đã được chọn trước đó). số lượng mẫu trong các lớp sau khi cân bằng có thể lớn hơn số lượng mẫu ban đầu của mỗi lớp, vì mẫu có thể được chọn nhiều lần.
        #              False: Lấy mẫu lại không dùng phép thay thế (mỗi mẫu chỉ được chọn một lần).
        if showProgess:
            verbose=2
        else:
            verbose=0
        
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=verbose)
        
        grid_search.fit(self.X_train, self.y_train)

        self.bestParas = grid_search.best_params_

        self.n_estimators = self.bestParas['n_estimators']
        self.max_depth = self.bestParas['max_depth']
        self.min_samples_split = self.bestParas['min_samples_split']
        self.min_samples_leaf = self.bestParas['min_samples_leaf']
        self.max_features = self.bestParas['max_features']
        
        print(f'Best parameters: {self.bestParas}')
    
        self.model = grid_search.best_estimator_

        self._assessment()

        self.printOutAssessment(ispredict=False)

        return self.model
    
    def make_up_df_assessment(self) -> pd.DataFrame: 
        n = self.classification_report.split()
        p = self.confusion_matrix
        l = self.printOutFeatureImportance(returnDf=True, printOut=False)
        if len(l) < 5:
            # thêm các dòng trống vào dataframe l
            for _ in range(5-len(l)):
                new_row = pd.DataFrame({'feature': [''], 'importance': ['']})
                l = pd.concat([l, new_row], ignore_index=True)

        data = {
            'label': [self.model.classes_[0], self.model.classes_[1],'accuracy','macro avg','weighted avg','Important features'],
            'precision': [n[5], n[10], '', n[20], n[25], l.iloc[0]['feature']],
            'recall': [n[6], n[11], '', n[21], n[26], l.iloc[1]['feature']],
            'f1-score': [n[7], n[12], n[15], n[19], n[25], l.iloc[2]['feature']],
            'support': [n[8], n[13], n[16], n[22], n[28], l.iloc[3]['feature']],
            f'Predict {self.model.classes_[0]}': [p[0,0], p[1,0], '', '', '', l.iloc[4]['feature']],
            f'Predict {self.model.classes_[1]}': [p[0,1], p[1,1], '', '', '', l.iloc[5]['feature']],
            
        }
        df = pd.DataFrame(data)
        return df
    
    def make_up_df_assessment_simple(self) -> pd.DataFrame: 
        n = self.classification_report.split()
        # p = self.confusion_matrix
        l = self.printOutFeatureImportance(returnDf=True, printOut=False)
        l = l['feature'].to_list()
        # if len(l) < 11:
        #     # thêm các dòng trống vào dataframe l
        #     for _ in range(11-len(l)):
        #         new_row = pd.DataFrame({'feature': [''], 'importance': ['']})
        #         l = pd.concat([l, new_row], ignore_index=True)

        data = {
            f'Precision {self.model.classes_[0]}': [n[5]],
            f'Recall {self.model.classes_[0]}': [n[6]],
            f'F1-score {self.model.classes_[0]}': [n[7]],
            f'Precision {self.model.classes_[1]}': [n[10]],
            f'Recall {self.model.classes_[1]}': [n[11]],
            f'F1-score {self.model.classes_[1]}': [n[12]],
            'Accuracy': [n[15]],
            'Feature': [', '.join(l)], 
        }
        df = pd.DataFrame(data)
        df[f'Precision {self.model.classes_[0]}'] = pd.to_numeric(df[f'Precision {self.model.classes_[0]}'], errors='coerce')
        df[f'Recall {self.model.classes_[0]}'] = pd.to_numeric(df[f'Recall {self.model.classes_[0]}'], errors='coerce')
        df[f'F1-score {self.model.classes_[0]}'] = pd.to_numeric(df[f'F1-score {self.model.classes_[0]}'], errors='coerce')
        df[f'Precision {self.model.classes_[1]}'] = pd.to_numeric(df[f'Precision {self.model.classes_[1]}'], errors='coerce')
        df[f'Recall {self.model.classes_[1]}'] = pd.to_numeric(df[f'Recall {self.model.classes_[1]}'], errors='coerce')
        df[f'F1-score {self.model.classes_[1]}'] = pd.to_numeric(df[f'F1-score {self.model.classes_[1]}'], errors='coerce')
        df['Accuracy'] = pd.to_numeric(df['Accuracy'], errors='coerce')
        return df




