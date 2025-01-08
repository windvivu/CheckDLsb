from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from ML._MakeSample import SampleXYtseries
import pandas as pd

class ModelClassifier:
    def _note_(self):
        ''' Need overidding'''
        self._note = '''
        Lớp trừu tượng, cần được kế thừa và rewrite một số phương thức
        '''
    def __str__(self) -> str:
        return self._note
    
    def __init__(self, DataObject:SampleXYtseries, class_weight=None, verbose=0, internalInfo=True) -> None:
        ''' Need overidding
        \n Data đầu vào phải là object SampleXYtseries.
        \n verbose > 0 nếu muốn hiển thị thông tin khi training. Hãy cứ để 0
        \n internalInfo=True nếu muốn hiển thị thông tin quá trình training và validation
        \n Quy trình traning: init ModelClassifier -> obj.ProcessDataPCAtraining -> trainModel
        \n                    hoặc sau khi obj.ProcessDataPCAtraining -> trainModelGridSearch nếu cần                                   
        '''
        self._note_()
        self.DataObject = DataObject        
        self.class_weight = class_weight
        if self.class_weight=='SMOTE':
            self.class_weight = None
            self.__useSMOTE = True
        else:
            self.__useSMOTE = False
        self.verbose = verbose
        self.internalInfo = internalInfo

        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None

        self.X_train_hard = None
        self.y_train_hard = None
        self.X_test_hard = None
        self.y_test_hard = None
        self.y_pred_hard = None

        self.features = None
        self.bestParas = None

        self.confusion_matrix = None
        self.classification_report = None
        self.confusion_matrix_predict = None
        self.classification_report_predict = None


        if DataObject.isDatapredict:
            raise ValueError('This SampleXYtseries is for predict only. Cannot use it for training model purpose.')
    
       
        
    def _split_hard_test(self, X, y, test_size, shuffle):
        '''
        test_size: int or float

        nêu test_size là int thì là số lượng mẫu cho tập test
        
        nếu test_size là float thì là tỷ lệ số lượng mẫu cho tập test
        '''
        import numpy as np
        # bộ test cứng ở cuối của x, y. Bộ traning thì xáo trộn random nếu cần
        if isinstance(test_size, int):
            num_test = test_size
        else:
            num_test = int(len(X) * test_size)  # Số lượng mẫu cho tập test
        num_train = len(X) - num_test       # Số lượng mẫu cho tập train

        # Tách dữ liệu
        X_train = X.iloc[:num_train]
        X_test = X.iloc[num_train:]
        y_train = y[:num_train]
        y_test = y[num_train:]

        if shuffle:
            # Xáo trộn dữ liệu train
            indices = np.arange(len(X_train))
            np.random.seed(42)  # Đảm bảo tái lập kết quả
            np.random.shuffle(indices)

            X_train = X_train.iloc[indices]
            y_train = y_train[indices]

     
        return X_train, X_test, y_train, y_test

    
    def ProcessDataPCAsplit(self, split=0, test_size=0.3, shuffle=True, cuttail=0):
        '''
        split = 0: chia train/test bằng hàm train_test_split của sklearn
        
        split = 1: chia cứng phần đầu là train, phần sau là test theo tỷ lệ, không chọn ngẫu nhiên. Tuỳ chọn xáo trộn phần đầu
        
        cuttail sẽ cắt bớt số lượng mẫu cuối của tập train, giả lập trong thực tế dữ liệu đuôi
        bị mất do chưa có đủ forcast tương lai
        '''
        if self.internalInfo: print('Prepare data for training/testing...')
        X, y = self.DataObject.PROCESSPCA(realDT=False)

        self.features = X.columns
        if split == 0:
            if test_size == 0: test_size = None
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=shuffle)
        else:
            X_train, X_test, y_train, y_test = self._split_hard_test(X, y, test_size=test_size, shuffle=shuffle)
        
        X_train = X_train.iloc[0:len(X_train) - cuttail]
        y_train = y_train[0:len(y_train) - cuttail]

        if self.__useSMOTE:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        
        self.X_train = X_train
        self.y_train = y_train
        # lưu để test sau này
        self.X_test = X_test
        self.y_test = y_test
    
    def TestProcessDataPCAhardsplit(self, test_size=0.3, suffle=True): # fortest
        if self.internalInfo: print('Prepare data for hard training...')
        X, y = self.DataObject.PROCESSPCA(realDT=False)
        X_train_hard, X_test_hard, y_train_hard, y_test_hard = self._split_hard_test(X, y, test_size=test_size, shuffle=suffle)

        slideTss = len(self.DataObject.tss) - len(y_test_hard)
        self.tss = self.DataObject.tss.iloc[slideTss:]

        self.X_train_hard = X_train_hard
        self.y_train_hard = y_train_hard
        # lưu để test sau này
        self.X_test_hard = X_test_hard
        self.y_test_hard = y_test_hard
    
    def ProcessDataPCAhardsplit(self, test_size=0, suffle=True):
        if self.internalInfo: print('Prepare data for hard training...')
        X, y = self.DataObject.PROCESSPCA(realDT=False)
        X_train_hard, X_test_hard, y_train_hard, y_test_hard = self._split_hard_test(X, y, test_size=test_size, shuffle=suffle)

        self.X_train_hard = X_train_hard
        self.y_train_hard = y_train_hard
        self.X_test_hard = None
        self.y_test_hard = None

    def PCAtrainmore(self, newdata:pd.DataFrame):
        if self.DataObject.makeMoreSampleXY(newdata) is None:
            return
        if self.internalInfo: print('\tTraining more data...')
        X, y = self.DataObject.transformSampleXYpca()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        if self.__useSMOTE:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        
        self.X_train = X_train
        self.y_train = y_train
        # lưu để test sau này
        self.X_test = X_test
        self.y_test = y_test

        self.trainModel()
    
    def ProcessPredict(self, data:pd.DataFrame, realDT=True, copyScalerIfPredict=True):
        if self.model is None:
            print('Model is not trained yet. Train model first.')
            return
        
        splpred = SampleXYtseries(data, typeMake=self.DataObject.typeMake , features=self.DataObject.features, typeScale=self.DataObject.typeScale,
                          pastCandle=self.DataObject.pastCandle, foreCast=self.DataObject.foreCast, thresoldDiff=self.DataObject.thresoldDiff)
        
        splpred.makeSampleXY(realDT=realDT)
        splpred.copyScaler(self.DataObject)
        Xpred, rverify = splpred.transformSampleXYpca(copyScalerIfPredict=copyScalerIfPredict) # ở đây realDT=True nế rverify được trả về None

        return self.predict(Xpred, rverify)
    
    
    def trainModel(self):
        ''' Need overidding'''

        if self.X_train is None:
            print('Data is not prepared yet. Prepare data first.')
            return
        
        title = 'Need overidding'
        if self.internalInfo: print('Train model ' + title + '...')
        
        if self.internalInfo: print('\tTraining model...')

        # .............

        if self.internalInfo:  print('\tAssessing model...')
        
        self._assessment()

        if self.internalInfo: self.printOutAssessment(ispredict=False)
    
    def TestTrainAndPredictHardModel(self): # fortest
        ''' Need overidding'''

        if self.X_train_hard is None:
            print('Data is not HARD prepared yet. Prepare HARD data first.')
            return

        title = 'Need overidding'
        if self.internalInfo: print('Train model ' + title + '...')
        
        # ...
           
    def _assessment(self):
        if self.model is None:
            print('Model is not trained yet. Train model first.')
            return
        
        y_pred = self.model.predict(self.X_test)
        self.y_pred = y_pred # lưu lại để test sau này
        self.confusion_matrix = confusion_matrix(self.y_test, y_pred)
        self.classification_report = classification_report(self.y_test, y_pred, zero_division=0) # zero_division=0: tránh chia cho mẫu = 0

    def _hard_assessment(self):    
        if self.model is None:
            print('Model is not trained yet. Train model first.')
            return
        
        self.confusion_matrix = confusion_matrix(self.y_test_hard, self.y_pred_hard)
        self.classification_report = classification_report(self.y_test_hard, self.y_pred_hard, zero_division=0) # zero_division=0: tránh chia cho mẫu = 0

    def hard_to_excel(self, filename):
        if self.y_pred_hard is None:
            print('No prediction value yet!')
            return

        df = self.X_test_hard.copy()
        df['y_true'] = self.y_test_hard
        df['y_pred'] = self.y_pred_hard

        df.to_excel(filename, index=False)
    
    def printOutAssessment(self, ispredict:bool):
        if ispredict:
            _confusion_matrix = self.confusion_matrix_predict
            _classification_report = self.classification_report_predict
        else:
            _confusion_matrix = self.confusion_matrix
            _classification_report = self.classification_report

        if _confusion_matrix is None or _classification_report is None:
            print('No assessment value yet!')
            return

        print("Confusion Matrix:")
        print(_confusion_matrix)
        print("\nClassification Report:")
        print(_classification_report)
    
    def cross_validation(self, cv=5, scoring=None):
        '''
        Scoring mặc định accuracy cho bài toán nhị phân, R² score cho hồi quy định lượng, hoặc có thể chọn:
        '''
        if self.model is None:
            print('Model is not trained yet. Train model first.')
            return
        
        print('Prepare data for cross validation...')
        X, y = self.DataObject.PROCESSPCA(realDT=False)
        
        print('Cross validation...')
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
        print(f"    Cross-validation scores: {cv_scores}")
        print(f"    Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")


    def predict(self, Xnew, Ytruenew=None):
        if self.model is None:
            print('Model is not trained yet. Train model first.')
            return
        
        self.ypredict = self.model.predict(Xnew)

        if Ytruenew is not None:
            self.confusion_matrix_predict = confusion_matrix(Ytruenew, self.ypredict)
            self.classification_report_predict = classification_report(Ytruenew, self.ypredict, zero_division=0)
            if self.internalInfo: self.printOutAssessment(ispredict=True)
        
        return self.ypredict