from ML._MakeSample import SampleXY
from ML._ModelClassifier import ModelClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

class ModelSGDClassifier(ModelClassifier):
    def _note_(self):
        self._note = '''
        Data đầu vào phải là object SampleXY. Object này đã được dựng trước đó, quy định các feature, label, thresoldDiff từ data gôc.
         Ex: SampleXY(data, features=['close','MA5','MA10'], label='close', pastCandle=14, foreCast=5, thresoldDiff=0.02)
        
        loss: 'hinge' (default), 'log', 'modified_huber', 'squared_hinge', 'perceptron', or a regression loss: 'squared_loss', 'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive'
        penalty: 'l2' (default), 'l1', or 'elasticnet'
        alpha: L2 regularization term
        l1_ratio: The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1. l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1 penalty; defaults to 0.15.
        max_iter: The maximum number of passes over the training data (aka epochs). It only impacts the behavior in the fit method, and not the partial_fit method.
        '''
    def __init__(self, DataObject:SampleXY, loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, max_iter=1000, class_weight=None, labelEncode=False, verbose=0, internalInfo=True, clipFeatures:list=[], pca_components=None) -> None:
        super().__init__(DataObject, labelEncode=labelEncode, verbose=verbose, internalInfo=internalInfo, clipFeatures=clipFeatures, pca_components=pca_components)
        
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.class_weight = class_weight
    
    def trainModel(self):
        class_weight = self._get_class_weight('SGD Classifier')

        X_train, y_train = self._prepareData(useSMOTE=self.useSMOTE)
       
        if self.internalInfo: print('\tTraining model...')


        self.model = SGDClassifier(loss=self.loss, penalty=self.penalty, alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=self.max_iter, class_weight=class_weight, warm_start=True, random_state=42)
        
        self.model.fit(X_train, y_train)

        if self.internalInfo: 
            self._assessment()

    def trainModelGridSearch(self, showProgess=True):
        '''
         Chạy trainModel với tham số mặc định.
        Có thể chạy trainModelGridSearch trước để có best model, và bestparams, sau đó chạy trainModel nếu muốn
        training tiếp theo best params
        trainModelGridSearch thường mất thời gian
        '''

        # Khởi trạo mô hình
        X_train, y_train = self._prepareData()

        param_grid = {
            'loss': ['hinge'], # , 'log_loss', 'squared_error','log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'
            'penalty': ['l2', 'l1', 'elasticnet'],
            'alpha': [1e-4, 1e-3, 1e-2],
            'l1_ratio': [0.15, 0.25, 0.5, 0.75],
            'max_iter': [1000, 1500, 2000, 3000],
        }

        sgd = SGDClassifier(random_state=42)
        if showProgess:
            verbose=2
        else:
            verbose=0
        grid_search = GridSearchCV(estimator=sgd, param_grid=param_grid, n_jobs=-1, cv=5, verbose=verbose)
        
        grid_search.fit(X_train, y_train)

        self.bestParas = grid_search.best_params_

        self.loss = self.bestParas['loss']
        self.penalty = self.bestParas['penalty']
        self.alpha = self.bestParas['alpha']
        self.l1_ratio = self.bestParas['l1_ratio']
        self.max_iter = self.bestParas['max_iter']

        print(f'Best parameters: {self.bestParas}')

        self.model = grid_search.best_estimator_

        self._assessment()

        return self.model
    
        