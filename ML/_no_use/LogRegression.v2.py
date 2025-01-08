from ML._MakeSample import SampleXYtseries
from ML._ModelClassifier import ModelClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

class ModelLogRegression(ModelClassifier):
    def _note_(self):
        self._note = '''
        ** Test thấy không hiệu quả lắm
        Data đầu vào phải là object SampleXY. Object này đã được dựng trước đó, quy định các feature, label, thresoldDiff từ data gôc.
         Ex: SampleXY(data, features=['close','MA5','MA10'], label='close', pastCandle=14, foreCast=5, thresoldDiff=0.02)
        
        Chiến lược tự động với LogisticRegression. strategy='auto'
        Các chiến lược khác: strategy='ovr': mỗi class sẽ được dự đoán riêng biệt trong trường hợp phân loại đa lớp
        Thuật toán tối ưu hoá mặc định: solver='lbfgs'
            'lbfgs': "Limited-memory Broyden–Fletcher–Goldfarb–Shanno". Đây là một thuật toán tối ưu hóa được sử dụng để tìm các tham số tối ưu cho mô hình hồi quy logistic.
                    Hiệu quả cho các bộ dữ liệu nhỏ và trung bình. Xử lý tốt các bài toán đa lớp. Thường hội tụ nhanh hơn so với một số thuật toán khác.
            'sag' và 'saga': Hiệu quả cho các bộ dữ liệu rất lớn.
            'liblinear': Tốt cho các bài toán nhị phân và các bộ dữ liệu nhỏ.

        class_weight: mặc định None -> ưu tiên: class_weight='SMOTE'
        class_weight='balanced': áp dụng khi mẫu mất cân bằng. Một tuỳ chọn khi không dùng Data Augmentation
            hoặc dùng chỉ định tỷ lệ forcus cho từng lớp: class_weight = {-1: 2, 0: 1, 1: 2}
        class_weight='SMOTE': dùng SMOTE để tăng cường dữ liệu, chống mất cân bằng    
        
        max_iter=100 mặc định, co thể tăng số vòng lặp tối đa lên 1000. Sử dụng nếu có lỗi: ITERATIONS REACHED LIMIT
        verbose=1: hiển thị thông tin khi training - không cần thiết lắm
        
        '''
    def __init__(self, DataObject:SampleXYtseries, strategy='auto', solver='lbfgs', class_weight='SMOTE', max_iter=100, verbose=0, internalInfo=True) -> None:
        super().__init__(DataObject, class_weight=class_weight, verbose=verbose, internalInfo=internalInfo)
        
        self.strategy = strategy
        self.solver = solver
        self.max_iter = max_iter
               
    
    def trainModel(self):
        if self.X_train is None:
            print('Data is not prepared yet. Prepare data first.')
            return
        
        title = 'Logistic Regression'
        if self.internalInfo: print('Train model ' + title + '...')
             

        if self.strategy == 'ovr':
            self.model = OneVsRestClassifier(LogisticRegression(solver=self.solver, class_weight=self.class_weight, max_iter=self.max_iter,verbose=self.verbose, random_state=42,))
        else:
            self.model = LogisticRegression(solver=self.solver, class_weight=self.class_weight, max_iter=self.max_iter, verbose=self.verbose, random_state=42)
        
                
        self.model.fit(self.X_train, self.y_train)

        if self.internalInfo: 
            self._assessment()

    
    