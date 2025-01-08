import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from collections import Counter
import plotly.graph_objects as go
import plotly.subplots as sp
import matplotlib.pyplot as plt
import seaborn as sbn
from tqdm import tqdm

class SampleXYtseries:
    def __init__(self, data:pd.DataFrame, typeMake:int, features:list, typeScale=str, label:str='close', pastCandle=7, foreCast=1, thresoldDiff=0.01, testNoSelfScale=False) -> None:
        '''
        Tạo dữ liệu để training cho mô hình.
        \ntypeMake=3: 3 lớp, với 2 tăng, 1 giảm, 0 không đổi
        \ntypeMake=2: 2 lớp, với vị thế tăng là dương tính
        \ntypeMake=-2: 2 lớp, với vị thế giảm là dương tính
        \ntypeMake= other : phân 3 lớp
        \ntypeScale: 'max', 'minmax', 'none'. Với max: self-scale theo max,minmax (dùng cho ML); với minmax: self-scale theo thang minmax scaler (dùng cho DL); 
        \n            none: không self-scale (chỉ dùng cho pca hoặc diff (ML)) -> đã thử nghiệm thấy không ổ định
        \nVới max scaler sẽ phần nào giữ nguyên được sự biến thiên giữa các field từng row. Với minmax scaler sẽ chỉ coi row là một thực thể, không quan tâm đến biến thiên giữa cá field từng row.
        \nVậy dùng minmax cho DL và dùng max cho ML -> đã thử nghiệm thấy minmax cũng dùng được cho ML (Random Rorest)
        \nlabel mặc định là 'close'
        \n
        \nfeaturesOtherscale: list các field dao động ở thang khác, cần self-scale riêng
        \n ** testNoSelfScale = True chỉ để test xem dữ liệu có khác nhau giữa có self-scale và không self-scale. Thực tế sẽ luôn để = False

        \n Quy trình: init SampleXYtseries -> obj.makeSampleXY -> <get X_flatten, y> -> transformSampleXY2matrix -> <get X_matrix, y>
                                                                                     -> transformSampleXYpca -> <get X df/np, y>
        \nCó thể dùng 1 hàm duy nhất PROCESSMATRIX hoặc PROCESSPCA để thực hiện toàn bộ quy trình
        \nMục tiêu của class này để tạo ra bộ X và y, không xác định cho train hay test mẫu. Biến realDT chỉ là để muốn có y hay không.
        Vì dữ liệu thực tế để dự đoán, chưa tồn tại y. Khi đó y ở self.__y chỉ là kết quả ffill tránh bị dropna mất dữ liệu đuôi
        '''
        self.__data = data
        self.__typeMake = typeMake
        self.__features = features
        self.__featuresOtherscale = []
        self.__featuresOtherPlot = []
        self.__label = label
        self.__pastCandle = pastCandle
        if self.__pastCandle < 0: self.__pastCandle=0
        self.__foreCast = foreCast
        if self.__foreCast < 1: self.__foreCast=1
        self.__thresoldDiff = thresoldDiff
        self.__testNoSelfScale = testNoSelfScale
        self.__typeScale = typeScale
        if self.__typeScale not in ['max', 'minmax', 'none']:
            raise ValueError('Invalide typeScale')

        self.__setFeaturesOtherscale()

        self.__temp = None
        self._pca = {}
        self._scaler = None
        self.__X = None
        self.__y = None
        self.__isDatapredict = False
    
    def copyScaler(self, DataObject):
        self._pca = DataObject._pca
        self._scaler = DataObject._scaler
        self.__isDatapredict = True

    def PROCESSMATRIX(self, realDT:bool, showProgress=True, square=False):
        if self.__testNoSelfScale:
            raise ValueError('testNoSelfScale=True, không thể sử dụng PROCESSMATRIX')
        
        self.makeSampleXY(realDT)
        showProgress = not realDT # nếu dữ liệu thật để predict thì không hiện progress
        return self.transformSampleXY2matrix(showProgress=showProgress, square=square)
    
    def PROCESSPCA(self, realDT:bool, to='df', plotCorrelation = False):
        '''
        Lưu ý: plotCorrelation=True sẽ chỉ vẽ chart correlation chứ không trả về X, y
        '''
        if self.__testNoSelfScale:
            raise ValueError('testNoSelfScale=True, không thể sử dụng PROCESSPCA')
        
        self.makeSampleXY(realDT)
        return self.transformSampleXYpca(to=to, plotCorrelation=plotCorrelation)
    
    def __PROCESSDIFF(self, realDT:bool, stepDiff:int,  diffPercent:bool, plotCorrelation = False):
        '''
        Lưu ý: plotCorrelation=True sẽ chỉ vẽ chart correlation chứ không trả về X, y
        \ndiffPercent: True: tính difference theo tỷ lệ; False: tính theo chênh lệch
        \nTạm không dùng vì không thấy hiệu quả
        '''
        if self.__testNoSelfScale:
            raise ValueError('testNoSelfScale=True, không thể sử dụng PROCESSDIFF')
        
        self.makeSampleXY(realDT)
        return self.transformSampleXYdiff(stepDiff, diffPercent=diffPercent, plotCorrelation=plotCorrelation)
    
     
    def get1sampleX(self, index, square=False):
        '''
        Trả về một mẫu sample X tại index ở dạng matrix. Mỗi hàng là một feature, giá trị là các biến thiên theo thứ tự thời gian
        \nsquare = True thì sẽ điều chỉnh thành ma trận vuông
        '''
        
        if self.__temp is None:
            print('Must run makeSampleXYwTimeseries() first')
            return None
        
        if index < 0 or index >= len(self.__temp):
            print('Index out of range')
            return None

        onerow = self.__temp.iloc[index:index+1]
        rows = []

        fieldScale = list(set(self.__features) - set(self.__featuresOtherscale) - set(self.__featuresOtherPlot))

        for f in fieldScale:
            itm = []
            for i in range(0-self.__pastCandle, 1):
                itm.append(onerow[f+'_'+str(i)].values[0])
            rows.append(itm)
        
        for f in self.__featuresOtherscale:
            itm = []
            for i in range(0-self.__pastCandle, 1):
                itm.append(onerow[f+'_'+str(i)].values[0])
            rows.append(itm)
        
        for f in self.__featuresOtherPlot:
            itm = []
            for i in range(0-self.__pastCandle, 1):
                itm.append(onerow[f+'_'+str(i)].values[0])
            rows.append(itm)
        
        rows = np.array(rows)

        if square:
            rows = self.__make_square_matrix(rows)
        return rows
    
    def transformSampleXYpca(self, to='df', plotCorrelation = False, copyScalerIfPredict=True):
        '''
        Trả về X và y với X ở đã giảm chiều dữ liệu những field series f_i_t, f_i_t-1, f_i_t-2, ..., f_i_t-m gom lại thành một
        \n f_i_t tức là các cột IND_-1, IND_-2, ..., IND_-n giảm chiều thành một cột duy nhất là f_i_t. Số m cột f_i_t tương ứng số features
        \nto: 'df' hoặc 'np', tức là X sẽ ở dạng dataframe hay numpy
        \nLưu ý: plotCorrelation=True sẽ chỉ vẽ chart correlation chứ không trả về X, y
        '''
        if to != 'df': to = 'np'
        
        if self.__temp is None:
            print('Must run makeSampleXY() first')
            return None
        
        
        if self.__typeScale == 'none': # do chưa self-scale gì cả nên cần chuẩn hoá trước khi pca
            if self.__isDatapredict == False:
                self._scaler = StandardScaler()
                # self._scaler = MinMaxScaler() -> tính đổi qua nhưng không có gì khác biệt

                self.__temp[self._featuresWseries] = self._scaler.fit_transform(self.__temp[self._featuresWseries])
            else:
                if self._scaler is None: raise ValueError('Must copyScaler() from SampleXYtseries object which has been trained')
                if copyScalerIfPredict:
                    self.__temp[self._featuresWseries] = self._scaler.transform(self.__temp[self._featuresWseries]) 
                else:
                    self._scaler = StandardScaler()
                    self.__temp[self._featuresWseries] = self._scaler.fit_transform(self.__temp[self._featuresWseries])
        
        if self.__isDatapredict:
            if len(self._pca) == 0: raise ValueError('Must copyScaler() from SampleXYtseries object which has been trained')

        X = pd.DataFrame()
        for f in self.__features:
            fe = []
            for i in range(self.__pastCandle + 1):
                
                fe.append(f + '_' + str(i-self.__pastCandle))
            
            if self.__isDatapredict:
                if copyScalerIfPredict:
                    _x = self._pca[f].transform(self.__temp[fe])
                else:
                    self._pca[f] = PCA(n_components=1)
                    _x = self._pca[f].fit_transform(self.__temp[fe])
            else:
                self._pca[f] = PCA(n_components=1)
                _x = self._pca[f].fit_transform(self.__temp[fe])
            
            X[f] = _x.flatten()

        if plotCorrelation:
            X['label'] = self.__y
            self.plotCorrelation(X)
            return None

       
        if to == 'np': X = np.array(X)
        if self.__realDT:
            y_ = None
        else:
            y_ = self.__y.values
        return X, y_
    
    def transformSampleXYdiff(self, stepDiff:int, diffPercent:bool, plotCorrelation = False):
        '''
        Trả về X, y với X là biến thiên giá trị theo stepDiff dựa trên field series f_i_t, f_i_t-1, f_i_t-2, ..., f_i_t-n
        \nLuôn trả về numpy array do được StandardScaler
        \nLưu ý: plotCorrelation=True sẽ chỉ vẽ chart correlation chứ không trả về X, y
        '''
        X = pd.DataFrame()
        if stepDiff < 1:
            raise ValueError('stepDiff must be greater than 0')
        if stepDiff > self.__pastCandle:
            raise ValueError('stepDiff must be less than pastCandle')
        
        if self.__isDatapredict and self._scaler is None:
            raise ValueError('Must copyScaler() from SampleXYtseries object which has been trained')
        
        for f in self.__features:
            currentf = self.__temp[f + '_0']
            passf = self.__temp[f + '_-' + str(stepDiff)]
            if diffPercent:
                diff = (currentf - passf)/currentf
            else:
                diff = (currentf - passf)
            X[f] = diff
        
        if plotCorrelation:
            X['label'] = self.__y
            self.plotCorrelation(X)
            return None
        
        if self.__isDatapredict:
            # X = self._scaler.transform(X) # Chố này không dùng nữa, do thấy PCA mới có hiệu quả hơn lấy PCA của tập dữ liệu cũ
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)
        else:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)

        if self.__realDT:
            y_ = None
        else:
            y_ = self.__y.values
        return X, y_
            
    
    def transformSampleXY2matrix(self, showProgress=True, square=False):
        '''
        Trả về một X và y với X ở dạng matrix. Mỗi hàng là một feature, giá trị là các biến thiên theo thứ tự thời gian.
        Phục vụ cho deeplearning
        \nshowProgress: hiển thị thanh tiến trình
        \nsquare: nếu True, sẽ điều chỉnh thành ma trận vuông
        '''       
        if self.__temp is None:
            print('Must run makeSampleXY() first')
            return None
        
        X = []
        if showProgress:
            iter = tqdm(range(len(self.__temp)))
        else:
            iter = range(len(self.__temp))

        for i in iter:
            X.append(self.get1sampleX(i, square=square))
        X = np.array(X)
        if self.__realDT:
            y_ = None
        else:
            y_ = self.__y.values
        return X, y_
    
    def plotlySampleX(self, index):
        '''
        Visualize 1 mẫu sampleX timeseries bằng thư viện plotly
        '''
        
        sample = self.get1sampleX(index)
        if sample is None: return

        fieldScale = list(set(self.__features) - set(self.__featuresOtherscale) - set(self.__featuresOtherPlot))

        numrow = 1
        if len(self.__featuresOtherscale)>0: numrow += 1
        if len(self.__featuresOtherPlot)>0: numrow += 1
        if numrow == 2: row_heights=[0.8, 0.2]
        if numrow == 3: row_heights=[0.6, 0.2, 0.2]
                
        if numrow > 1:
            fig = sp.make_subplots(rows=numrow, cols=1, shared_xaxes=True, row_heights=row_heights, vertical_spacing=0.05)
        else:
            fig = sp.make_subplots(rows=1, cols=1, shared_xaxes=True)


        for i in range(len(fieldScale)):
            fig.add_trace(go.Scatter(x=[i for i in range(sample.shape[1])], y=sample[i], mode='lines', name=fieldScale[i]),row=1, col=1)
        
        since = len(fieldScale)
        end = len(self.__features) - len(self.__featuresOtherPlot)
        numplot = 1
        if len(self.__featuresOtherscale)>0:
            numplot += 1
            j = 0
            for i in range(since, end):
                fig.add_trace(go.Scatter(x=[i for i in range(sample.shape[1])], y=sample[i], mode='lines', name=self.__featuresOtherscale[j]),row=numplot, col=1)
                j += 1
        
        if len(self.__featuresOtherPlot)>0:
            numplot += 1
            j = 0
            for i in range(end, len(self.__features)):
                fig.add_trace(go.Scatter(x=[i for i in range(sample.shape[1])], y=sample[i], mode='lines', name=self.__featuresOtherPlot[j]),row=numplot, col=1)
                j += 1

        title = 'Sample' + str(index)
        if self.typeMake == 2:
            if self.__y[index] == 1: title += ': Up'
            if self.__y[index] == 0: title += ': None'
        elif self.typeMake == -2:
            if self.__y[index] == 1: title += ': Down'
            if self.__y[index] == 0: title += ': None'
        else:
            if self.__y[index] == 2: title += ': Up'
            if self.__y[index] == 1: title += ': Down'
            if self.__y[index] == 0: title += ': None'
        
        fig.update_layout(title_text=title)
        fig.show()
    
    def makeSampleXY(self, realDT:bool):
        '''
        Trả về data đã sãn sàng cho training hoặc predict, mỗi row là một sample timeseries ở dạng flatten.
        \nBiến thiên giá trị đã được đưa vào thành các feature F_i_t, F_i_t-1, F_i_t-2, ..., F_i_t-n
        \nTuy nhiên việc này có vẻ thừa đối với ML (gây đa cộng tuyến), chỉ có tác dụng với Deeplearning (DL sẽ nhìn thấy các biến thiên dữ liệu)
        \n
        \nLuu ý là trả về tuple: data và y. y có thể là None nếu realDT=True
        \nrealDT: khi tạo sample từ dữ liệu thật, mới tinh. Lúc đó chưa có y, y sẽ là None
        '''
        self.__realDT = realDT
        if self.__typeMake == 3:
            return self.__makeSampleXYwts(numClass=3, realDT = realDT)
        elif self.typeMake == 2:
            return self.__makeSampleXYwts(numClass=2, realDT = realDT)
        elif self.typeMake == -2:
            return self.__makeSampleXYwts(numClass=-2, realDT = realDT)
        else:
            return self.__makeSampleXYwts(numClass=3, realDT = realDT)

    def __makeSampleXYwts(self, numClass:int, realDT:bool):
        '''
        realDT: khi sample từ dữ liệu thật, mới, chưa có y
        '''
        self.__temp = pd.DataFrame()
        self._featuresWseries = []
        self._featuresOtherscaleWseries = []
        self._featuresOtherPlotWseries = []
        
        for f in self.__features:
            for i in range(self.__pastCandle + 1):
                self.__temp[f + '_' + str(i-self.__pastCandle)] = self.__data[f].shift(self.__pastCandle-i)
                self.__temp = self.__temp.copy()  # Tạo bản sao để giảm phân mảnh
                self._featuresWseries.append(f + '_' + str(i-self.__pastCandle))
                if f in self.__featuresOtherscale:
                    self._featuresOtherscaleWseries.append(f + '_' + str(i-self.__pastCandle))
                if f in self.__featuresOtherPlot:
                    self._featuresOtherPlotWseries.append(f + '_' + str(i-self.__pastCandle))
        
        # label tương ứng ngày hiện tại -> nó không di chuyển gì cả
        _label_0 = self.__data[self.__label]
          
        # label ngày kế
        _label_x = self.__data[self.__label].shift(0 - self.__foreCast)

        # tính lệch giá trị
        _label_diff = (_label_x - _label_0)/_label_0
        
        self.__temp[self.__label+'_00'] = _label_0 
        self.__temp[self.__label+'_'+str(self.__foreCast)] = _label_x
        self.__temp['diff_'+self.__label] = _label_diff

        self.__temp['timestamp'] = self.__data['timestamp']

        if numClass == 2:
            self.__temp['label'] = np.where( (_label_diff > 0) & (abs(_label_diff) > self.__thresoldDiff), 1, 0) 
        elif numClass ==-2:
            self.__temp['label'] = np.where((_label_diff < 0) & (abs(_label_diff) > self.__thresoldDiff), 1, 0) 
        else:
            self.__temp['label'] = np.where( (_label_diff > 0) & (abs(_label_diff) > self.__thresoldDiff), 2,
                                np.where((_label_diff < 0) & (abs(_label_diff) > self.__thresoldDiff), 1, 0) )

        if realDT: # nếu trong trường hợp dự đoán thật, dùng forward fill để điền vào các ô thiếu dữ liệu do tương lai chưa sảy ra (diff = close0 - close(future))
            # khi đó, dropNan sẽ không làm mất những dòng sau cùng, cũng là dòng cần dự đoán.
            self.__temp.ffill(inplace=True)
            
        self.__temp = self.__temp.dropna() # làm mất dữ liệu đuôi và đầu nếu khong có ffill
        self.__temp.reset_index(drop=True, inplace=True)

        self.tss = self.__temp['timestamp']

        # SELF-SCALE: tự chuẩn hoá dữ liệu để trích xuất sự tương quan giữa các field (chỉ chuẩn hoá trên các field mà giá trị giao động vô định)
        if self.__testNoSelfScale == False:
            fieldScale = list(set(self._featuresWseries) - set(self._featuresOtherscaleWseries) - set(self._featuresOtherPlotWseries))
            Xscale = self.__temp[fieldScale]
            if self.__typeScale == 'max':
                npmax = np.max(Xscale, axis=1)
                Xscale = (Xscale.T/npmax).T
                self.__temp[fieldScale] = Xscale
            elif self.__typeScale == 'minmax':
                scaler = MinMaxScaler()
                Xscale = scaler.fit_transform(Xscale.T).T
                self.__temp[fieldScale] = Xscale
            elif self.__typeScale == 'none':
                pass
            
            # những feature có thang 100, self-scale về giới hạn (0,1)
            if len(self._featuresOtherscaleWseries) > 0:
                XOtherscale = self.__temp[self._featuresOtherscaleWseries]
                const = 100
                npmax2 = np.full((XOtherscale.shape[0],), const)
                npmax2 = pd.Series(npmax2)
                XOtherscale = (XOtherscale.T/npmax2).T
                self.__temp[self._featuresOtherscaleWseries] = XOtherscale
            
            # những feature được self-scale theo fieldScale có điều nằm ở nhóm khác (plot khác)
            for f in self.__featuresOtherPlot:
                fseri = []
                for i in range(self.__pastCandle + 1):
                    fseri.append(f + '_' + str(i-self.__pastCandle))

                XOtherPlot = self.__temp[fseri]
                if self.__typeScale == 'max':
                    npmax3 = np.max(XOtherPlot, axis=1)
                    XOtherPlot = (XOtherPlot.T/npmax3).T
                    self.__temp[fseri] = XOtherPlot
                elif self.__typeScale == 'minmax':
                    scaler2 = MinMaxScaler()
                    XOtherPlot = scaler2.fit_transform(XOtherPlot.T).T
                    self.__temp[fseri] = XOtherPlot
                elif self.__typeScale == 'none':
                    pass 


        self.__X = self.__temp[self._featuresWseries]
        self.__y = self.__temp['label']
        if realDT:
            y_ = None
        else:
            y_ = self.__y
        return self.__X, y_
    
    # def __makeSampleXYwts_more(self, datanew:pd.DataFrame, numClass:int): <--- tạm không dùng
    #     '''
    #     Tạo thêm mẫu dữ liệu, gộp vào mẫu trước đo
    #     '''
    #     self.__temp_more = pd.DataFrame()
                
    #     for f in self.__features:
    #         for i in range(self.__pastCandle + 1):
    #             self.__temp_more[f + '_' + str(i-self.__pastCandle)] = datanew[f].shift(self.__pastCandle-i)
    #             self.__temp_more = self.__temp_more.copy()  # Tạo bản sao để giảm phân mảnh
                
        
    #     # label tương ứng ngày hiện tại -> nó không di chuyển gì cả
    #     _label_0 = datanew[self.__label]
          
    #     # label ngày kế
    #     _label_x = datanew[self.__label].shift(0 - self.__foreCast)

    #     # tính lệch giá trị
    #     _label_diff = (_label_x - _label_0)/_label_0
        
    #     self.__temp_more[self.__label+'_00'] = _label_0 
    #     self.__temp_more[self.__label+'_'+str(self.__foreCast)] = _label_x
    #     self.__temp_more['diff_'+self.__label] = _label_diff

    #     if numClass == 2:
    #         self.__temp_more['label'] = np.where( (_label_diff > 0) & (abs(_label_diff) > self.__thresoldDiff), 1, 0) 
    #     elif numClass ==-2:
    #         self.__temp_more['label'] = np.where((_label_diff < 0) & (abs(_label_diff) > self.__thresoldDiff), 1, 0) 
    #     else:
    #         self.__temp_more['label'] = np.where( (_label_diff > 0) & (abs(_label_diff) > self.__thresoldDiff), 2,
    #                             np.where((_label_diff < 0) & (abs(_label_diff) > self.__thresoldDiff), 1, 0) )


    #     self.__temp_more = self.__temp_more.dropna() # làm bất dữ liệu đuôi nếu khong có ffill
    #     self.__temp_more.reset_index(drop=True, inplace=True)

    #     # SELF-SCALE: tự chuẩn hoá dữ liệu để trích xuất sự tương quan giữa các field (chỉ chuẩn hoá trên các field mà giá trị giao động vô định)
    #     if self.__testNoSelfScale == False:
    #         fieldScale = list(set(self._featuresWseries) - set(self._featuresOtherscaleWseries) - set(self._featuresOtherPlotWseries))
    #         Xscale = self.__temp_more[fieldScale]
    #         if self.__typeScale == 'max':
    #             npmax = np.max(Xscale, axis=1)
    #             Xscale = (Xscale.T/npmax).T
    #             self.__temp_more[fieldScale] = Xscale
    #         elif self.__typeScale == 'minmax':
    #             scaler = MinMaxScaler()
    #             Xscale = scaler.fit_transform(Xscale.T).T
    #             self.__temp_more[fieldScale] = Xscale
    #         elif self.__typeScale == 'none':
    #             pass
            
    #         # những feature có thang 100, self-scale về giới hạn (0,1)
    #         if len(self._featuresOtherscaleWseries) > 0:
    #             XOtherscale = self.__temp_more[self._featuresOtherscaleWseries]
    #             const = 100
    #             npmax2 = np.full((XOtherscale.shape[0],), const)
    #             npmax2 = pd.Series(npmax2)
    #             XOtherscale = (XOtherscale.T/npmax2).T
    #             self.__temp_more[self._featuresOtherscaleWseries] = XOtherscale
            
    #         # những feature đước self-scale theo fieldScale có điều nằm ở nhóm khác (plot khác)
    #         for f in self.__featuresOtherPlot:
    #             fseri = []
    #             for i in range(self.__pastCandle + 1):
    #                 fseri.append(f + '_' + str(i-self.__pastCandle))

    #             XOtherPlot = self.__temp_more[fseri]
    #             if self.__typeScale == 'max':
    #                 npmax3 = np.max(XOtherPlot, axis=1)
    #                 XOtherPlot = (XOtherPlot.T/npmax3).T
    #                 self.__temp_more[fseri] = XOtherPlot
    #             elif self.__typeScale == 'minmax':
    #                 scaler2 = MinMaxScaler()
    #                 XOtherPlot = scaler2.fit_transform(XOtherPlot.T).T
    #                 self.__temp_more[fseri] = XOtherPlot
    #             elif self.__typeScale == 'none':
    #                 pass 

    #     self.__temp = pd.concat([self.__temp, self.__temp_more], ignore_index=True)

    #     self.__X = self.__temp[self._featuresWseries]
    #     self.__y = self.__temp['label']

    #     return self.__X, self.__y
    
    # def makeMoreSampleXY(self, datanew:pd.DataFrame):
    #     '''
    #     Nguyên tác là dữ liệu mới sau khi xử lý: gnas nhãn, scale.. như dữ liệu cũ rồi concat vào dữ liệu cũ.
    #     '''
    #     if self.__temp is None:
    #         print('There is no initial data!')
    #         return None
        
    #     for initF in self.__features:
    #         if initF not in datanew.columns:
    #             raise ValueError('New data columns are not suitable with inital data')
        
        
    #     if self.__typeMake == 3:
    #         return self.__makeSampleXYwts_more(datanew, numClass=3)
    #     elif self.typeMake == 2:
    #         return self.__makeSampleXYwts_more(datanew, numClass=2)
    #     elif self.typeMake == -2:
    #         return self.__makeSampleXYwts_more(datanew, numClass=-2)
    #     else:
    #         return self.__makeSampleXYwts_more(datanew, numClass=3)

    
    def __make_square_matrix(self, matrix):
        rows, cols = matrix.shape
        if rows == cols:
            return matrix
        
        max_dim = max(rows, cols)
        
        if max_dim == rows:
            # Nếu số hàng lớn hơn hoặc bằng số cột, thì thêm cột vào trái và phải
            pad_cols = max_dim - cols
            pad_left = pad_cols // 2
            pad_right = pad_cols - pad_left
            square_matrix = np.pad(matrix, ((0, 0), (pad_left, pad_right)), 
                                mode='constant', constant_values=0)
        
        if max_dim == cols:
            # Nếu số cột lớn hơn số hàng, thêm hàng vào trên và dưới
            pad_rows = max_dim - rows
            pad_top = pad_rows // 2
            pad_bottom = pad_rows - pad_top
            square_matrix = np.pad(matrix, ((pad_top, pad_bottom), (0, 0)), 
                                mode='constant', constant_values=0)
        
        return square_matrix
    
    def __setFeaturesOtherscale(self):
        # những feature Otherscale 'ADX','AROON'... cần self-scale riêng về giới hạn (0,1) nếu typeScale != 'none'
        # những feature OtherPlot 'AD','OBV'... thì vẫn theo self-scale chung. Nhưng để ở đây để hiển thị ở subplot khac cho dễ nhìn
        defineOtherscale = ['ADX','AROON','CMO','DX','MFI','MINUS_DI','PLUS_DI', 'RSI','ULTOSC','WILLR']
        defineOtherPlot = ['AD','OBV','ADOSC','ATR','NATR','TRANGE']        
        self.__features = list(set(self.__features))
        for f in self.__features:
            done = False
            for defi in defineOtherscale:
                if defi in f:
                    self.__featuresOtherscale.append(f)
                    done = True
                    break
            if done: continue
            for defi in defineOtherPlot:
                if defi in f:
                    self.__featuresOtherPlot.append(f)
                    break

        self.__features.sort()
        self.__featuresOtherscale.sort()
        self.__featuresOtherPlot.sort()

    def plotCorrelation(self, data:pd.DataFrame):
        '''
        Visualize correlation static data
        '''
        plt.figure(figsize=(10, 8))
        sbn.heatmap(data.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation matrix')
        plt.show()

    def dumpTemp(self, path:str):
        '''
        Lưu dữ liệu tạm vào file
        '''
        self.__temp.to_excel(path, index=False)


    @property
    def data(self):
        '''
        Trả về bảng dữ liệu gốc đầu vào
        '''
        return self.__data

    @property
    def typeMake(self):
        '''
        Trả về kiểu dữ liệu được tạo ra
        '''
        return self.__typeMake
    
    @property
    def typeScale(self):
        '''
        Trả về kiểu chuẩn hoá dữ liệu
        '''
        return self.__typeScale
    
    @property
    def features(self):
        '''
        Trả về danh sách các field được chọn
        '''
        return self.__features
    
    @property
    def featuresOtherscale(self):
        '''
        Trả về danh sách các field self-scale với thang khác
        '''
        return self.__featuresOtherscale
    
    @property
    def featuresOtherPlot(self):
        '''
        Trả về danh sách các field self-scale cùng thang chung nhưng vẽ ở plot khác cho dễ nhìn
        '''
        return self.__featuresOtherPlot
    
    @property
    def label(self):
        '''
        Trả về field label
        '''
        return self.__label
    
    @property
    def pastCandle(self):
        '''
        Trả về số lượng nến trước
        '''
        return self.__pastCandle
    
    @property
    def foreCast(self):
        '''
        Trả khoảng nến trước để dự đoán
        '''
        return self.__foreCast
    
    @property
    def thresoldDiff(self):
        '''
        Trả về ngưỡng lệch giá dự đoán
        '''
        return self.__thresoldDiff
    
    @property   
    def tempDataScaled(self):
        '''
        Trả về bảng dữ liệu tạm. Bảng này chứa toàn bộ dữ liệu đã chuẩn bị
        '''
        return self.__temp
    
    # propery return percent of each class
    @property
    def classPercent(self):
        '''
        Trả về tỷ lệ mỗi class trong tập dữ liệu
        '''
        total = len(self.__temp['label'])
        percent_label = {}
        for k,v in dict(Counter(self.__temp['label'])).items():
            percent_label[k] = v/total
        
        return percent_label
    
    @property
    def X(self):
        '''
        Trả về bảng dữ liệu X
        '''
        return self.__X
    
    @property
    def y(self):
        '''
        Trả về bảng dữ liệu y
        '''
        return self.__y
    
    @property
    def isDatapredict(self):
        '''
        Trả về giá trị xác định có phải là dữ liệu dùng để dự đoán hay không
        '''
        return self.__isDatapredict
    
    
def checkLabelHist(data:pd.DataFrame, typeMake:int, foreCast:int, thresoldDiff:float, showHist=True, label:str='close'):
    '''
    typeMake: 2: tăng là dương tính, còn lại là âm tính
    
             -2: giảm là dương tính, còn lại là âm tính
             
             3: tăng là nhóm 2, giảm là nhóm 1, không đổi là nhóm 0 
    '''
    # label tương ứng ngày hiện tại -> nó không di chuyển gì cả
    data = data.copy()
    _label_0 = data[label]
    # label ngày kế
    _label_x = data[label].shift(0 - foreCast)

    _label_diff = (_label_x - _label_0)/_label_0

    if typeMake == 2:
        data['label'] = np.where( (_label_diff > 0) & (abs(_label_diff) > thresoldDiff), 1, 0) 
    elif typeMake ==-2:
        data['label'] = np.where((_label_diff < 0) & (abs(_label_diff) > thresoldDiff), 1, 0) 
    else:
        data['label'] = np.where( (_label_diff > 0) & (abs(_label_diff) > thresoldDiff), 2,
                        np.where((_label_diff < 0) & (abs(_label_diff) > thresoldDiff), 1, 0) )
    
    if showHist:
        data['label'].hist()
        plt.show()
    else:
        dfstatistic = data['label'].value_counts()
        # thêm dòng để chưa tỷ lệ giưã các class
        dfstatistic = dfstatistic/len(data)
        print(dfstatistic)
        return dfstatistic
    
def addLabelUpDown(data:pd.DataFrame, foreCast:int, thresoldDiff:float, label:str='close'):
    '''
    Thêm label vào dữ liệu với giá trị (1,0,-1) tương ứng với tăng, không đổi, giảm
    '''
    # label tương ứng ngày hiện tại -> nó không di chuyển gì cả
    data = data.copy()
    _label_0 = data[label]
    # label ngày kế
    _label_x = data[label].shift(0 - foreCast)

    _label_diff = (_label_x - _label_0)/_label_0

    data['label'] = np.where( (_label_diff > 0) & (abs(_label_diff) > thresoldDiff), 1,
                    np.where((_label_diff < 0) & (abs(_label_diff) > thresoldDiff), -1, 0) )
    
    # fillna with 0
    data.fillna(0, inplace=True)
    
    return data['label']

