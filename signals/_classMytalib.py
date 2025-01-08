import pandas as pd

class funcOverlap:
    '''
    https://ta-lib.github.io/ta-lib-python/func_groups/overlap_studies.html
    '''
    @staticmethod
    def MA() -> str: return 'MA'
    @staticmethod
    def EMA() -> str: return 'EMA'
    @staticmethod
    def DEMA() -> str: return 'DEMA'
    @staticmethod
    def KAMA() -> str: return 'KAMA'
    @staticmethod
    def MIDPOINT() -> str: return 'MIDPOINT'
    @staticmethod
    def MIDPRICE() -> str: return 'MIDPRICE'
    @staticmethod
    def SAR() -> str: return 'SAR'
    @staticmethod
    def T3() -> str: return 'T3'
    @staticmethod
    def TEMA() -> str: return 'TEMA'
    @staticmethod
    def TRIMA() -> str: return 'TRIMA'
    @staticmethod
    def WMA() -> str: return 'WMA'
    @staticmethod
    def BBANDS() -> str: return 'BBANDS'

class funcMomentumIndicator:
    '''
    https://ta-lib.github.io/ta-lib-python/func_groups/momentum_indicators.html
    '''
    @staticmethod
    def ADX() -> str: return 'ADX' # can use
    @staticmethod
    def ADXR() -> str: return 'ADXR' # can use
    @staticmethod
    def _APO() -> str: return 'APO'
    @staticmethod
    def AROON() -> str: return 'AROON' # can use
    @staticmethod
    def _AROONOSC() -> str: return 'AROONOSC'
    @staticmethod
    def _BOP() -> str: return 'BOP'
    @staticmethod
    def _CCI() -> str: return 'CCI'
    @staticmethod
    def CMO() -> str: return 'CMO' # can use
    @staticmethod
    def DX() -> str: return 'DX' # can use
    @staticmethod
    def _MACD() -> str: return 'MACD'
    @staticmethod
    def _MACDEXT() -> str: return 'MACDEXT'
    @staticmethod
    def _MACDFIX() -> str: return 'MACDFIX'
    @staticmethod
    def MFI() -> str: return 'MFI' # can use
    @staticmethod
    def MINUS_DI() -> str: return 'MINUS_DI' # can use
    @staticmethod
    def _MINUS_DM() -> str: return 'MINUS_DM'
    @staticmethod
    def _MOM() -> str: return 'MOM'
    @staticmethod
    def PLUS_DI() -> str: return 'PLUS_DI' # can use
    @staticmethod
    def _PLUS_DM() -> str: return 'PLUS_DM'
    @staticmethod
    def _PPO() -> str: return 'PPO'
    @staticmethod
    def _ROC() -> str: return 'ROC'
    @staticmethod
    def _ROCP() -> str: return 'ROCP'
    @staticmethod
    def _ROCR() -> str: return 'ROCR'
    @staticmethod
    def _ROCR100() -> str: return 'ROCR100'
    @staticmethod
    def RSI() -> str: return 'RSI' # can use
    @staticmethod
    def _STOCH() -> str: return 'STOCH'
    @staticmethod
    def _STOCHF() -> str: return 'STOCHF'
    @staticmethod
    def _STOCHRSI() -> str: return 'STOCHRSI'
    @staticmethod
    def _TRIX() -> str: return 'TRIX'
    @staticmethod
    def ULTOSC() -> str: return 'ULTOSC' # can use
    @staticmethod
    def WILLR() -> str: return 'WILLR' # can use

class funcVolumeIndicator:
    '''
    https://ta-lib.github.io/ta-lib-python/func_groups/volume_indicators.html
    '''
    @staticmethod
    def AD() -> str: return 'AD'
    @staticmethod
    def ADOSC() -> str: return 'ADOSC' # consider using
    @staticmethod
    def OBV() -> str: return 'OBV'

class funcVolatilityIndicator:
    '''
    https://ta-lib.github.io/ta-lib-python/func_groups/volatility_indicators.html
    '''
    @staticmethod
    def ATR() -> str: return 'ATR' # consider using
    @staticmethod
    def NATR() -> str: return 'NATR'
    @staticmethod
    def TRANGE() -> str: return 'TRANGE'

        
class MyTalib:
    def __str__(self) -> str:
        return ''' https://ta-lib.github.io/ta-lib-python/doc_index.html \n
                   https://github.com/twopirllc/pandas-ta '''
    
    def __init__(self, type='') -> None:
        '''
        type = 'pandas_ta' -> chỉ định dùng pandas_ta
        \nTất cả ở đây ngầm định rằng lấy giá đóng cửa close là giá chính thức
        \ntype = 'multilib' -> dùng cả 2 pandas_ta và talib chỉ trong khi test xem cả 2 có giống nhau không
        \ntype mặc định: tự động chọn talib nếu có, nếu không có thì chọn pandas_ta
        '''
        self.__chosen = None
        try:
            if type == 'pandas_ta':
               import pandas_ta as ta
               self.ta = ta
               self.__chosen = 'pandas_ta'
            elif type == 'multilib':
                self.__chosen = 'multilib'
                import talib as ta
                self.talib = ta
                import pandas_ta as pta
                self.pta = pta
            else: 
                try:
                    import talib as ta
                    self.__chosen = 'ta'
                except:
                    import pandas_ta as ta
                    self.__chosen = 'pandas_ta'
                finally:
                    self.ta = ta
        except:
            print('No talib or pandas_ta')
            self.ta = None
    
    def chosen(self):
        return self.__chosen

    def normalize(self, value, min_value=-100, max_value=100):
        # Chuẩn hóa giá trị về khoảng (0, 1)
        normalized_value = (value - min_value) / (max_value - min_value)
        # Chuyển đổi về khoảng (0, 100)
        return normalized_value * 100
    
    
    #################Overlap Studies Functions####################
      

    def MA(self, data:pd.DataFrame, timeperiod=5):
        '''
        MA (Moving Average) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó được sử dụng để xác định giá trị trung bình của một dãy dữ liệu trong một khoảng thời gian nhất định.

        OK for talib and pandas_ta
        '''
        if self.__chosen == 'ta':
            return self.ta.MA(data['close'], timeperiod=timeperiod)
        elif self.__chosen == 'pandas_ta':
            return self.ta.sma(data['close'], length=timeperiod)
        elif self.__chosen == 'multilib':
            sr1 = self.talib.SMA(data['close'], timeperiod=timeperiod)
            sr1.fillna(0, inplace=True)
            sr2 = self.pta.sma(data['close'], length=timeperiod)
            sr2.fillna(0, inplace=True)
            result = (sr1 != sr2).astype(int)
            return str(result.sum()) + '/' + str(len(result))
    
    def EMA(self, data:pd.DataFrame, timeperiod=5):
        '''
        EMA (Exponential Moving Average) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó được sử dụng để xác định giá trị trung bình của một dãy dữ liệu trong một khoảng thời gian nhất định.

        OK for talib and pandas_ta
        '''
        if self.__chosen == 'ta':
            return self.ta.EMA(data['close'], timeperiod=timeperiod)
        elif self.__chosen == 'pandas_ta':
            return self.ta.ema(data['close'], length=timeperiod)
        elif self.__chosen == 'multilib':
            sr1 = self.talib.EMA(data['close'], timeperiod=timeperiod)
            sr1.fillna(0, inplace=True)
            sr2 = self.pta.ema(data['close'], length=timeperiod)
            sr2.fillna(0, inplace=True)
            result = (sr1 != sr2).astype(int)
            return str(result.sum()) + '/' + str(len(result))

    def DEMA(self, data:pd.DataFrame, timeperiod=30):
        '''
        DEMA (Double Exponential Moving Average) là một chỉ báo kỹ thuật trong thư viện ta-lib. 
        Nó được phát triển bởi Patrick G. Mulloy vào năm 1994 và là một dạng trung bình động hàm mũ (Exponential Moving Average) có khả năng giảm độ trễ so với các trung bình động khác.
        
        OK for talib and pandas_ta
        '''
        if self.__chosen == 'ta':
            # OK
            return self.ta.DEMA(data['close'], timeperiod=timeperiod)
        elif self.__chosen == 'pandas_ta':
            # OK
            return self.ta.dema(data['close'], length=timeperiod)
        elif self.__chosen == 'multilib':
            sr1 = self.talib.DEMA(data['close'], timeperiod=timeperiod)
            sr1.fillna(0, inplace=True)
            sr2 = self.pta.dema(data['close'], length=timeperiod)
            sr2.fillna(0, inplace=True)
            result = (sr1 != sr2).astype(int)
            return str(result.sum()) + '/' + str(len(result))

    def KAMA(self, data:pd.DataFrame, timeperiod=30):
        '''
        KAMA (Kaufman's Adaptive Moving Average) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó được phát triển bởi Perry J. Kaufman vào năm 1998 và là một dạng trung bình động thích ứng (Adaptive Moving Average) có khả năng tự điều chỉnh độ nhạy của nó dựa trên độ biến động của giá.
        Đặc điểm chính của KAMA:
        Adaptive: KAMA có thể điều chỉnh tốc độ phản ứng của nó với biến động giá. Khi giá biến động mạnh, KAMA trở nên "nhạy" hơn và di chuyển nhanh hơn. Ngược lại, khi giá biến động ít, KAMA trở nên "mượt mà" hơn và ít phản ứng với biến động nhỏ.
        Noise Filtering: KAMA giúp lọc nhiễu (noise) tốt hơn so với các trung bình động đơn giản (SMA) hoặc trung bình động hàm mũ (EMA) bằng cách điều chỉnh tốc độ phản ứng của nó dựa trên sự thay đổi của giá. Điều này giúp tránh các tín hiệu sai lệch trong các giai đoạn biến động thấp.
        Tham số của KAMA:
        KAMA có ba tham số chính:

        timeperiod: Khoảng thời gian sử dụng để tính toán KAMA (ví dụ: 10, 20, 30 ngày).
        fastlimit: Giới hạn trên của hệ số điều chỉnh (thường là giá trị nhỏ hơn 1, ví dụ: 0.666).
        slowlimit: Giới hạn dưới của hệ số điều chỉnh (thường là giá trị nhỏ hơn fastlimit, ví dụ: 0.0645).
        Công dụng của KAMA:
        KAMA thường được sử dụng để xác định xu hướng, phát hiện các tín hiệu mua/bán khi giá cắt qua KAMA, và giúp các nhà phân tích kỹ thuật ra quyết định giao dịch bằng cách lọc bớt nhiễu giá.

        OK for talib và pandas_ta, tuy nhiên chỉ số 2 thư viện có sự khác biệt nhiều
        '''
        if self.__chosen == 'ta':
            return self.ta.KAMA(data['close'], timeperiod=timeperiod)
        elif self.__chosen == 'pandas_ta':
            return self.ta.kama(data['close'], length=timeperiod)
        elif self.__chosen == 'multilib':
            sr1 = self.talib.KAMA(data['close'], timeperiod=timeperiod)
            sr1.fillna(0, inplace=True)
            sr2 = self.pta.kama(data['close'], length=timeperiod)
            sr2.fillna(0, inplace=True)
            result = (sr1 != sr2).astype(int)
            return str(result.sum()) + '/' + str(len(result))
         
    def MIDPOINT(self, data:pd.DataFrame, timeperiod=14):
        '''
        MIDPOINT trong thư viện ta-lib là một chỉ báo kỹ thuật tính giá trị trung điểm của một dãy dữ liệu trong một khoảng thời gian nhất định. Nó được sử dụng để xác định mức giá trung bình giữa giá cao nhất và giá thấp nhất trong một khoảng thời gian.
        /nMIDPOINT = (H + L) / 2

        OK for talib and pandas_ta
        '''
        if self.__chosen == 'ta':
            return self.ta.MIDPOINT(data['close'], timeperiod=timeperiod)
        elif self.__chosen == 'pandas_ta':
            return self.ta.midpoint(data['close'], length=timeperiod)
        elif self.__chosen == 'multilib':
            sr1 = self.talib.MIDPOINT(data['close'], timeperiod=timeperiod)
            sr1.fillna(0, inplace=True)
            sr2 = self.pta.midpoint(data['close'], length=timeperiod)
            sr2.fillna(0, inplace=True)
            result = (sr1 != sr2).astype(int)
            return str(result.sum()) + '/' + str(len(result))

    def MIDPRICE(self, data:pd.DataFrame, timeperiod=14):
        '''
        MIDPOINT: Tính giá trị trung điểm dựa trên toàn bộ dãy dữ liệu cho một khoảng thời gian cụ thể.
        MIDPRICE: Tính trung bình của giá cao nhất và thấp nhất cho từng khoảng thời gian được xác định (như ngày, tuần, tháng) và sau đó lấy trung bình các giá trị đó.

        OK for talib and pandas_ta
        '''
        if self.__chosen == 'ta':
            return self.ta.MIDPRICE(data['high'], data['low'], timeperiod=timeperiod)
        elif self.__chosen == 'pandas_ta':
            return self.ta.midprice(data['high'], data['low'], length=timeperiod)
        elif self.__chosen == 'multilib':
            sr1 = self.talib.MIDPRICE(data['high'], data['low'], timeperiod=timeperiod)
            sr1.fillna(0, inplace=True)
            sr2 = self.pta.midprice(data['high'], data['low'], length=timeperiod)
            sr2.fillna(0, inplace=True)
            result = (sr1 != sr2).astype(int)
            return str(result.sum()) + '/' + str(len(result))

    def SAR(self, data:pd.DataFrame, acceleration=0.02, maximum=0.2):
        '''
        SAR (Stop and Reverse) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó được phát triển bởi J. Welles Wilder Jr. vào năm 1978 và được sử dụng để xác định điểm dừng và đảo chiều của xu hướng giá.
        SAR dựa trên giá và thời gian. Nó được hiển thị trên biểu đồ giá dưới dạng các điểm trên hoặc dưới giá hiện tại. Khi giá vượt qua điểm SAR, chỉ báo SAR sẽ đảo chiều và xuất hiện ở phía đối diện của giá.
        SAR có thể được sử dụng để xác định điểm dừng lỗ hoặc điểm dừng lời trong giao dịch.
        Tham số của SAR:
        SAR có hai tham số chính:

        acceleration: Tỷ lệ tăng tốc của điểm SAR khi giá tiếp tục di chuyển theo xu hướng.
        maximum: Giá trị tối đa mà điểm SAR có thể đạt được.

        Lưu ý: talib_pandas không có chỉ số này
        '''
        if self.__chosen == 'ta':
            return self.ta.SAR(data['high'], data['low'], acceleration=acceleration, maximum=maximum)
        elif self.__chosen == 'pandas_ta':
            return self.ta.sar(data['high'], data['low'], acceleration=acceleration, maximum=maximum)
        elif self.__chosen == 'multilib':
            sr1 = self.talib.SAR(data['high'], data['low'], acceleration=acceleration, maximum=maximum)
            sr1.fillna(0, inplace=True)
            sr2 = self.talib.sar(data['high'], data['low'], acceleration=acceleration, maximum=maximum)
            sr2.fillna(0, inplace=True)
            result = (sr1 != sr2).astype(int)
            return str(result.sum()) + '/' + str(len(result))
        
    def T3(self, data:pd.DataFrame, timeperiod=10, vfactor=0.7):
        '''
        T3 (Triple Exponential Moving Average) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó được phát triển bởi Tim Tillson vào năm 1998 và là một dạng trung bình động hàm mũ (Exponential Moving Average) có khả năng giảm độ trễ so với các trung bình động khác.
        T3 được tính toán dựa trên ba bước:

        Bước 1: Tính toán giá trị EMA (Exponential Moving Average) của dãy dữ liệu cơ bản (close, high, low, open, volume, etc.) với một khoảng thời gian cụ thể.
        Bước 2: Tính toán giá trị EMA của EMA ở bước 1 với cùng khoảng thời gian.
        Bước 3: Tính toán giá trị EMA của EMA ở bước 2 với cùng khoảng thời gian.
        Tham số của T3:
        T3 có hai tham số chính:
        timeperiod: Khoảng thời gian sử dụng để tính toán T3 (ví dụ: 10, 20, 30 ngày).
        vfactor: Hệ số điều chỉnh tốc độ phản ứng của T3 với biến động giá.

        OK for talib and pandas_ta
        '''
        if self.__chosen == 'ta':
            return self.ta.T3(data['close'], timeperiod=timeperiod, vfactor=vfactor)
        elif self.__chosen == 'pandas_ta':
            return self.ta.t3(data['close'], length=timeperiod, vfactor=vfactor)
        elif self.__chosen == 'multilib':
            sr1 = self.talib.T3(data['close'], timeperiod=timeperiod, vfactor=vfactor)
            sr1.fillna(0, inplace=True)
            sr2 = self.pta.t3(data['close'], length=timeperiod, vfactor=vfactor)
            sr2.fillna(0, inplace=True)
            result = (sr1 != sr2).astype(int)
            return str(result.sum()) + '/' + str(len(result))

    def TEMA(self, data:pd.DataFrame, timeperiod=10):
        '''
        TEMA (Triple Exponential Moving Average) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó là một dạng trung bình động hàm mũ (Exponential Moving Average) có khả năng giảm độ trễ so với các trung bình động khác.
        TEMA được tính toán dựa trên ba bước:

        Bước 1: Tính toán giá trị EMA (Exponential Moving Average) của dãy dữ liệu cơ bản (close, high, low, open, volume, etc.) với một khoảng thời gian cụ thể.
        Bước 2: Tính toán giá trị EMA của EMA ở bước 1 với cùng khoảng thời gian.
        Bước 3: Tính toán giá trị EMA của EMA ở bước 2 với cùng khoảng thời gian.
        Tham số của TEMA:
        TEMA có một tham số chính:
        timeperiod: Khoảng thời gian sử dụng để tính toán TEMA (ví dụ: 10, 20, 30 ngày).
        '''
        if self.__chosen == 'ta':
            return self.ta.TEMA(data['close'], timeperiod=timeperiod)
        elif self.__chosen == 'pandas_ta':
            return self.ta.tema(data['close'], length=timeperiod)
        elif self.__chosen == 'multilib':
            sr1 = self.talib.TEMA(data['close'], timeperiod=timeperiod)
            sr1.fillna(0, inplace=True)
            sr2 = self.pta.tema(data['close'], length=timeperiod)
            sr2.fillna(0, inplace=True)
            result = (sr1 != sr2).astype(int)
            return str(result.sum()) + '/' + str(len(result))

    def TRIMA(self, data:pd.DataFrame, timeperiod=10):
        '''
        TRIMA (Triangular Moving Average) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó là một dạng trung bình động hàm mũ (Exponential Moving Average) có khả năng giảm độ trễ so với các trung bình động khác.
        TRIMA được tính toán dựa trên ba bước:

        Bước 1: Tính toán giá trị SMA (Simple Moving Average) của dãy dữ liệu cơ bản (close, high, low, open, volume, etc.) với một khoảng thời gian cụ thể.
        Bước 2: Tính toán giá trị SMA của SMA ở bước 1 với cùng khoảng thời gian.
        Bước 3: Tính toán giá trị SMA của SMA ở bước 2 với cùng khoảng thời gian.
        Tham số của TRIMA:
        TRIMA có một tham số chính:
        timeperiod: Khoảng thời gian sử dụng để tính toán TRIMA (ví dụ: 10, 20, 30 ngày).

        OK for talib and pandas_ta
        '''
        if self.__chosen == 'ta':
            return self.ta.TRIMA(data['close'], timeperiod=timeperiod)
        elif self.__chosen == 'pandas_ta':
            return self.ta.trima(data['close'], length=timeperiod)
        elif self.__chosen == 'multilib':
            sr1 = self.talib.TRIMA(data['close'], timeperiod=timeperiod)
            sr1.fillna(0, inplace=True)
            sr2 = self.pta.trima(data['close'], length=timeperiod)
            sr2.fillna(0, inplace=True)
            result = (sr1 != sr2).astype(int)
            return str(result.sum()) + '/' + str(len(result))
    
    def WMA(self, data:pd.DataFrame, timeperiod=10):
        '''
        WMA (Weighted Moving Average) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó là một dạng trung bình động có khả năng giảm độ trễ so với các trung bình động khác.
        WMA được tính toán dựa trên một dãy trọng số cụ thể. Mỗi giá trị trong dãy dữ liệu cơ bản (close, high, low, open, volume, etc.) được nhân với một trọng số tương ứng và sau đó lấy trung bình các giá trị đó.
        Tham số của WMA:
        WMA có một tham số chính:
        timeperiod: Khoảng thời gian sử dụng để tính toán WMA (ví dụ: 10, 20, 30 ngày).

        OK for talib and pandas_ta
        '''
        if self.__chosen == 'ta':
            return self.ta.WMA(data['close'], timeperiod=timeperiod)
        elif self.__chosen == 'pandas_ta':
            return self.ta.wma(data['close'], length=timeperiod)
        elif self.__chosen == 'multilib':
            sr1 = self.talib.WMA(data['close'], timeperiod=timeperiod)
            sr1.fillna(0, inplace=True)
            sr2 = self.pta.wma(data['close'], length=timeperiod)
            sr2.fillna(0, inplace=True)
            result = (sr1 != sr2).astype(int)
            return str(result.sum()) + '/' + str(len(result))
        
    def BBANDS(self, data:pd.DataFrame, timeperiod=10, nbdevup=2, nbdevdn=2, matype=0):
        '''
        BBANDS (Bollinger Bands) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó được phát triển bởi John Bollinger vào năm 1980 và được sử dụng để xác định mức hỗ trợ và kháng cự của giá.
        BBANDS dựa trên giá và thời gian. Nó được hiển thị trên biểu đồ giá dưới dạng ba dải (band) trên hoặc dưới giá hiện tại. Dải trên (upper band) và dải dưới (lower band) được tính toán dựa trên độ lệch chuẩn của giá so với giá trung bình động (moving average).
        BBANDS có thể được sử dụng để xác định mức giá mục tiêu, điểm mua và điểm bán trong giao dịch.
        Tham số của BBANDS:
        BBANDS có bốn tham số chính:

        timeperiod: Khoảng thời gian sử dụng để tính toán BBANDS (ví dụ: 10, 20, 30 ngày).
        nbdevup: Số lần độ lệch chuẩn của giá được sử dụng để tính toán dải trên (upper band).
        nbdevdn: Số lần độ lệch chuẩn của giá được sử dụng để tính toán dải dưới (lower band).
        matype: Phương pháp tính toán giá trung bình động (moving average) được sử dụng để tính toán dải trung bình (middle band). Có ba phương pháp chính:

        SMA (Simple Moving Average): Phương pháp trung bình động đơn giản.
        EMA (Exponential Moving Average): Phương pháp trung bình động hàm mũ.
        WMA (Weighted Moving Average): Phương pháp trung bình động có trọng số.

        ** Lưu ý: trả về 3 feature: df['BBANDSup'], df['BBANDS'], df['BBANDSlow']
        Only OK for talib. Pandas_ta trả về nhiều chỉ số hơn, sẽ nghiên cứu sau
        '''
        if self.__chosen == 'ta':
            # OK 
            return self.ta.BBANDS(data['close'], timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)
        elif self.__chosen == 'pandas_ta':
            # return self.ta.bbands(data['close'], length=timeperiod, std=nbdevup)
            print('Nghiên cứu sau')
        elif self.__chosen == 'multilib':
            print('Nghiên cứu sau')
            # _, sr1, _ = self.talib.BBANDS(data['close'], timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)
            # sr1.fillna(0, inplace=True)
            # _, sr2, _ = self.pta.bbands(data['close'], length=timeperiod, std=nbdevup)
            # sr2.fillna(0, inplace=True)
            # result = (sr1 != sr2).astype(int)
            # return str(result.sum()) + '/' + str(len(result))
    
        
   #################Momentum Indicator Functions####################
    

    def ADX(self, data:pd.DataFrame, timeperiod=14):
        '''
        ADX (Average Directional Movement Index) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó được phát triển bởi J. Welles Wilder Jr. vào năm 1978 và được sử dụng để xác định xu hướng của giá.
        ADX dựa trên giá và thời gian. Nó được hiển thị trên biểu đồ giá dưới dạng một dải (band) từ 0 đến 100. Giá trị của ADX thường dao động từ 0 đến 100. ADX có thể được sử dụng để xác định xu hướng mạnh yếu của giá.
        ADX có thể được sử dụng để xác định xu hướng của giá, phát hiện các tín hiệu mua/bán khi ADX cắt qua mức 20 hoặc 25, và giúp các nhà phân tích kỹ thuật ra quyết định giao dịch bằng cách xác định mức độ mạnh yếu của xu hướng.
        Tham số của ADX:
        ADX có một tham số chính:
        timeperiod: Khoảng thời gian sử dụng để tính toán ADX (ví dụ: 10, 20, 30 ngày) -> thường là 14

        Only OK for talib. Với pandas_ta trả về 3 đường, sẽ nghiên cứu sau
        '''
        if self.__chosen == 'ta':
            # Trả về 1 đường
            return self.ta.ADX(data['high'], data['low'], data['close'], timeperiod=timeperiod)
        elif self.__chosen == 'pandas_ta':
            # Trả về 3 đường
            return self.ta.adx(data['high'], data['low'], data['close'], length=timeperiod)
        elif self.__chosen == 'multilib':
            print('Nghiên cứu sau')
            # sr1 = self.talib.ADX(data['high'], data['low'], data['close'], timeperiod=timeperiod)
            # sr1.fillna(0, inplace=True)
            # sr2 = self.pta.adx(data['high'], data['low'], data['close'], length=timeperiod)
            # sr2.fillna(0, inplace=True)
            # result = (sr1 != sr2).astype(int)
            # return str(result.sum()) + '/' + str(len(result))
        
    def ADXR(self, data:pd.DataFrame, timeperiod=14):
        '''
        ADXR (Average Directional Movement Rating) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó được phát triển bởi J. Welles Wilder Jr. vào năm 1978 và là một dạng chỉ báo kết hợp giữa ADX và ADX của một khoảng thời gian trước đó.
        ADXR dựa trên giá và thời gian. Nó được hiển thị trên biểu đồ giá dưới dạng một dải (band) từ 0 đến 100. Giá trị của ADXR thường dao động từ 0 đến 100. ADXR có thể được sử dụng để xác định xu hướng mạnh yếu của giá.
        ADXR có thể được sử dụng để xác định xu hướng của giá, phát hiện các tín hiệu mua/bán khi ADXR cắt qua mức 20 hoặc 25, và giúp các nhà phân tích kỹ thuật ra quyết định giao dịch bằng cách xác định mức độ mạnh yếu của xu hướng.
        Tham số của ADXR:
        ADXR có một tham số chính: 
        timeperiod: Khoảng thời gian sử dụng để tính toán ADXR (ví dụ: 10, 20, 30 ngày) -> thường là 14
        
        Lưu ý: talib_pandas không có chỉ số này
        '''
        if self.__chosen == 'ta':
            return self.ta.ADXR(data['high'], data['low'], data['close'], timeperiod=timeperiod)
        elif self.__chosen == 'pandas_ta':
            print('Lưu ý: talib_pandas không có chỉ số này')
            # return self.ta.adxr(data['high'], data['low'], data['close'], length=timeperiod)
        elif self.__chosen == 'multilib':
            # sr1 = self.talib.ADXR(data['high'], data['low'], data['close'], timeperiod=timeperiod)
            # sr1.fillna(0, inplace=True)
            # sr2 = self.pta.adxr(data['high'], data['low'], data['close'], length=timeperiod)
            # sr2.fillna(0, inplace=True)
            # result = (sr1 != sr2).astype(int)
            # return str(result.sum()) + '/' + str(len(result))
            print('Lưu ý: talib_pandas không có chỉ số này')
    
    def __APO(self, data:pd.DataFrame, fastperiod=12, slowperiod=26, matype=0):
        '''
        APO (Absolute Price Oscillator) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó được sử dụng để xác định sự khác biệt giữa hai giá trị trung bình động (moving average) của giá.
        APO dựa trên giá và thời gian. Nó được hiển thị trên biểu đồ giá dưới dạng một dải (band) hoặc một đường trung bình (line). APO có thể được sử dụng để xác định xu hướng của giá, phát hiện các tín hiệu mua/bán khi APO cắt qua mức 0, và giúp các nhà phân tích kỹ thuật ra quyết định giao dịch bằng cách xác định mức độ mạnh yếu của xu hướng.
        Tham số của APO:
        APO có ba tham số chính:
        fastperiod: Khoảng thời gian sử dụng để tính toán giá trị trung bình nhanh (ví dụ: 10, 20, 30 ngày) -> thường là 12
        slowperiod: Khoảng thời gian sử dụng để tính toán giá trị trung bình chậm (ví dụ: 10, 20, 30 ngày) -> thường là 26
        matype: Phương pháp tính toán giá trung bình động (moving average) được sử dụng để tính toán APO. Có ba phương pháp chính:
            SMA (Simple Moving Average): Phương pháp trung bình động đơn giản.
            EMA (Exponential Moving Average): Phương pháp trung bình động hàm mũ.
            WMA (Weighted Moving Average): Phương pháp trung bình động
        
        OK for talib và pandas_ta. Tuy vậy Tạm chưa dùng chỉ báo này vì giao động vô định, có số âm.
        '''
        if self.__chosen == 'ta':
            return self.ta.APO(data['close'], fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)
        elif self.__chosen == 'pandas_ta':
            return self.ta.apo(data['close'], fastperiod=fastperiod, slowperiod=slowperiod)
        elif self.__chosen == 'multilib':
            sr1 = self.talib.APO(data['close'], fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)
            sr1.fillna(0, inplace=True)
            sr2 = self.pta.apo(data['close'], fastperiod=fastperiod, slowperiod=slowperiod)
            sr2.fillna(0, inplace=True)
            result = (sr1 != sr2).astype(int)
            return str(result.sum()) + '/' + str(len(result))
    
    # return many
    def AROON(self, data:pd.DataFrame, timeperiod=14):
        '''
        AROON (Aroon) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó được phát triển bởi Tushar Chande vào năm 1995 và được sử dụng để xác định xu hướng của giá.
        AROON dựa trên giá và thời gian. Nó được hiển thị trên biểu đồ giá dưới dạng hai đường (line) hoặc hai dải (band). AROON có thể được sử dụng để xác định xu hướng của giá, phát hiện các tín hiệu mua/bán khi AROON Up cắt qua AROON Down, và giúp các nhà phân tích kỹ thuật ra quyết định giao dịch bằng cách xác định mức độ mạnh yếu của xu hướng.
        Chỉ báo kỹ thuật này tập trung vào 25 phiên giao dịch gần nhất, và được theo dõi trong phạm vi từ 0 đến 100.
        Chỉ số Aroon Up trên 50 có nghĩa là giá đã đạt mức cao mới trong vòng 12,5 phiên vừa qua. Aroon Up gần 100 có nghĩa là mức cao nhất đã được ghi nhận gần đây.
        Tương tự, khi Aroon Down trên 50, mức giá thấp nhất đã được ghi nhận trong 12,5 phiên. Aroon Down gần 100 có nghĩa là mức thấp nhất đã được ghi nhận gần đây.
        timeperiod: Khoảng thời gian sử dụng để tính toán AROON (ví dụ: 10, 20, 30 ngày) -> thường là 14

        Trả về 2 chỉ số aroondown, aroonup
        OK for talib và pandas_ta
        '''
        if self.__chosen == 'ta':
            return self.ta.AROON(data['high'], data['low'], timeperiod=timeperiod)
        elif self.__chosen == 'pandas_ta':
            result = self.ta.aroon(data['high'], data['low'], length=timeperiod)
            return result.iloc[:, 0], result.iloc[:, 1]
        elif self.__chosen == 'multilib':
            sr1, _ = self.talib.AROON(data['high'], data['low'], timeperiod=timeperiod)
            sr1.fillna(0, inplace=True)
            result = self.pta.aroon(data['high'], data['low'], length=timeperiod)
            sr2 = result.iloc[:, 0]
            sr2.fillna(0, inplace=True)
            result = (sr1 != sr2).astype(int)
            return str(result.sum()) + '/' + str(len(result))
    
    def __AROONOSC(self, data:pd.DataFrame, timeperiod=14):
        '''
        AROONOSC (Aroon Oscillator) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó được phát triển bởi Tushar Chande vào năm 1995 và là một dạng chỉ báo kết hợp giữa AROON Up và AROON Down.
        AROONOSC dựa trên giá và thời gian. Nó được hiển thị trên biểu đồ giá dưới dạng một dải (band) hoặc một đường trung bình (line). AROONOSC có thể được sử dụng để xác định xu hướng của giá, phát hiện các tín hiệu mua/bán khi AROONOSC cắt qua mức 0, và giúp các nhà phân tích kỹ thuật ra quyết định giao dịch bằng cách xác định mức độ mạnh yếu của xu hướng.
        Tham số của AROONOSC:
        AROONOSC có một tham số chính:
        timeperiod: Khoảng thời gian sử dụng để tính toán AROONOSC (ví dụ: 10, 20, 30 ngày) -> thường là 14
        
        Lưu ý: talib_pandas không có chỉ số này vì nó trả về cả 3 down, up, osc ở AROON
        Tạm không dùng vì có giá trị âm
        '''
        if self.__chosen == 'ta':
            return self.ta.AROONOSC(data['high'], data['low'], timeperiod=timeperiod)
        elif self.__chosen == 'pandas_ta':
            return self.ta.aroonosc(data['high'], data['low'], length=timeperiod)
        elif self.__chosen == 'multilib':
            sr1 = self.talib.AROONOSC(data['high'], data['low'], timeperiod=timeperiod)
            sr1.fillna(0, inplace=True)
            sr2 = self.pta.aroonosc(data['high'], data['low'], length=timeperiod)
            sr2.fillna(0, inplace=True)
            result = (sr1 != sr2).astype(int)
            return str(result.sum()) + '/' + str(len(result))
    
    def __BOP(self, data:pd.DataFrame):
        '''
        BOP (Balance Of Power) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó được sử dụng để xác định sức mạnh của mua/bán trên thị trường.
        BOP dựa trên giá và thời gian. Nó được hiển thị trên biểu đồ giá dưới dạng một dải (band) hoặc một đường trung bình (line). BOP có thể được sử dụng để xác định sức mạnh của mua/bán trên thị trường, phát hiện các tín hiệu mua/bán khi BOP cắt qua mức 0, và giúp các nhà phân tích kỹ thuật ra quyết định giao dịch bằng cách xác định mức độ mạnh yếu của mua/bán.
        Tham số của BOP:
        BOP không có tham số chính.
        Khi giá đóng cửa cao hơn giá mở cửa, BOP sẽ là dương, cho thấy sức mua đang tăng lên.
        Ngược lại, khi giá đóng cửa thấp hơn giá mở cửa, BOP sẽ là âm, cho thấy sức bán đang tăng lên. Nếu giá đóng cửa bằng giá mở cửa, BOP sẽ là 0, cho thấy sức mua và sức bán đang cân bằng.

        OK for talib và pandas_ta. Tuy nhiên tạm không dùng vì có giá trị âm. Tuy nhiên có thể xem xét dùng ngoài marchin learning vì nó có liên quan đến mô hình nến.
        '''
        if self.__chosen == 'ta':
            return self.ta.BOP(data['open'], data['high'], data['low'], data['close'])
        elif self.__chosen == 'pandas_ta':
            return self.ta.bop(data['open'], data['high'], data['low'], data['close'])
        elif self.__chosen == 'multilib':
            sr1 = self.talib.BOP(data['open'], data['high'], data['low'], data['close'])
            sr1.fillna(0, inplace=True)
            sr2 = self.pta.bop(data['open'], data['high'], data['low'], data['close'])
            sr2.fillna(0, inplace=True)
            result = (sr1 != sr2).astype(int)
            return str(result.sum()) + '/' + str(len(result))
    
    def __CCI(self, data:pd.DataFrame, timeperiod=14):
        '''
        CCI (Commodity Channel Index) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó được phát triển bởi Donald Lambert vào năm 1980 và được sử dụng để xác định mức độ mua/bán quá mua/quá bán của giá.
        CCI dựa trên giá và thời gian. Nó được hiển thị trên biểu đồ giá dưới dạng một dải (band) từ -100 đến 100. Giá trị của CCI thường dao động từ -100 đến 100. CCI có thể được sử dụng để xác định mức độ mua/bán quá mua/quá bán của giá, phát hiện các tín hiệu mua/bán khi CCI cắt qua mức 0, và giúp các nhà phân tích kỹ thuật ra quyết định giao dịch bằng cách xác định mức độ mạnh yếu của mua/bán.
        Tham số của CCI:
        CCI có một tham số chính:
        timeperiod: Khoảng thời gian sử dụng để tính toán CCI (ví dụ: 10, 20, 30 ngày) -> thường là 14

        OK for talib và pandas_ta. Tuy nhiên cần đưa khảng giới hạn nó về 0, 0.5, 1 thay vì -100, 0, 100
        Tạm không dùng vì giá trị nó vượt ra khỏi 100 và -100
        '''
        if self.__chosen == 'ta':
            sr = self.ta.CCI(data['high'], data['low'], data['close'], timeperiod=timeperiod)
            sr = sr.apply(self.normalize)
            return sr
        elif self.__chosen == 'pandas_ta':
            sr = self.ta.cci(data['high'], data['low'], data['close'], length=timeperiod)
            sr = sr.apply(self.normalize)
            return sr
        elif self.__chosen == 'multilib':
            sr1 = self.talib.CCI(data['high'], data['low'], data['close'], timeperiod=timeperiod)
            sr1.fillna(0, inplace=True)
            sr2 = self.pta.cci(data['high'], data['low'], data['close'], length=timeperiod)
            sr2.fillna(0, inplace=True)
            result = (sr1 != sr2).astype(int)
            return str(result.sum()) + '/' + str(len(result))
    
    def CMO(self, data:pd.DataFrame, timeperiod=14):
        '''
        CMO (Chande Momentum Oscillator) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó được phát triển bởi Tushar Chande vào năm 1994 và được sử dụng để xác định động lượng của giá.
        CMO dựa trên giá và thời gian. Nó được hiển thị trên biểu đồ giá dưới dạng một dải (band) hoặc một đường trung bình (line). CMO có thể được sử dụng để xác định động lượng của giá, phát hiện các tín hiệu mua/bán khi CMO cắt qua mức 0, và giúp các nhà phân tích kỹ thuật ra quyết định giao dịch bằng cách xác định mức độ mạnh yếu của động lượng.
        Tham số của CMO:
        CMO có một tham số chính:
        timeperiod: Khoảng thời gian sử dụng để tính toán CMO (ví dụ: 10, 20, 30 ngày) -> thường là 14

        OK for talib và pandas_ta. Tuy nhiên cần đưa khảng giới hạn nó về 0, 0.5, 1 thay vì -100, 0, 100
        '''
        if self.__chosen == 'ta':
            sr = self.ta.CMO(data['close'], timeperiod=timeperiod)
            sr = sr.apply(self.normalize)
            return sr
        elif self.__chosen == 'pandas_ta':
            sr = self.ta.cmo(data['close'], length=timeperiod)
            sr = sr.apply(self.normalize)
            return sr
        elif self.__chosen == 'multilib':
            sr1 = self.talib.CMO(data['close'], timeperiod=timeperiod)
            sr1.fillna(0, inplace=True)
            sr2 = self.pta.cmo(data['close'], length=timeperiod)
            sr2.fillna(0, inplace=True)
            result = (sr1 != sr2).astype(int)
            return str(result.sum()) + '/' + str(len(result))
    
    def DX(self, data:pd.DataFrame, timeperiod=14):
        '''
        DX (Directional Movement Index) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó được phát triển bởi J. Welles Wilder Jr. vào năm 1978 và được sử dụng để xác định xu hướng của giá.
        DX dựa trên giá và thời gian. Nó được hiển thị trên biểu đồ giá dưới dạng một dải (band) từ 0 đến 100. Giá trị của DX thường dao động từ 0 đến 100. DX có thể được sử dụng để xác định xu hướng mạnh yếu của giá.
        DX có thể được sử dụng để xác định xu hướng của giá, phát hiện các tín hiệu mua/bán khi DX cắt qua mức 20 hoặc 25, và giúp các nhà phân tích kỹ thuật ra quyết định giao dịch bằng cách xác định mức độ mạnh yếu của xu hướng.
        Tham số của DX:
        DX có một tham số chính:
        timeperiod: Khoảng thời gian sử dụng để tính toán DX (ví dụ: 10, 20, 30 ngày) -> thường là 14

        Lưu ý: pandas_ta không có chỉ số này
        '''
        if self.__chosen == 'ta':
            return self.ta.DX(data['high'], data['low'], data['close'], timeperiod=timeperiod)
        elif self.__chosen == 'pandas_ta':
            print('Lưu ý: pandas_ta không có chỉ số này')
            # return self.ta.dx(data['high'], data['low'], data['close'], length=timeperiod)
        elif self.__chosen == 'multilib':
            print('Lưu ý: pandas_ta không có chỉ số này')
            # sr1 = self.talib.DX(data['high'], data['low'], data['close'], timeperiod=timeperiod)
            # sr1.fillna(0, inplace=True)
            # sr2 = self.pta.dx(data['high'], data['low'], data['close'], length=timeperiod)
            # sr2.fillna(0, inplace=True)
            # result = (sr1 != sr2).astype(int)
            # return str(result.sum()) + '/' + str(len(result))
    
    def _MACD(self, data:pd.DataFrame, fastperiod=12, slowperiod=26, signalperiod=9):
        '''
        MACD (Moving Average Convergence Divergence) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó được phát triển bởi Gerald Appel vào năm 1979 và được sử dụng để xác định sự chuyển đổi và phân kỳ của giá.
        MACD dựa trên giá và thời gian. Nó được hiển thị trên biểu đồ giá dưới dạng một dải (band) hoặc một đường trung bình (line). MACD có thể được sử dụng để xác định sự chuyển đổi và phân kỳ của giá, phát hiện các tín hiệu mua/bán khi MACD cắt qua mức 0, và giúp các nhà phân tích kỹ thuật ra quyết định giao dịch bằng cách xác định mức độ mạnh yếu của sự chuyển đổi và phân kỳ.
        Tham số của MACD:
        MACD có ba tham số chính:
        fastperiod: Khoảng thời gian sử dụng để tính toán giá trị trung bình nhanh (ví dụ: 10, 20, 30 ngày) -> thường là 12
        slowperiod: Khoảng thời gian sử dụng để tính toán giá trị trung bình chậm (ví dụ: 10, 20, 30 ngày) -> thường là 26
        signalperiod: Khoảng thời gian sử dụng để tính toán giá trị trung bình tín hiệu (ví dụ: 10, 20, 30 ngày) -> thường là 9

        Trả về: macd, macdsignal, macdhist

        OK for talib và pandas_ta. Tuy nhiên dao động bất định nên cân nhắc dùng 2 đường EMA (12), EMA (26) và EMA (9)
        '''
        if self.__chosen == 'ta':
            return self.ta.MACD(data['close'], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
        elif self.__chosen == 'pandas_ta':
            result = self.ta.macd(data['close'], fast=fastperiod, slow=slowperiod, signal=signalperiod)
            return result.iloc[:, 0], result.iloc[:, 1], result.iloc[:, 2]
        elif self.__chosen == 'multilib':
            sr1 = self.talib.MACD(data['close'], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)[0]
            sr1.fillna(0, inplace=True)
            result = self.pta.macd(data['close'], fast=fastperiod, slow=slowperiod, signal=signalperiod)
            sr2 = result.iloc[:, 0]
            sr2.fillna(0, inplace=True)
            result = (sr1 != sr2).astype(int)
            return str(result.sum()) + '/' + str(len(result))
    
    def __MACDEXT(self, data:pd.DataFrame, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0):
        '''
        MACDEXT (MACD with controllable MA type) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó là một dạng mở rộng của MACD với khả năng chọn loại giá trị trung bình động (moving average) được sử dụng để tính toán MACD.
        MACDEXT dựa trên giá và thời gian. Nó được hiển thị trên biểu đồ giá dưới dạng một dải (band) hoặc một đường trung bình (line). MACDEXT có thể được sử dụng để xác định sự chuyển đổi và phân kỳ của giá, phát hiện các tín hiệu mua/bán khi MACDEXT cắt qua mức 0, và giúp các nhà phân tích kỹ thuật ra quyết định giao dịch bằng cách xác định mức độ mạnh yếu của sự chuyển đổi và phân kỳ.
        Tham số của MACDEXT:
        MACDEXT có sáu tham số chính:
        fastperiod: Khoảng thời gian sử dụng để tính toán giá trị trung bình nhanh (ví dụ: 10, 20, 30 ngày) -> thường là 12
        fastmatype: Phương pháp tính toán giá trung bình động (moving average) được sử dụng để tính toán giá trị trung bình nhanh. Có ba phương pháp chính:
            SMA (Simple Moving Average): Phương pháp trung bình động đơn giản.
            EMA (Exponential Moving Average): Phương pháp trung bình động hàm mũ.
            WMA (Weighted Moving Average): Phương pháp trung bình động có trọng số.
        slowperiod: Khoảng thời gian sử dụng để tính toán giá trị trung bình chậm (ví dụ: 10, 20, 30 ngày) -> thường là 26
        slowmatype: Phương pháp tính toán giá trung bình động (moving average) được sử dụng để tính toán giá trị trung bình chậm. Có ba phương pháp chính:
            SMA (Simple Moving Average): Phương pháp trung bình động đơn giản.
            EMA (Exponential Moving Average): Phương pháp trung bình động hàm mũ.
            WMA (Weighted Moving Average): Phương pháp trung bình động có trọng số.
        signalperiod: Khoảng thời gian sử dụng để tính toán giá trị trung bình tín hiệu (ví dụ: 10, 20, 30 ngày) -> thường là 9
        signalmatype: Phương pháp tính toán giá trung bình động (moving average) được sử dụng để tính toán giá trị trung bình tín hiệu. Có ba phương pháp chính:
            SMA (Simple Moving Average): Phương pháp trung bình động đơn giản.
            EMA (Exponential Moving Average): Phương pháp trung bình động hàm mũ.
            WMA (Weighted Moving Average): Phương pháp trung bình động có trọng số.

        Trả về: macd, macdsignal, macdhist

        Lưu ý: talib_pandas không có chỉ số này

        Tạm không dùng
        '''
        if self.__chosen == 'ta':
            return self.ta.MACDEXT(data['close'], fastperiod=fastperiod, fastmatype=fastmatype, slowperiod=slowperiod, slowmatype=slowmatype, signalperiod=signalperiod, signalmatype=signalmatype)
        elif self.__chosen == 'pandas_ta':
            return self.ta.macd_ext(data['close'], fast=fastperiod, fastma=fastmatype, slow=slowperiod, slowma=slowmatype, signal=signalperiod, signalm=signalmatype)
        elif self.__chosen == 'multilib':
            sr1 = self.talib.MACDEXT(data['close'], fastperiod=fastperiod, fastmatype=fastmatype, slowperiod=slowperiod, slowmatype=slowmatype, signalperiod=signalperiod, signalmatype=signalmatype)[0]
            sr1.fillna(0, inplace=True)
            sr2 = self.pta.macd_ext(data['close'], fast=fastperiod, fastma=fastmatype, slow=slowperiod, slowma=slowmatype, signal=signalperiod, signalm=signalmatype)
            sr2.fillna(0, inplace=True)
            result = (sr1 != sr2).astype(int)
            return str(result.sum()) + '/' + str(len(result))
    
    def __MACDFIX(self, data:pd.DataFrame, signalperiod=9):
        '''
        MACDFIX (Moving Average Convergence Divergence Fix 12/26) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó là một dạng cố định của MACD với giá trị trung bình nhanh và chậm là 12 và 26.
        MACDFIX dựa trên giá và thời gian. Nó được hiển thị trên biểu đồ giá dưới dạng một dải (band) hoặc một đường trung bình (line). MACDFIX có thể được sử dụng để xác định sự chuyển đổi và phân kỳ của giá, phát hiện các tín hiệu mua/bán khi MACDFIX cắt qua mức 0, và giúp các nhà phân tích kỹ thuật ra quyết định giao dịch bằng cách xác định mức độ mạnh yếu của sự chuyển đổi và phân kỳ.
        Tham số của MACDFIX:
        MACDFIX có một tham số chính:
        signalperiod: Khoảng thời gian sử dụng để tính toán giá trị trung bình tín hiệu (ví dụ: 10, 20, 30 ngày) -> thường là 9

        Trả về: macd, macdsignal, macdhist

        Lưu ý: talib_pandas không có chỉ số này

        Tạm không dùng
        '''
        if self.__chosen == 'ta':
            return self.ta.MACDFIX(data['close'], signalperiod=signalperiod)
        elif self.__chosen == 'pandas_ta':
            return self.ta.macd_fix(data['close'], signal=signalperiod)
        elif self.__chosen == 'multilib':
            sr1 = self.talib.MACDFIX(data['close'], signalperiod=signalperiod)[0]
            sr1.fillna(0, inplace=True)
            sr2 = self.pta.macd_fix(data['close'], signal=signalperiod)
            sr2.fillna(0, inplace=True)
            result = (sr1 != sr2).astype(int)
            return str(result.sum()) + '/' + str(len(result))
    
    def MFI(self, data:pd.DataFrame, timeperiod=14):
        '''
        MFI (Money Flow Index) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó được phát triển bởi Gene Quong và Avrum Soudack vào năm 1979 và được sử dụng để xác định sức mạnh của mua/bán trên thị trường.
        MFI dựa trên giá và thời gian. Nó được hiển thị trên biểu đồ giá dưới dạng một dải (band) từ 0 đến 100. Giá trị của MFI thường dao động từ 0 đến 100. MFI có thể được sử dụng để xác định sức mạnh của mua/bán trên thị trường, phát hiện các tín hiệu mua/bán khi MFI cắt qua mức 20 hoặc 80, và giúp các nhà phân tích kỹ thuật ra quyết định giao dịch bằng cách xác định mức độ mạnh yếu của mua/bán.
        Tham số của MFI:
        MFI có một tham số chính:
        timeperiod: Khoảng thời gian sử dụng để tính toán MFI (ví dụ: 10, 20, 30 ngày) -> thường là 14

        OK for talib vaf pandas_ta
        '''
        if self.__chosen == 'ta':
            return self.ta.MFI(data['high'], data['low'], data['close'], data['volume'], timeperiod=timeperiod)
        elif self.__chosen == 'pandas_ta':
            return self.ta.mfi(data['high'], data['low'], data['close'], data['volume'], length=timeperiod)
        elif self.__chosen == 'multilib':
            sr1 = self.talib.MFI(data['high'], data['low'], data['close'], data['volume'], timeperiod=timeperiod)
            sr1.fillna(0, inplace=True)
            sr2 = self.pta.mfi(data['high'], data['low'], data['close'], data['volume'], length=timeperiod)
            sr2.fillna(0, inplace=True)
            result = (sr1 != sr2).astype(int)
            return str(result.sum()) + '/' + str(len(result))
    
    def MINUS_DI(self, data:pd.DataFrame, timeperiod=14):
        '''
        MINUS_DI (Minus Directional Indicator) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó được phát triển bởi J. Welles Wilder Jr. vào năm 1978 và được sử dụng để xác định xu hướng giảm của giá.
        MINUS_DI dựa trên giá và thời gian. Nó được hiển thị trên biểu đồ giá dưới dạng một dải (band) từ 0 đến 100. Giá trị của MINUS_DI thường dao động từ 0 đến 100. MINUS_DI có thể được sử dụng để xác định xu hướng giảm của giá, phát hiện các tín hiệu mua/bán khi MINUS_DI cắt qua mức 20 hoặc 25, và giúp các nhà phân tích kỹ thuật ra quyết định giao dịch bằng cách xác định mức độ mạnh yếu của xu hướng giảm.
        Tham số của MINUS_DI:
        MINUS_DI có một tham số chính:
        timeperiod: Khoảng thời gian sử dụng để tính toán MINUS_DI (ví dụ: 10, 20, 30 ngày) -> thường là 14

        Lưu ý: talib_pandas không có chỉ số này
        '''
        if self.__chosen == 'ta':
            return self.ta.MINUS_DI(data['high'], data['low'], data['close'], timeperiod=timeperiod)
        elif self.__chosen == 'pandas_ta':
            print('talib_pandas không có chỉ số này')
            # return self.ta.minus_di(data['high'], data['low'], data['close'], length=timeperiod)
        elif self.__chosen == 'multilib':
            print('talib_pandas không có chỉ số này')
            # sr1 = self.talib.MINUS_DI(data['high'], data['low'], data['close'], timeperiod=timeperiod)
            # sr1.fillna(0, inplace=True)
            # sr2 = self.pta.minus_di(data['high'], data['low'], data['close'], length=timeperiod)
            # sr2.fillna(0, inplace=True)
            # result = (sr1 != sr2).astype(int)
            # return str(result.sum()) + '/' + str(len(result))
    
    def __MINUS_DM(self, data:pd.DataFrame, timeperiod=14):
        '''
        MINUS_DM (Minus Directional Movement) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó được phát triển bởi J. Welles Wilder Jr. vào năm 1978 và được sử dụng để xác định động lượng giảm của giá.
        MINUS_DM dựa trên giá và thời gian. Nó được hiển thị trên biểu đồ giá dưới dạng một dải (band) hoặc một đường trung bình (line). MINUS_DM có thể được sử dụng để xác định động lượng giảm của giá, phát hiện các tín hiệu mua/bán khi MINUS_DM cắt qua mức 0, và giúp các nhà phân tích kỹ thuật ra quyết định giao dịch bằng cách xác định mức độ mạnh yếu của động lượng giảm.
        Tham số của MINUS_DM:
        MINUS_DM có một tham số chính:
        timeperiod: Khoảng thời gian sử dụng để tính toán MINUS_DM (ví dụ: 10, 20, 30 ngày) -> thường là 14

        Lưu ý: talib_pandas không có chỉ số này

        Tạm không dùng vì có ít thông tin về chỉ số này
        '''
        if self.__chosen == 'ta':
            return self.ta.MINUS_DM(data['high'], data['low'], timeperiod=timeperiod)
        elif self.__chosen == 'pandas_ta':
            print('talib_pandas không có chỉ số này')
            # return self.ta.minus_dm(data['high'], data['low'], length=timeperiod)
        elif self.__chosen == 'multilib':
            print('talib_pandas không có chỉ số này')
            # sr1 = self.talib.MINUS_DM(data['high'], data['low'], timeperiod=timeperiod)
            # sr1.fillna(0, inplace=True)
            # sr2 = self.pta.minus_dm(data['high'], data['low'], length=timeperiod)
            # sr2.fillna(0, inplace=True)
            # result = (sr1 != sr2).astype(int)
            # return str(result.sum()) + '/' + str(len(result))
    
    def __MOM(self, data:pd.DataFrame, timeperiod=10):
        '''
        MOM (Momentum) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó được sử dụng để xác định động lượng của giá.
        MOM dựa trên giá và thời gian. Nó được hiển thị trên biểu đồ giá dưới dạng một dải (band) hoặc một đường trung bình (line). MOM có thể được sử dụng để xác định động lượng của giá, phát hiện các tín hiệu mua/bán khi MOM cắt qua mức 0, và giúp các nhà phân tích kỹ thuật ra quyết định giao dịch bằng cách xác định mức độ mạnh yếu của động lượng.
        Tham số của MOM:
        MOM có một tham số chính:
        timeperiod: Khoảng thời gian sử dụng để tính toán MOM (ví dụ: 10, 20, 30 ngày) -> thường là 10

        Tạm không sử dụng do giao động vô định và có giá trị âm
        '''
        if self.__chosen == 'ta':
            return self.ta.MOM(data['close'], timeperiod=timeperiod)
        elif self.__chosen == 'pandas_ta':
            return self.ta.mom(data['close'], length=timeperiod)
        elif self.__chosen == 'multilib':
            sr1 = self.talib.MOM(data['close'], timeperiod=timeperiod)
            sr1.fillna(0, inplace=True)
            sr2 = self.pta.mom(data['close'], length=timeperiod)
            sr2.fillna(0, inplace=True)
            result = (sr1 != sr2).astype(int)
            return str(result.sum()) + '/' + str(len(result))
    
    def PLUS_DI(self, data:pd.DataFrame, timeperiod=14):
        '''
        PLUS_DI (Plus Directional Indicator) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó được phát triển bởi J. Welles Wilder Jr. vào năm 1978 và được sử dụng để xác định xu hướng tăng của giá.
        PLUS_DI dựa trên giá và thời gian. Nó được hiển thị trên biểu đồ giá dưới dạng một dải (band) từ 0 đến 100. Giá trị của PLUS_DI thường dao động từ 0 đến 100. PLUS_DI có thể được sử dụng để xác định xu hướng tăng của giá, phát hiện các tín hiệu mua/bán khi PLUS_DI cắt qua mức 20 hoặc 25, và giúp các nhà phân tích kỹ thuật ra quyết định giao dịch bằng cách xác định mức độ mạnh yếu của xu hướng tăng.
        Tham số của PLUS_DI:
        PLUS_DI có một tham số chính:
        timeperiod: Khoảng thời gian sử dụng để tính toán PLUS_DI (ví dụ: 10, 20, 30 ngày) -> thường là 14

        Lưu ý: talib_pandas không có chỉ số này
        '''
        if self.__chosen == 'ta':
            return self.ta.PLUS_DI(data['high'], data['low'], data['close'], timeperiod=timeperiod)
        elif self.__chosen == 'pandas_ta':
            print('talib_pandas không có chỉ số này')
            # return self.ta.plus_di(data['high'], data['low'], data['close'], length=timeperiod)
        elif self.__chosen == 'multilib':
            print('talib_pandas không có chỉ số này')
            # sr1 = self.talib.PLUS_DI(data['high'], data['low'], data['close'], timeperiod=timeperiod)
            # sr1.fillna(0, inplace=True)
            # sr2 = self.pta.plus_di(data['high'], data['low'], data['close'], length=timeperiod)
            # sr2.fillna(0, inplace=True)
            # result = (sr1 != sr2).astype(int)
            # return str(result.sum()) + '/' + str(len(result))
    
    def __PLUS_DM(self, data:pd.DataFrame, timeperiod=14):
        '''
        PLUS_DM (Plus Directional Movement) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó được phát triển bởi J. Welles Wilder Jr. vào năm 1978 và được sử dụng để xác định động lượng tăng của giá.
        PLUS_DM dựa trên giá và thời gian. Nó được hiển thị trên biểu đồ giá dưới dạng một dải (band) hoặc một đường trung bình (line). PLUS_DM có thể được sử dụng để xác định động lượng tăng của giá, phát hiện các tín hiệu mua/bán khi PLUS_DM cắt qua mức 0, và giúp các nhà phân tích kỹ thuật ra quyết định giao dịch bằng cách xác định mức độ mạnh yếu của động lượng tăng.
        Tham số của PLUS_DM:
        PLUS_DM có một tham số chính:
        timeperiod: Khoảng thời gian sử dụng để tính toán PLUS_DM (ví dụ: 10, 20, 30 ngày) -> thường là 14

        Lưu ý: talib_pandas không có chỉ số này

        Không sử dụng vì ít thông tin về chỉ số này. Giá trị giao động vô định
        '''
        if self.__chosen == 'ta':
            return self.ta.PLUS_DM(data['high'], data['low'], timeperiod=timeperiod)
        elif self.__chosen == 'pandas_ta':
            print('talib_pandas không có chỉ số này')
            # return self.ta.plus_dm(data['high'], data['low'], length=timeperiod)
        elif self.__chosen == 'multilib':
            print('talib_pandas không có chỉ số này')
            # sr1 = self.talib.PLUS_DM(data['high'], data['low'], timeperiod=timeperiod)
            # sr1.fillna(0, inplace=True)
            # sr2 = self.pta.plus_dm(data['high'], data['low'], length=timeperiod)
            # sr2.fillna(0, inplace=True)
            # result = (sr1 != sr2).astype(int)
            # return str(result.sum()) + '/' + str(len(result))
    
    def __PPO(self, data:pd.DataFrame, fastperiod=12, slowperiod=26, matype=0):
        '''
        PPO (Percentage Price Oscillator) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó được sử dụng để xác định sự chuyển đổi và phân kỳ của giá dựa trên phần trăm giá trị giữa giá trị trung bình nhanh và giá trị trung bình chậm.
        PPO dựa trên giá và thời gian. Nó được hiển thị trên biểu đồ giá dưới dạng một dải (band) hoặc một đường trung bình (line). PPO có thể được sử dụng để xác định sự chuyển đổi và phân kỳ của giá, phát hiện các tín hiệu mua/bán khi PPO cắt qua mức 0, và giúp các nhà phân tích kỹ thuật ra quyết định giao dịch bằng cách xác định mức độ mạnh yếu của sự chuyển đổi và phân kỳ.
        Tham số của PPO:
        PPO có ba tham số chính:
        fastperiod: Khoảng thời gian sử dụng để tính toán giá trị trung bình nhanh (ví dụ: 10, 20, 30 ngày) -> thường là 12
        slowperiod: Khoảng thời gian sử dụng để tính toán giá trị trung bình chậm (ví dụ: 10, 20, 30 ngày) -> thường là 26
        matype: Phương pháp tính toán giá trung bình động (moving average) được sử dụng để tính toán giá trị trung bình. Có ba phương pháp chính:
            SMA (Simple Moving Average): Phương pháp trung bình động đơn giản.
            EMA (Exponential Moving Average): Phương pháp trung bình động hàm mũ.
            WMA (Weighted Moving Average): Phương pháp trung bình động có trọng số.
        
        Tạm không dùng vì giao động vô định
        '''
        if self.__chosen == 'ta':
            return self.ta.PPO(data['close'], fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)
        elif self.__chosen == 'pandas_ta':
            return self.ta.ppo(data['close'], fast=fastperiod, slow=slowperiod, signal=matype)
        elif self.__chosen == 'multilib':
            sr1 = self.talib.PPO(data['close'], fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)
            sr1.fillna(0, inplace=True)
            sr2 = self.pta.ppo(data['close'], fast=fastperiod, slow=slowperiod, signal=matype)
            sr2.fillna(0, inplace=True)
            result = (sr1 != sr2).astype(int)
            return str(result.sum()) + '/' + str(len(result))
        
    def __ROC(self, data:pd.DataFrame, timeperiod=10):
        '''
        ROC (Rate of Change) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó được sử dụng để xác định động lượng của giá dựa trên tỷ lệ phần trăm thay đổi giá trị giữa giá trị hiện tại và giá trị trong quá khứ.
        ROC dựa trên giá và thời gian. Nó được hiển thị trên biểu đồ giá dưới dạng một dải (band) hoặc một đường trung bình (line). ROC có thể được sử dụng để xác định động lượng của giá, phát hiện các tín hiệu mua/bán khi ROC cắt qua mức 0, và giúp các nhà phân tích kỹ thuật ra quyết định giao dịch bằng cách xác định mức độ mạnh yếu của động lượng.
        Tham số của ROC:
        ROC có một tham số chính:
        timeperiod: Khoảng thời gian sử dụng để tính toán ROC (ví dụ: 10, 20, 30 ngày) -> thường là 10

        Tạm không dùng vì giao động vô định
        '''
        if self.__chosen == 'ta':
            return self.ta.ROC(data['close'], timeperiod=timeperiod)
        elif self.__chosen == 'pandas_ta':
            return self.ta.roc(data['close'], length=timeperiod)
        elif self.__chosen == 'multilib':
            sr1 = self.talib.ROC(data['close'], timeperiod=timeperiod)
            sr1.fillna(0, inplace=True)
            sr2 = self.pta.roc(data['close'], length=timeperiod)
            sr2.fillna(0, inplace=True)
            result = (sr1 != sr2).astype(int)
            return str(result.sum()) + '/' + str(len(result))
    
    def __ROCP(self, data:pd.DataFrame, timeperiod=10):
        '''
        ROCP (Rate of Change Percentage) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó được sử dụng để xác định động lượng của giá dựa trên tỷ lệ phần trăm thay đổi giá trị giữa giá trị hiện tại và giá trị trong quá khứ.
        ROCP dựa trên giá và thời gian. Nó được hiển thị trên biểu đồ giá dưới dạng một dải (band) hoặc một đường trung bình (line). ROCP có thể được sử dụng để xác định động lượng của giá, phát hiện các tín hiệu mua/bán khi ROCP cắt qua mức 0, và giúp các nhà phân tích kỹ thuật ra quyết định giao dịch bằng cách xác định mức độ mạnh yếu của động lượng.
        Tham số của ROCP:
        ROCP có một tham số chính:
        timeperiod: Khoảng thời gian sử dụng để tính toán ROCP (ví dụ: 10, 20, 30 ngày) -> thường là 10

        Lưu ý: talib_pandas không có chỉ số này

        Tạm không dùng vì giao động vô định
        '''
        if self.__chosen == 'ta':
            return self.ta.ROCP(data['close'], timeperiod=timeperiod)
        elif self.__chosen == 'pandas_ta':
            print('talib_pandas không có chỉ số này')
            # return self.ta.rocp(data['close'], length=timeperiod)
        elif self.__chosen == 'multilib':
            print('talib_pandas không có chỉ số này')
            # sr1 = self.talib.ROCP(data['close'], timeperiod=timeperiod)
            # sr1.fillna(0, inplace=True)
            # sr2 = self.pta.rocp(data['close'], length=timeperiod)
            # sr2.fillna(0, inplace=True)
            # result = (sr1 != sr2).astype(int)
            # return str(result.sum()) + '/' + str(len(result))
    
    def __ROCR(self, data:pd.DataFrame, timeperiod=10):
        '''
        ROCR (Rate of Change Ratio) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó được sử dụng để xác định động lượng của giá dựa trên tỷ lệ phần trăm thay đổi giá trị giữa giá trị hiện tại và giá trị trong quá khứ.
        ROCR dựa trên giá và thời gian. Nó được hiển thị trên biểu đồ giá dưới dạng một dải (band) hoặc một đường trung bình (line). ROCR có thể được sử dụng để xác định động lượng của giá, phát hiện các tín hiệu mua/bán khi ROCR cắt qua mức 0, và giúp các nhà phân tích kỹ thuật ra quyết định giao dịch bằng cách xác định mức độ mạnh yếu của động lượng.
        Tham số của ROCR:
        ROCR có một tham số chính:
        timeperiod: Khoảng thời gian sử dụng để tính toán ROCR (ví dụ: 10, 20, 30 ngày) -> thường là 10

        Lưu ý: talib_pandas không có chỉ số này

        Tạm không dùng vì giao động vô định
        '''
        if self.__chosen == 'ta':
            return self.ta.ROCR(data['close'], timeperiod=timeperiod)
        elif self.__chosen == 'pandas_ta':
            print('talib_pandas không có chỉ số này')
            # return self.ta.rocr(data['close'], length=timeperiod)
        elif self.__chosen == 'multilib':
            print('talib_pandas không có chỉ số này')
            # sr1 = self.talib.ROCR(data['close'], timeperiod=timeperiod)
            # sr1.fillna(0, inplace=True)
            # sr2 = self.pta.rocr(data['close'], length=timeperiod)
            # sr2.fillna(0, inplace=True)
            # result = (sr1 != sr2).astype(int)
            # return str(result.sum()) + '/' + str(len(result))
    
    def __ROCR100(self, data:pd.DataFrame, timeperiod=10):
        '''
        ROCR100 (Rate of Change Ratio 100) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó được sử dụng để xác định động lượng của giá dựa trên tỷ lệ phần trăm thay đổi giá trị giữa giá trị hiện tại và giá trị trong quá khứ.
        ROCR100 dựa trên giá và thời gian. Nó được hiển thị trên biểu đồ giá dưới dạng một dải (band) hoặc một đường trung bình (line). ROCR100 có thể được sử dụng để xác định động lượng của giá, phát hiện các tín hiệu mua/bán khi ROCR100 cắt qua mức 0, và giúp các nhà phân tích kỹ thuật ra quyết định giao dịch bằng cách xác định mức độ mạnh yếu của động lượng.
        Tham số của ROCR100:
        ROCR100 có một tham số chính:
        timeperiod: Khoảng thời gian sử dụng để tính toán ROCR100 (ví dụ: 10, 20, 30 ngày) -> thường là 10

        Lưu ý: talib_pandas không có chỉ số này

        Tạm không dùng vì giao động vô định
        '''
        if self.__chosen == 'ta':
            return self.ta.ROCR100(data['close'], timeperiod=timeperiod)
        elif self.__chosen == 'pandas_ta':
            print('talib_pandas không có chỉ số này')
            # return self.ta.rocr100(data['close'], length=timeperiod)
        elif self.__chosen == 'multilib':
            print('talib_pandas không có chỉ số này')
            # sr1 = self.talib.ROCR100(data['close'], timeperiod=timeperiod)
            # sr1.fillna(0, inplace=True)
            # sr2 = self.pta.rocr100(data['close'], length=timeperiod)
            # sr2.fillna(0, inplace=True)
            # result = (sr1 != sr2).astype(int)
            # return str(result.sum()) + '/' + str(len(result))
    
    def RSI(self, data:pd.DataFrame, timeperiod=14):
        '''
        RSI (Relative Strength Index) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó được phát triển bởi J. Welles Wilder Jr. vào năm 1978 và được sử dụng để xác định sức mạnh của mua/bán trên thị trường.
        RSI dựa trên giá và thời gian. Nó được hiển thị trên biểu đồ giá dưới dạng một dải (band) từ 0 đến 100. Giá trị của RSI thường dao động từ 0 đến 100. RSI có thể được sử dụng để xác định sức mạnh của mua/bán trên thị trường, phát hiện các tín hiệu mua/bán khi RSI cắt qua mức 30 hoặc 70, và giúp các nhà phân tích kỹ thuật ra quyết định giao dịch bằng cách xác định mức độ mạnh yếu của mua/bán.
        Tham số của RSI:
        RSI có một tham số chính:
        timeperiod: Khoảng thời gian sử dụng để tính toán RSI (ví dụ: 10, 20, 30 ngày) -> thường là 14

        OK for talib và pandas_ta
        '''
        if self.__chosen == 'ta':
            return self.ta.RSI(data['close'], timeperiod=timeperiod)
        elif self.__chosen == 'pandas_ta':
            return self.ta.rsi(data['close'], length=timeperiod)
        elif self.__chosen == 'multilib':
            sr1 = self.talib.RSI(data['close'], timeperiod=timeperiod)
            sr1.fillna(0, inplace=True)
            sr2 = self.pta.rsi(data['close'], length=timeperiod)
            sr2.fillna(0, inplace=True)
            result = (sr1 != sr2).astype(int)
            return str(result.sum()) + '/' + str(len(result))
    
    def __STOCH(self, data:pd.DataFrame, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0):
        '''
        STOCH (Stochastic Oscillator) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó được phát triển bởi George Lane vào năm 1950 và được sử dụng để xác định sự quá mua và sự quá bán của giá.
        STOCH dựa trên giá và thời gian. Nó được hiển thị trên biểu đồ giá dưới dạng một dải (band) từ 0 đến 100. STOCH có thể được sử dụng để xác định sự quá mua và sự quá bán của giá, phát hiện các tín hiệu mua/bán khi %K cắt qua %D, và giúp các nhà phân tích kỹ thuật ra quyết định giao dịch bằng cách xác định mức độ mạnh yếu của sự quá mua và sự quá bán.
        Tham số của STOCH:
        STOCH có năm tham số chính:
        fastk_period: Khoảng thời gian sử dụng để tính toán %K (ví dụ: 10, 20, 30 ngày) -> thường là 5
        slowk_period: Khoảng thời gian sử dụng để tính toán %D (ví dụ: 10, 20, 30 ngày) -> thường là 3
        slowk_matype: Phương pháp tính toán giá trung bình động (moving average) được sử dụng để tính toán %D. Có ba phương pháp chính:
            SMA (Simple Moving Average): Phương pháp trung bình động đơn giản.
            EMA (Exponential Moving Average): Phương pháp trung bình động hàm mũ.
            WMA (Weighted Moving Average): Phương pháp tru ng bình động có trọng số.
        slowd_period: Khoảng thời gian sử dụng để tính toán %D (ví dụ: 10, 20, 30 ngày) -> thường là 3
        slowd_matype: Phương pháp tính toán giá trung bình động (moving average) được sử dụng để tính toán %D. Có ba phương pháp chính:
            SMA (Simple Moving Average): Phương pháp trung bình động đơn giản.
            EMA (Exponential Moving Average): Phương pháp trung bình động hàm mũ.
            WMA (Weighted Moving Average): Phương pháp trung bình động có trọng số.
        
        Lưu ý: trả về 2 chỉ số: slowk, slowd

        Tạm không dùng vì giao động vô định
        '''
        if self.__chosen == 'ta':
            return self.ta.STOCH(data['high'], data['low'], data['close'], fastk_period=fastk_period, slowk_period=slowk_period, slowk_matype=slowk_matype, slowd_period=slowd_period, slowd_matype=slowd_matype)
        elif self.__chosen == 'pandas_ta':
            return self.ta.stoch(data['high'], data['low'], data['close'], fastk=fastk_period, slowk=slowk_period, slowd=slowd_period)
        elif self.__chosen == 'multilib':
            sr1 = self.talib.STOCH(data['high'], data['low'], data['close'], fastk_period=fastk_period, slowk_period=slowk_period, slowk_matype=slowk_matype, slowd_period=slowd_period, slowd_matype=slowd_matype)[0]
            sr1.fillna(0, inplace=True)
            sr2 = self.pta.stoch(data['high'], data['low'], data['close'], fastk=fastk_period, slowk=slowk_period, slowd=slowd_period)
            sr2.fillna(0, inplace=True)
            result = (sr1 != sr2).astype(int)
            return str(result.sum()) + '/' + str(len(result))
    
    def __STOCHF(self, data:pd.DataFrame, fastk_period=5, fastd_period=3, fastd_matype=0):
        '''
        STOCHF (Stochastic Fast) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó được phát triển bởi George Lane vào năm 1950 và được sử dụng để xác định sự quá mua và sự quá bán của giá.
        STOCHF dựa trên giá và thời gian. Nó được hiển thị trên biểu đồ giá dưới dạng một dải (band) từ 0 đến 100. STOCHF có thể được sử dụng để xác định sự quá mua và sự quá bán của giá, phát hiện các tín hiệu mua/bán khi %K cắt qua %D, và giúp các nhà phân tích kỹ thuật ra quyết định giao dịch bằng cách xác định mức độ mạnh yếu của sự quá mua và sự quá bán.
        Tham số của STOCHF:
        STOCHF có ba tham số chính:
        fastk_period: Khoảng thời gian sử dụng để tính toán %K (ví dụ: 10, 20, 30 ngày) -> thường là 5
        fastd_period: Khoảng thời gian sử dụng để tính toán %D (ví dụ: 10, 20, 30 ngày) -> thường là 3
        fastd_matype: Phương pháp tính toán giá trung bình động (moving average) được sử dụng để tính toán %D. Có ba phương pháp chính:
            SMA (Simple Moving Average): Phương pháp trung bình động đơn giản.
            EMA (Exponential Moving Average): Phương pháp trung bình động hàm mũ.
            WMA (Weighted Moving Average): Phương pháp trung bình động có trọng số.
        
        Lưu ý: pandas_ta không có hàm này.

        Trả về 2 chỉ số: fastk, fastd

        Tạm không dùng vì giao động vô định
        '''
        if self.__chosen == 'ta':
            return self.ta.STOCHF(data['high'], data['low'], data['close'], fastk_period=fastk_period, fastd_period=fastd_period, fastd_matype=fastd_matype)
        elif self.__chosen == 'pandas_ta':
            print('pandas_ta không có hàm này')
            # return self.ta.stochf(data['high'], data['low'], data['close'], fastk=fastk_period, fastd=fastd_period)
        elif self.__chosen == 'multilib':
            print('pandas_ta không có hàm này')
            # sr1 = self.talib.STOCHF(data['high'], data['low'], data['close'], fastk_period=fastk_period, fastd_period=fastd_period, fastd_matype=fastd_matype)[0]
            # sr1.fillna(0, inplace=True)
            # sr2 = self.pta.stochf(data['high'], data['low'], data['close'], fastk=fastk_period, fastd=fastd_period)
            # sr2.fillna(0, inplace=True)
            # result = (sr1 != sr2).astype(int)
            # return str(result.sum()) + '/' + str(len(result))
    
    def __STOCHRSI(self, data:pd.DataFrame, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0):
        '''
        STOCHRSI (Stochastic Relative Strength Index) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó được phát triển bởi George Lane vào năm 1950 và được sử dụng để xác định sự quá mua và sự quá bán của giá.
        STOCHRSI dựa trên giá và thời gian. Nó được hiển thị trên biểu đồ giá dưới dạng một dải (band) từ 0 đến 100. STOCHRSI có thể được sử dụng để xác định sự quá mua và sự quá bán của giá, phát hiện các tín hiệu mua/bán khi %K cắt qua %D, và giúp các nhà phân tích kỹ thuật ra quyết định giao dịch bằng cách xác định mức độ mạnh yếu của sự quá mua và sự quá bán.
        Tham số của STOCHRSI:
        STOCHRSI có bốn tham số chính:
        timeperiod: Khoảng thời gian sử dụng để tính toán RSI (ví dụ: 10, 20, 30 ngày) -> thường là 14
        fastk_period: Khoảng thời gian sử dụng để tính toán %K (ví dụ: 10, 20, 30 ngày) -> thường là 5
        fastd_period: Khoảng thời gian sử dụng để tính toán %D (ví dụ: 10, 20, 30 ngày) -> thường là 3
        fastd_matype: Phương pháp tính toán giá trung bình động (moving average) được sử dụng để tí nh toán %D. Có ba phương pháp chính:
            SMA (Simple Moving Average): Phương pháp trung bình động đơn giản.
            EMA (Exponential Moving Average): Phương pháp trung bình động hàm mũ.
            WMA (Weighted Moving Average): Phương pháp trung bình động có trọng số.
        
        Trả về: fastk, fastd

        Tạm không dùng vì giao động vô định
        '''
        if self.__chosen == 'ta':
            return self.ta.STOCHRSI(data['close'], timeperiod=timeperiod, fastk_period=fastk_period, fastd_period=fastd_period, fastd_matype=fastd_matype)
        elif self.__chosen == 'pandas_ta':
            return self.ta.stochrsi(data['close'], length=timeperiod, fastk=fastk_period, fastd=fastd_period)
        elif self.__chosen == 'multilib':
            sr1 = self.talib.STOCHRSI(data['close'], timeperiod=timeperiod, fastk_period=fastk_period, fastd_period=fastd_period, fastd_matype=fastd_matype)[0]
            sr1.fillna(0, inplace=True)
            sr2 = self.pta.stochrsi(data['close'], length=timeperiod, fastk=fastk_period, fastd=fastd_period)
            sr2.fillna(0, inplace=True)
            result = (sr1 != sr2).astype(int)
            return str(result.sum()) + '/' + str(len(result))
    
    def __TRIX(self, data:pd.DataFrame, timeperiod=30):
        '''
        TRIX (Triple Exponential Moving Average) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó được phát triển bởi Jack Hutson vào năm 1980 và được sử dụng để xác định xu hướng của giá.
        TRIX dựa trên giá và thời gian. Nó được hiển thị trên biểu đồ giá dưới dạng một dải (band) hoặc một đường trung bình (line). TRIX có thể được sử dụng để xác định xu hướng của giá, phát hiện các tín hiệu mua/bán khi TRIX cắt qua mức 0, và giúp các nhà phân tích kỹ thuật ra quyết định giao dịch bằng cách xác định mức độ mạnh yếu của xu hướng.
        Tham số của TRIX:
        TRIX có một tham số chính:
        timeperiod: Khoảng thời gian sử dụng để tính toán TRIX (ví dụ: 10, 20, 30 ngày) -> thường là 30

        Tạm không dùng vì giao động vô định
        '''
        if self.__chosen == 'ta':
            return self.ta.TRIX(data['close'], timeperiod=timeperiod)
        elif self.__chosen == 'pandas_ta':
            return self.ta.trix(data['close'], length=timeperiod)
        elif self.__chosen == 'multilib':
            sr1 = self.talib.TRIX(data['close'], timeperiod=timeperiod)
            sr1.fillna(0, inplace=True)
            sr2 = self.pta.trix(data['close'], length=timeperiod)
            sr2.fillna(0, inplace=True)
            result = (sr1 != sr2).astype(int)
            return str(result.sum()) + '/' + str(len(result))
    
    def ULTOSC(self, data:pd.DataFrame, timeperiod1=7, timeperiod2=14, timeperiod3=28):
        '''
        ULTOSC (Ultimate Oscillator) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó được phát triển bởi Larry Williams vào năm 1976 và được sử dụng để xác định sự quá mua và sự quá bán của giá.
        ULTOSC dựa trên giá và thời gian. Nó được hiển thị trên biểu đồ giá dưới dạng một dải (band) từ 0 đến 100. ULTOSC có thể được sử dụng để xác định sự quá mua và sự quá bán của giá, phát hiện các tín hiệu mua/bán khi ULTOSC cắt qua mức 30 hoặc 70, và giúp các nhà phân tích kỹ thuật ra quyết định giao dịch bằng cách xác định mức độ mạnh yếu của sự quá mua và sự quá bán.
        Tham số của ULTOSC:
        ULTOSC có ba tham số chính:
        timeperiod1: Khoảng thời gian sử dụng để tính toán giá trị trung bình nhanh (ví dụ: 10, 20, 30 ngày) -> thường là 7
        timeperiod2: Khoảng thời gian sử dụng để tính toán giá trị trung bình nhanh (ví dụ: 10, 20, 30 ngày) -> thường là 14
        timeperiod3: Khoảng thời gian sử dụng để tính toán giá trị trung bình nhanh (ví dụ: 10, 20, 30 ngày) -> thường là 28

        Lưu ý: pandas_ta không có chỉ số này
        '''
        if self.__chosen == 'ta':
            return self.ta.ULTOSC(data['high'], data['low'], data['close'], timeperiod1=timeperiod1, timeperiod2=timeperiod2, timeperiod3=timeperiod3)
        elif self.__chosen == 'pandas_ta':
            print('pandas_ta không có chỉ số này')
            # return self.ta.ultosc(data['high'], data['low'], data['close'], length1=timeperiod1, length2=timeperiod2, length3=timeperiod3)
        elif self.__chosen == 'multilib':
            print('pandas_ta không có chỉ số này')
            # sr1 = self.talib.ULTOSC(data['high'], data['low'], data['close'], timeperiod1=timeperiod1, timeperiod2=timeperiod2, timeperiod3=timeperiod3)
            # sr1.fillna(0, inplace=True)
            # sr2 = self.pta.ultosc(data['high'], data['low'], data['close'], length1=timeperiod1, length2=timeperiod2, length3=timeperiod3)
            # sr2.fillna(0, inplace=True)
            # result = (sr1 != sr2).astype(int)
            # return str(result.sum()) + '/' + str(len(result))
    
    def WILLR(self, data:pd.DataFrame, timeperiod=14):
        '''
        WILLR (Williams %R) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó được phát triển bởi Larry Williams vào năm 1973 và được sử dụng để xác định sự quá mua và sự quá bán của giá.
        WILLR dựa trên giá và thời gian. Nó được hiển thị trên biểu đồ giá dưới dạng một dải (band) từ -100 đến 0. WILLR có thể được sử dụng để xác định sự quá mua và sự quá bán của giá, phát hiện các tín hiệu mua/bán khi WILLR cắt qua mức -20 hoặc -80, và giúp các nhà phân tích kỹ thuật ra quyết định giao dịch bằng cách xác định mức độ mạnh yếu của sự quá mua và sự quá bán.
        Tham số của WILLR:
        WILLR có một tham số chính:
        timeperiod: Khoảng thời gian sử dụng để tính toán WILLR (ví dụ: 10, 20, 30 ngày) -> thường là 14

        OK for talib và pandas_ta

        Tuy vậy, vì giao động trong khoảng -100 đến 0 nên cần chuyển đôi
        '''
        if self.__chosen == 'ta':
            sr = self.ta.WILLR(data['high'], data['low'], data['close'], timeperiod=timeperiod)
            sr = sr.apply(self.normalize, args=(-100, 0))
            return sr
        elif self.__chosen == 'pandas_ta':
            sr = self.ta.willr(data['high'], data['low'], data['close'], length=timeperiod)
            sr = sr.apply(self.normalize, args=(-100, 0))
            return sr
        elif self.__chosen == 'multilib':
            sr1 = self.talib.WILLR(data['high'], data['low'], data['close'], timeperiod=timeperiod)
            sr1.fillna(0, inplace=True)
            sr2 = self.pta.willr(data['high'], data['low'], data['close'], length=timeperiod)
            sr2.fillna(0, inplace=True)
            result = (sr1 != sr2).astype(int)
            return str(result.sum()) + '/' + str(len(result))
    
    #################Volume Indicator Functions####################

    
    def AD(self, data:pd.DataFrame):
        '''
        AD (Accumulation/Distribution Line) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó được phát triển bởi Marc Chaikin vào năm 1982 và được sử dụng để xác định áp lực mua/bán trên thị trường.
        AD dựa trên giá và khối lượng giao dịch. Nó được hiển thị dưới dạng một đường trung bình (line). AD có thể được sử dụng để xác định áp lực mua/bán trên thị trường, phát hiện các tín hiệu mua/bán khi AD cắt qua mức 0, và giúp các nhà phân tích kỹ thuật ra quyết định giao dịch bằng cách xác định mức độ mạnh yếu của áp lực mua/bán.
        Tham số của AD:
        AD không có tham số chính.

        OK for talib và pandas_ta. Nếu dùng với class MakeSample, nằm trong nhóm OtherScale
        '''
        if self.__chosen == 'ta':
            return self.ta.AD(data['high'], data['low'], data['close'], data['volume'])
        elif self.__chosen == 'pandas_ta':
            return self.ta.ad(data['high'], data['low'], data['close'], data['volume'])
        elif self.__chosen == 'multilib':
            sr1 = self.talib.AD(data['high'], data['low'], data['close'], data['volume'])
            sr1.fillna(0, inplace=True)
            sr2 = self.pta.ad(data['high'], data['low'], data['close'], data['volume'])
            sr2.fillna(0, inplace=True)
            result = (sr1 != sr2).astype(int)
            return str(result.sum()) + '/' + str(len(result))
    
    def ADOSC(self, data:pd.DataFrame, fastperiod=3, slowperiod=10):
        '''
        ADOSC (Accumulation/Distribution Oscillator) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó được phát triển bởi Marc Chaikin vào năm 1982 và được sử dụng để xác định áp lực mua/bán trên thị trường.
        ADOSC dựa trên giá và khối lượng giao dịch. Nó được hiển thị dưới dạng một dải (band) hoặc một đường trung bình (line). ADOSC có thể được sử dụng để xác định áp lực mua/bán trên thị trường, phát hiện các tín hiệu mua/bán khi ADOSC cắt qua mức 0, và giúp các nhà phân tích kỹ thuật ra quyết định giao dịch bằng cách xác định mức độ mạnh yếu của áp lực mua/bán.
        Tham số của ADOSC:
        ADOSC có hai tham số chính:
        fastperiod: Khoảng thời gian sử dụng để tính toán giá trị trung bình nhanh (ví dụ: 10, 20, 30 ngày) -> thường là 3
        slowperiod: Khoảng thời gian sử dụng để tính toán giá trị trung bình chậm (ví dụ: 10, 20, 30 ngày) -> thường là 10

        OK for talib và pandas_ta. Nếu dùng với class MakeSample, nằm trong nhóm OtherScale.
        Ưu tiên dùng chỉ số này vì độ nhạy
        '''
        if self.__chosen == 'ta':
            return self.ta.ADOSC(data['high'], data['low'], data['close'], data['volume'], fastperiod=fastperiod, slowperiod=slowperiod)
        elif self.__chosen == 'pandas_ta':
            return self.ta.adosc(data['high'], data['low'], data['close'], data['volume'], fast=fastperiod, slow=slowperiod)
        elif self.__chosen == 'multilib':
            sr1 = self.talib.ADOSC(data['high'], data['low'], data['close'], data['volume'], fastperiod=fastperiod, slowperiod=slowperiod)
            sr1.fillna(0, inplace=True)
            sr2 = self.pta.adosc(data['high'], data['low'], data['close'], data['volume'], fast=fastperiod, slow=slowperiod)
            sr2.fillna(0, inplace=True)
            result = (sr1 != sr2).astype(int)
            return str(result.sum()) + '/' + str(len(result))
    
    def OBV(self, data:pd.DataFrame):
        '''
        OBV (On Balance Volume) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó được phát triển bởi Joe Granville vào năm 1963 và được sử dụng để xác định áp lực mua/bán trên thị trường.
        OBV dựa trên giá và khối lượng giao dịch. Nó được hiển thị dưới dạng một đường trung bình (line). OBV có thể được sử dụng để xác định áp lực mua/bán trên thị trường, phát hiện các tín hiệu mua/bán khi OBV cắt qua mức 0, và giúp các nhà phân tích kỹ thuật ra quyết định giao dịch bằng cách xác định mức độ mạnh yếu của áp lực mua/bán.
        Tham số của OBV:
        OBV không có tham số chính.
        OK for talib và pandas_ta. Nếu dùng với class MakeSample, nằm trong nhóm OtherScale
        Chỉ số này kém nhạy, có thể không nên dùng
        '''
        if self.__chosen == 'ta':
            return self.ta.OBV(data['close'], data['volume'])
        elif self.__chosen == 'pandas_ta':
            return self.ta.obv(data['close'], data['volume'])
        elif self.__chosen == 'multilib':
            sr1 = self.talib.OBV(data['close'], data['volume'])
            sr1.fillna(0, inplace=True)
            sr2 = self.pta.obv(data['close'], data['volume'])
            sr2.fillna(0, inplace=True)
            result = (sr1 != sr2).astype(int)
            return str(result.sum()) + '/' + str(len(result))

    
    #################Volatility Indicator Functions####################

    
    def ATR(self, data:pd.DataFrame, timeperiod=14):
        '''
        ATR (Average True Range) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó được phát triển bởi J. Welles Wilder Jr. vào năm 1978 và được sử dụng để xác định biên độ dao động của giá.
        ATR dựa trên giá và thời gian. Nó được hiển thị dưới dạng một dải (band) hoặc một đường trung bình (line). ATR có thể được sử dụng để xác định biên độ dao động của giá, phát hiện các tín hiệu mua/bán khi ATR tăng hoặc giảm, và giúp các nhà phân tích kỹ thuật ra quyết định giao dịch bằng cách xác định mức độ mạnh yếu của biên độ dao động.
        Tham số của ATR:
        ATR có một tham số chính:
        timeperiod: Khoảng thời gian sử dụng để tính toán ATR (ví dụ: 10, 20, 30 ngày) -> thường là 14

        OK for talib và pandas_ta. Nếu dùng với class MakeSample, nằm trong nhóm OtherScale

        '''
        if self.__chosen == 'ta':
            return self.ta.ATR(data['high'], data['low'], data['close'], timeperiod=timeperiod)
        elif self.__chosen == 'pandas_ta':
            return self.ta.atr(data['high'], data['low'], data['close'], length=timeperiod)
        elif self.__chosen == 'multilib':
            sr1 = self.talib.ATR(data['high'], data['low'], data['close'], timeperiod=timeperiod)
            sr1.fillna(0, inplace=True)
            sr2 = self.pta.atr(data['high'], data['low'], data['close'], length=timeperiod)
            sr2.fillna(0, inplace=True)
            result = (sr1 != sr2).astype(int)
            return str(result.sum()) + '/' + str(len(result))
    
    def NATR(self, data:pd.DataFrame, timeperiod=14):
        '''
        NATR (Normalized Average True Range) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó được phát triển bởi J. Welles Wilder Jr. vào năm 1978 và được sử dụng để xác định biên độ dao động của giá.
        NATR dựa trên giá và thời gian. Nó được hiển thị dưới dạng một dải (band) hoặc một đường trung bình (line). NATR có thể được sử dụng để xác định biên độ dao động của giá, phát hiện các tín hiệu mua/bán khi NATR tăng hoặc giảm, và giúp các nhà phân tích kỹ thuật ra quyết định giao dịch bằng cách xác định mức độ mạnh yếu của biên độ dao động.
        Tham số của NATR:
        NATR có một tham số chính:
        timeperiod: Khoảng thời gian sử dụng để tính toán NATR (ví dụ: 10, 20, 30 ngày) -> thường là 14

        OK for talib và pandas_ta. Nếu dùng với class MakeSample, nằm trong nhóm OtherScale
        Thấy không khác ATR mấy
        '''
        if self.__chosen == 'ta':
            return self.ta.NATR(data['high'], data['low'], data['close'], timeperiod=timeperiod)
        elif self.__chosen == 'pandas_ta':
            return self.ta.natr(data['high'], data['low'], data['close'], length=timeperiod)
        elif self.__chosen == 'multilib':
            sr1 = self.talib.NATR(data['high'], data['low'], data['close'], timeperiod=timeperiod)
            sr1.fillna(0, inplace=True)
            sr2 = self.pta.natr(data['high'], data['low'], data['close'], length=timeperiod)
            sr2.fillna(0, inplace=True)
            result = (sr1 != sr2).astype(int)
            return str(result.sum()) + '/' + str(len(result))
    
    def TRANGE(self, data:pd.DataFrame):
        '''
        TRANGE (True Range) là một chỉ báo kỹ thuật trong thư viện ta-lib. Nó được phát triển bởi J. Welles Wilder Jr. vào năm 1978 và được sử dụng để xác định biên độ dao động của giá.
        TRANGE dựa trên giá và thời gian. Nó được hiển thị dưới dạng một dải (band) hoặc một đường trung bình (line). TRANGE có thể được sử dụng để xác định biên độ dao động của giá, phát hiện các tín hiệu mua/bán khi TRANGE tăng hoặc giảm, và giúp các nhà phân tích kỹ thuật ra quyết định giao dịch bằng cách xác định mức độ mạnh yếu của biên độ dao động.
        Tham số của TRANGE:
        TRANGE không có tham số chính.

        Lưu ý: pandas_ta không có chỉ số này
        '''
        if self.__chosen == 'ta':
            return self.ta.TRANGE(data['high'], data['low'], data['close'])
        elif self.__chosen == 'pandas_ta':
            print('Lưu ý: pandas_ta không có chỉ số này')
            # return self.ta.trange(data['high'], data['low'], data['close'])
        elif self.__chosen == 'multilib':
            print('Lưu ý: pandas_ta không có chỉ số này')
            # sr1 = self.talib.TRANGE(data['high'], data['low'], data['close'])
            # sr1.fillna(0, inplace=True)
            # sr2 = self.pta.trange(data['high'], data['low'], data['close'])
            # sr2.fillna(0, inplace=True)
            # result = (sr1 != sr2).astype(int)
            # return str(result.sum()) + '/' + str(len(result))



def _neutro(val):
    return val

def _wrapfunction(funct, **kwargs):
    return funct(**kwargs)

def addIndicatorTemplate(data:pd.DataFrame): 
    mtalib = MyTalib()
    
    listIndi= [ 

           ['close', _neutro, {'val' : data['close']} ], #
           ['KAMA', mtalib.KAMA, {'data': data}],
           ['TEMA', mtalib.TEMA, {'data': data}],
       
           # monmentum:
      
           ['MINUS_DI', mtalib.MINUS_DI, {'data': data}],
           ['PLUS_DI', mtalib.PLUS_DI, {'data': data}],
           ['WILLR', mtalib.WILLR, {'data': data}],
           ['CMO', mtalib.CMO, {'data': data}],
     
           # volume

           # volatility
           ['TRANGE', mtalib.TRANGE, {'data': data}],

          ]
    
    data = data.copy() # copy này sẽ ngăn tác động lên data gốc
    
    for i in range(len(listIndi)):
        data[listIndi[i][0]] = _wrapfunction(listIndi[i][1], **listIndi[i][2])    
    
    return data, [itm[0] for itm in listIndi]       
    
def addIndicatorList(data:pd.DataFrame, listI:list):
    mtalib = MyTalib()
    listIndix= [ 

                ['close', _neutro, {'val' : data['close']} ],

                ['MA10', mtalib.MA, {'data' : data, 'timeperiod': 10} ],

                ['MA50', mtalib.MA, {'data': data, 'timeperiod': 50}],

                ['DEMA', mtalib.DEMA, {'data': data}],

                ['EMA12', mtalib.EMA, {'data': data, 'timeperiod': 12}],

                ['EMA26', mtalib.EMA, {'data': data, 'timeperiod': 26}],

                ['EMA9', mtalib.EMA, {'data': data, 'timeperiod': 9}],

                ['KAMA', mtalib.KAMA, {'data': data}],

                ['MIDPOINT', mtalib.MIDPOINT, {'data': data}],

                ['MIDPRICE', mtalib.MIDPRICE, {'data': data}],

                ['SAR', mtalib.SAR, {'data': data}],

                ['T3', mtalib.T3, {'data': data}],

                ['TEMA', mtalib.TEMA, {'data': data}],

                ['TRIMA', mtalib.TRIMA, {'data': data}],

                ['WMA', mtalib.WMA, {'data': data}],


                #--- momentum-----

                ['ADX', mtalib.ADX, {'data': data}],

                ['ADXR', mtalib.ADXR, {'data': data}],

                # ['AROON', mtalib.AROON, {'data': datax}],

                ['CMO', mtalib.CMO, {'data': data}],

                ['DX', mtalib.DX, {'data': data}],

                ['MFI', mtalib.MFI, {'data': data}],

                ['MINUS_DI', mtalib.MINUS_DI, {'data': data}],

                ['PLUS_DI', mtalib.PLUS_DI, {'data': data}],
                
                ['RSI', mtalib.RSI, {'data': data, 'timeperiod': 14}],

                ['ULTOSC', mtalib.ULTOSC, {'data': data}],

                ['WILLR', mtalib.WILLR, {'data': data}],

                #--- Volume-----

                ['ADOSC', mtalib.ADOSC, {'data': data}],

                ['AD', mtalib.AD, {'data': data}],

                #--- Volatility-----

                ['ATR', mtalib.ATR, {'data': data}],

                ['NATR', mtalib.NATR, {'data': data}],

                ['TRANGE', mtalib.TRANGE, {'data': data}],

            ]
    
    data = data.copy()
    errlist = []
    for i in listI:
        err = True
        for j in range(len(listIndix)):
            if str.upper(i) == str.upper(listIndix[j][0]):
                data[listIndix[j][0]] = _wrapfunction(listIndix[j][1], **listIndix[j][2])
                err = False
                break
        
        if err: errlist.append(i)
    
    if len(errlist) > 0:
        raise ValueError(f'Some typo error in the name of indicator: {errlist}')
 
    return data

