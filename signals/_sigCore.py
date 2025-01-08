import pandas as pd
import functions.myredis as rf

class sigcore:
    def __init__(self, symbol:str, tf:str, market:int, testing:bool=False):
        '''
        testing = True : chạy ở chế độ test thì findsig trả về toàn bộ df để kiểm tra

        testing này không phải testmode khi chạy từ file main chính, mà là test từng module detect riêng lẻ
        ''' 
        self.name = ''
        self.para = {} # tham số chạy detect mà bên ngoài có thể truyền vào bằng khai báo hoặc qua hàm findsig
        self.innerPara = {} # tham số chạy detect mà bên trong class sử dụng
        self.symbol = symbol
        self.tf = tf
        self.market = market
        self.testing = testing
        # lưu ý đây không phải là sig và signal of toàn bộ df, chỉ là kết qua từ hàm findsig
        self.dfsigs = pd.DataFrame()
        self.dfsignal = pd.DataFrame()
        # giới hạn số dòng ghi vào redis
        self.limit = 5
       
    #---- hàm chuẩn hoá kiểu dữ liệu timestamp--------------------------------------
    def _normalize_df(self, dfx:pd.DataFrame):
        '''
        Chuẩn hoá kiểu dữ liệu timestamp
        '''
        # tránh kiểu dữ liệu int bị chuyển thành float sẽ không thấy nhau khi làm key trong redis
        dfx['timestamp'] = dfx['timestamp'].astype('Int64')
        return dfx
    
    #---- hàm ghi vào redis---------------------------------------------------------
    def save_signal_to_redis(self):
        '''
        Ghi tín hiệu vào redis
        '''
        if self.dfsigs.empty or self.dfsignal.empty:
            return None

        # nếu đã tồn tại hash sig thì chỉ ghi giới hạn số dòng mới
        if rf.check_exist_df_sig(self.symbol, self.name):
            # ghi lại đường, dấu hiệu, model có giá trị để thể hiện trên chart
            rf.add_df_sig(self.dfsigs.tail(self.limit), self.symbol, self.name)
            # ghi lại tín hiệu xuất hiện vị thế
            rf.add_df_signal(self.dfsignal.tail(self.limit), self.symbol, self.name)
        # chưa tồn tại thì ghi toàn bộ
        else:
            rf.add_df_sig(self.dfsigs, self.symbol, self.name)
            rf.add_df_signal(self.dfsignal, self.symbol, self.name)

        # get value in column signal of last row
        return self.dfsignal.iloc[-1]['signal_'+self.name]
    
    #----khởi tạo ban đầu, có thể cần hoặc không cần viết lại ở lớp dẫn xuất-------
    def firstInit(self, hashkeystatus:str=''):
        pass

    #----khởi tạo mỗi lần findsig, có thể cần hoặc không cần viết lại ở lớp dẫn xuất-------
    def secondInit(self, hashkeystatus:str=''):
        pass

    #----Thực thi sau khi findsig() -------
    def thirdJob(self, hashkeystatus:str='', setDelay:bool=True):
        pass

    #----tìm tín hiệu, cần viết lại ở các class dẫn xuất-------
    def findsig(self, dfsub:pd.DataFrame, para:dict={}, hashkeystatus:str=''):
        if para != {}:
            self.para = para
        #---- xử lý tìm tín hiệu ở đây----
        #... dfsub
        # --------------------------------
        dfsub = self._normalize_df(dfsub)
        self.dfsigs = dfsub[['timestamp', 'sig_'+self.name]]
        self.dfsignal = dfsub[['timestamp', 'signal_'+self.name]]    
       
        raise NotImplementedError("Subclass must implement abstract method")
        # chỗ trả về này để phục vụ backtest, còn chạy realtime thì không cần trả về
        if self.testing:
            return self.dfsigs, self.dfsignal, dfsub
        else:
            return self.dfsigs, self.dfsignal
        
     