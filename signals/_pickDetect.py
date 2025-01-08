import pandas as pd
import os

def _checkDetectBy(_detectby, testing):
    listModules = initList(makeNew=False)
    if _detectby in listModules.keys():
        if testing:
            return _detectby
        else:
            if listModules[_detectby][0] == 1:
                return _detectby
            else:
                return ''
    return ''

def getListModulesAuto(makeNew = True):
    listModules = initList(makeNew)
    listModulesAuto = []
    for k, v in listModules.items():
        if v[0] ==1 & v[1] == 1:
            listModulesAuto.append(k)
    return listModulesAuto

def getListModulesManual(makeNew = False):
    listModules = initList(makeNew)
    listModulesManual = []
    for k, v in listModules.items():
        if v[0] ==1 & v[2] == 1:
            listModulesManual.append(k)
    return listModulesManual


def initList(makeNew):
    if makeNew == False: 
        if os.path.exists('signals/_listModules.xlsx') == False: makeNew = True

    if makeNew == False:
        df = pd.read_excel('signals/_listModules.xlsx', index_col=0)
        df.columns = [0, 1, 2]
        listModules = df.to_dict(orient='index')
    else:
        # lưu ý: đặt tên các module không được giống hoặc nằm trong nhau như Modabc và Modabcdef
        if os.path.exists('signals/modules.json'):
            listModules = pd.read_json('signals/modules.json').to_dict()
        else:
            listModules = {
                        # vị trí trong listModules: [1: có thể chạy, có quyền chạy rồi mới xét các vị trí tiếp
                        #                            1: có thể chạy tự động ở runRealtime, 1: có thể chạy manual ở runRealtime]
                        'Blank' : [1, 1, 1],
                        }

        df = pd.DataFrame(listModules).transpose()
        df.columns = ['RunRealtime', 'Auto', 'Manual']
        df.to_excel('signals/_listModules.xlsx')
    
    return listModules

def initDetectBy(detectby, testing):
    if _checkDetectBy(detectby, testing) == 'DjLongleg4testing':
        import signals.candlesticks.DjLongleg4testing as sig
    elif _checkDetectBy(detectby, testing) == 'Blank': 
        import signals.Blank as sig
    elif _checkDetectBy(detectby, testing) == 'Doji': 
        import signals.candlesticks.Doji as sig
    elif _checkDetectBy(detectby, testing) == 'Hammer': 
        import signals.candlesticks.Hammer as sig
    elif _checkDetectBy(detectby, testing) == 'Engulfing':
        import signals.candlesticks.Engulfing as sig
    elif _checkDetectBy(detectby, testing) == 'Test':
        import signals.test.Test as sig
    elif _checkDetectBy(detectby, testing) == 'RFRclassifier':
        import signals.ml.RFRclassifier as sig
    elif _checkDetectBy(detectby, testing) == 'RFRclassifierDummy':
        import signals.ml.RFRclassifierDummy as sig
    else:
        return None
    return sig
