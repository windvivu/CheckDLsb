checklist = {'1m': 60*1000,'3m':3*60*1000,'5m':5*60*1000,'15m':15*60*1000,'30m':30*60*1000,'1h':60*60*1000, \
              '2h': 2*60*60*1000, '4h': 4*60*60*1000, '6h': 6*60*60*1000, '8h': 8*60*60*1000, '12h': 12*60*60*1000, '1d': 24*60*60*1000
             }
def calStepts(timeframe):
    if timeframe in checklist.keys:
        return checklist[timeframe]
    else:
        return None

def calSteptsReverse(numseconds):
    for k,v in checklist.items():
        if v == numseconds:
            return k
    return None