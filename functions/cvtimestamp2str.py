import datetime
def cvtimestamp2str(timestamp, format='%Y-%m-%d %H:%M:%S', GMTs=0):
    
    if len(str(timestamp)) >= 13:
        timestamp = timestamp/1000
    
    # chuyển đổi múi giờ
    timestamp = timestamp + GMTs*3600

    datetime_str = datetime.datetime.fromtimestamp(timestamp)
    # định dạng thời gian
    datetime_str = datetime_str.strftime(format)
    return datetime_str