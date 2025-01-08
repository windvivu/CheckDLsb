import datetime

def period2timestamp(minutes=0, hours=0, days=0, months=0 , milisecond = False):
    
    if milisecond:
        m = 1000
    else:
        m=1

    # 1 tháng = 30*24*60*60*1000
    months = months * 30*24*60*60*m
    # 1 ngày = 24*60*60*1000
    days = days * 24*60*60*m
    # 1 giờ = 60*60*1000
    hours = hours * 60*60*m
    # 1 phút = 60*1000
    minutes = minutes * 60*m

    return int(months + days + hours + minutes)