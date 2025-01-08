def find_row_index_ts(df, x, start=0, end=None):
    '''
    Find the index of the row in the dataframe where the timestamp is equal to x.
    '''
    if end is None:
        end = len(df) - 1 
    if start > end:
        return -1
    mid = (start + end) // 2
    if df['timestamp'].iloc[mid] == x:
        return mid
    elif df['timestamp'].iloc[mid] < x:
        return find_row_index_ts(df, x, mid + 1, end)
    else:
        return find_row_index_ts(df, x, start, mid - 1)