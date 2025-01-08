def shorten_str(s, n=6):
    if len(s) > n:
        return s[:n//2] + '...' + s[-n//2:]
    return s