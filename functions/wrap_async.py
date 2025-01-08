def wrap_async(func, p, *args):
    '''
    func : là hàm cần bao bọc
    p: ký tự sẽ in ra khi đợi
    *arg: các tham số cần thiết cho hàm
    '''
    import threading
    import time

    t = threading.Thread(target=func, args=args)
    t.start()
    while t.is_alive():
        if p != '':
            print(p ,end='\r')
            p += p[0]
        time.sleep(1)

def wrap_async_return(func, p, *args):
    '''
    func : là hàm cần bao bọc
    p: ký tự sẽ in ra khi đợi
    *arg: các tham số cần thiết cho hàm
    '''
    from concurrent.futures import ThreadPoolExecutor
    import time

    with ThreadPoolExecutor() as executor:
            future = executor.submit(func, *args)
            while not future.done():
                print(p, end="", flush=True)
                time.sleep(0.5)
            print('')
            results = future.result()

    return results
