import os
import glob
from ftplib import FTP
import pandas as pd
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import functions.en_decode as ed

sync_all = False # True thì sẽ sync toàn bộ bất kể
runAlone = True
source = 'Binance'

def get_args():
    from argparse import ArgumentParser
    global runAlone, source

    parser = ArgumentParser()
    parser.add_argument('--runAlone','-rA',type=int, default=1)
    parser.add_argument('--source','-s',type=str, default=source)
    args = parser.parse_args()
    source = args.source
    if args.runAlone == 0:
        runAlone = False

get_args()

if runAlone:
    print('Sync data from:')
    print('\t1. Spot Binance')
    print('\t2. Futere Binance')
    print('\t3. Stock Yahoo finance')
    choice = input('Choose source: ')
    if choice.strip() == '1':
        source = 'Binance'
    elif choice.strip() == '2':
        source = 'FBinance'
    elif choice.strip() == '3':
        source = 'Yfinance'
    else:
        print('Invalid choice!')
        exit(0)


# Thông tin kết nối FTP server
FTP_HOST = '184.168.116.210'
FTP_USER = ed.mydecodeb64('WW01a1lYUmhRR2h0Y1M1MWpmdVl0cmc2Y3c9PQ==', 'jfuYtrg6')
FTP_PASS = ed.mydecodeb64('ZFh1WWhmNnJkZ0pkdEJBWkdGMFlVQmlhVzVoYm1ObA==','uYhf6rdgJdt')

if source == 'Yfinance':
    FTP_PATH = '/YFstocks'
    dirDB = './YFstocks'
elif source == 'FBinance':
    FTP_PATH = '/BNfuture'
    dirDB = './BNfuture'
else:
    FTP_PATH = '/BNspot'
    dirDB = './BNspot'

# set the current working directory is the same as the directory of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def upload_file(ftp, file_path):
    """
    Hàm upload file lên máy chủ FTP
    :param ftp: Đối tượng FTP
    :param file_path: Đường dẫn file cần upload
    """
    try:
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        print(f'Uploading {file_name} ({file_size} bytes)...')

        with open(file_path, 'rb') as file:
            ftp.storbinary(f'STOR {file_name}', file, callback=upload_progress(file_size))

        print(f'\nUpload {file_name} complete.')
    except Exception as e:
        print(f'Error uploading file: {e}')

def upload_progress(file_size):
    """
    Hàm callback để theo dõi tiến trình upload
    :param file_size: Kích thước file
    :return: Hàm callback
    """
    def callback(data):
        nonlocal uploaded
        uploaded += len(data)
        progress = (uploaded / file_size) * 100
        print(f'Uploaded {uploaded}/{file_size} bytes ({progress:.2f}%)', end='\r')

    uploaded = 0
    return callback

def download_file(ftp, remote_path, local_path):
    try:
        # Lấy kích thước tập tin từ máy chủ
        remote_size = ftp.size(remote_path)

        # Khởi tạo tiến trình tải
        loaded = 0
        with open(local_path, "wb") as file:
            def handle_binary(data):
                nonlocal loaded
                file.write(data)
                loaded += len(data)
                progress = (loaded / remote_size) * 100
                print(f'Downloaded ({progress:.2f}%)', end='\r')

            # Tải tập tin từ máy chủ
            ftp.retrbinary(f"RETR {remote_path}", handle_binary)
            print("\nDownload complete!")

    except Exception as e:
        print(f'Error downloading file: {e}')
  



ftp = FTP()
try:
    print('Connecting to FTP server...')
    ftp.connect(FTP_HOST)
    ftp.login(FTP_USER, FTP_PASS)
    ftp.cwd(FTP_PATH)

    print('Connected!')
except Exception as e:
    print(f'Error connecting to FTP: {e}')
    ftp.quit()
    exit(0)

# lấy danh sách file db trên máy chủ
try:
    print('Getting file list on host...')
    ftp.cwd(FTP_PATH)
    files_on_host = ftp.nlst()
    files_on_host = [f for f in files_on_host if f.endswith('.db')]
except Exception as e:
    print(f'Error getting file list: {e}')
    ftp.quit()
    exit(0)


# create directory dirDB if not exists
if not os.path.exists(dirDB):
    os.makedirs(dirDB)

# lấy danh sách file db trên máy cục bộ
files = glob.glob(dirDB + '/*.db')

#--------lấy danh sách file cần sync-------
if sync_all == False:
    if source=='Yfinance':
        listDt = pd.read_excel('stocklist.xlsx')
        listDt = listDt[listDt['sync'] == 'x']
        listSync = listDt['asset'].to_list()
        listSync = [f+'.db' for f in listSync]
    if source=='FBinance':
        listDt = pd.read_excel('futurelist.xlsx')
        listDt = listDt[listDt['sync'] == 'x']
        listSync = listDt['asset'].to_list()
        listSync = [f+'_USDT.db' for f in listSync] # lưu ý chỗ _USDT.db, đặc trưng của dữ liệu binance
    else:
        listDt = pd.read_excel('spotlist.xlsx')
        listDt = listDt[listDt['sync'] == 'x']
        listSync = listDt['asset'].to_list()
        listSync = [f+'_USDT.db' for f in listSync]

    files_on_host = [f for f in files_on_host if os.path.basename(f) in listSync]
    files = [f for f in files if os.path.basename(f) in listSync]
#-------------------------------------------

print('Files on host:', len(files_on_host), ' - Files on local:', len(files))
if len(files_on_host) == 0 and len(files) == 0:
    print('Nothing to sync!')
    ftp.quit()
    exit(0)

print('Preparing the list of files to sync...')

# Kiểm tra kích thước file trên host và đưa vào dictionary
files_on_host_dict = {}
try:
    for file in files_on_host:    
        file_size = ftp.size(file)
        files_on_host_dict[file] = file_size
except Exception as e:
    print(f'Error getting file size on host: {e}')
    ftp.quit()
    exit(0)

# Kiểm tra kích thước file cục bộ và đưa vào dictionary
files_on_local_dict = {}
for file in files:
    file_size = os.path.getsize(file)
    files_on_local_dict[os.path.basename(file)] = file_size

# So sánh kích thước file giữa host và local
files_to_download = []
need_down = 0
files_to_upload = []
need_up=0
for file, size in files_on_host_dict.items():
    if file not in files_on_local_dict or files_on_local_dict[file] < size:
        files_to_download.append(file)
        need_down += 1

for file, size in files_on_local_dict.items():
    if file not in files_on_host_dict or files_on_host_dict[file] < size:
        files_to_upload.append(file)
        need_up+=1

# upload file-------------------------
if len(files_to_upload) > 0:
    print('Begin uploading files...')

uploaded =0
for index, file in enumerate(files_to_upload):
    LOCAL_FILE_PATH = dirDB + '/' + file
    LOCAL_FILE = os.path.basename(LOCAL_FILE_PATH)

    print(f'{index+1}/{need_up}','-'*40)

    fileDownloading = glob.glob(dirDB + '/*.db.txt')
    fileDownloading = [os.path.basename(f) for f in fileDownloading]
    if file + '.txt' in fileDownloading:
        print(f'File {file} is downloading, skip uploading...')
        continue
       
    # Upload file
    try:
        local_file_size = os.path.getsize(LOCAL_FILE_PATH)
        # convert to MB
        local_file_size_mb = local_file_size / (1024 * 1024)
        print(f'File size: {local_file_size_mb:.2f} MB')
        print(f'Uploading file {LOCAL_FILE} ...')
        upload_file(ftp, LOCAL_FILE_PATH)
        uploaded += 1
    except Exception as e:
        print(f'Error uploading file: {e}')
        continue
    
print(f'---Upload done: {uploaded}/{need_up} files!---')

# download file-------------------------
if len(files_to_download) > 0:
    print('Begin download files...')

downloaded =0
for index, file in enumerate(files_to_download):
    REMOTE_FILE_NAME = file
    LOCAL_FILE_PATH = dirDB + '/' + file

    # check if file exists on server
    print(f'{index+1}/{need_down}','-'*40)

    fileDownloading = glob.glob(dirDB + '/*.db.txt')
    fileDownloading = [os.path.basename(f) for f in fileDownloading]
    if file + '.txt' in fileDownloading:
        print(f'File {file} is downloading, skip downloading...')
        continue

    # download file
    try:
        host_file_size = files_on_host_dict[file]
        # convert to MB
        host_file_size_mb = host_file_size / (1024 * 1024)
        print(f'File size: {host_file_size_mb:.2f} MB')
        print(f'Downloading file {REMOTE_FILE_NAME} ...')
        download_file(ftp, REMOTE_FILE_NAME, LOCAL_FILE_PATH)
        downloaded += 1
    except Exception as e:
        print(f'Error downloading file: {e}')
        continue

ftp.quit()
print(f'---Download done: {downloaded}/{need_down} files!---')