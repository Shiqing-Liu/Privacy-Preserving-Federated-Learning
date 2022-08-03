import os, time
TIME_STAMP = f"{time.localtime().tm_year}.{time.localtime().tm_mon}.{time.localtime().tm_mday}_{time.localtime().tm_hour}.{time.localtime().tm_min}"
SAVE_PATH = os.path.join(os.getcwd(), "results", TIME_STAMP)
if os.path.isdir(SAVE_PATH):
    raise FileExistsError

os.mkdir(SAVE_PATH)

SERVER_HOST = "localhost"
SERVER_PORT = 9990
PERSONALIZED = True
