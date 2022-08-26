import os, time, torch
TIME_STAMP = f"{time.localtime().tm_year}.{time.localtime().tm_mon}.{time.localtime().tm_mday}_{time.localtime().tm_hour}.{time.localtime().tm_min}"
SAVE_PATH = os.path.join(os.getcwd(), "results", TIME_STAMP)
if os.path.isdir(SAVE_PATH):
    raise FileExistsError
os.mkdir(SAVE_PATH)

# M1 GPU support
print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
print(f"Is MPS available? {torch.backends.mps.is_available()}")
device = "mps" if torch.backends.mps.is_available() else "cpu"
device="cpu"
print(f"Using device: {device}")

SERVER_HOST = "localhost"
SERVER_PORT = 9999

