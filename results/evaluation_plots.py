import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

PATH = os.path.join(os.getcwd(), "results", "MNIST_test")
df_agents = pd.read_csv(os.path.join(os.getcwd(), "results", "table_results.csv"))
data = {}

# Store configurations
for _, series in df_agents.iterrows():
    temp_dict = {}
    for name, value in zip(series.index.to_numpy(), series.to_numpy()):
        if name in ["number", "responsible", "acc (server)", "acc (c1)", "acc (c2)", "acc (c3)"]:
            continue
        temp_dict[name] = value
    data[series["number"]] = temp_dict

# Store data
for fname in os.listdir(PATH):
    n = int(fname[0])
    with open(os.path.join(PATH, fname, "configuration.txt"), "r") as f:
        lines = f.readlines()
    if n == 9:
        data[n]["server_acc"] = eval(lines[13][6:])
        data[n]["server_loss"] = eval(lines[15][8:])
    else:
        data[n]["server_acc"] = eval(lines[23][10:])
        data[n]["server_loss"] = eval(lines[24][6:])
        data[n]["server_received"] = eval(lines[25][15:])
        data[n]["server_sent"] = eval(lines[26][11:])
        start_line = 30
        for i in range(eval(data[n]["clients"])):
            cur_line = start_line + (12 * i)
            client = lines[cur_line][24]
            data[n][f"c{client}_signals"] = eval(lines[cur_line+2][9:])
            data[n][f"c{client}_acc"] = eval(lines[cur_line+4][10:])
            data[n][f"c{client}_loss"] = eval(lines[cur_line+5][6:])
            data[n][f"c{client}_received"] = eval(lines[cur_line+7][15:])
            data[n][f"c{client}_sent"] = eval(lines[cur_line+8][11:])

count = 0
for _, item in data.items():
    if len(item) != 9: count += 1
print(f"Found {count} of {len(data)} training results!")