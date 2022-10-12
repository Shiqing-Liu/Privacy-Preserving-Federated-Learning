import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

PATH = os.path.join(os.getcwd(), "results", "MNIST_test")
df_agents = pd.read_csv(os.path.join(os.getcwd(), "results", "table_results.csv"))
df_agents.ternary = [eval(x) for x in df_agents.ternary]
df_agents.personalized = [eval(x) for x in df_agents.personalized]
df_agents.iid = [eval(x) for x in df_agents.iid]
data = {}

# Store configurations
for _, series in df_agents.iterrows():
    temp_dict = {}
    for name, value in zip(series.index.to_numpy(), series.to_numpy()):
        if name in ["number", "responsible", "acc (server)", "acc (c1)", "acc (c2)", "acc (c3)"]:
            continue
        temp_dict[name] = value
    data[series["number"]] = temp_dict

def transform_ternary_data(tdata):
    cumsum = {}
    for (r, j) in tdata:
        if r in cumsum.keys():
            cumsum[r] += j
        else:
            cumsum[r] = j
    return np.array(list(zip(cumsum.keys(), cumsum.values())))

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
        data[n]["server_sent"] = transform_ternary_data(eval(lines[26][11:]))
        data[n]["server_received"] = transform_ternary_data(eval(lines[25][15:]))
        start_line = 30
        for i in range(eval(data[n]["clients"])):
            cur_line = start_line + (12 * i)
            client = lines[cur_line][24]
            data[n][f"c{client}_signals"] = eval(lines[cur_line+2][9:])
            data[n][f"c{client}_acc"] = eval(lines[cur_line+4][10:])
            data[n][f"c{client}_loss"] = eval(lines[cur_line+5][6:])
            data[n][f"c{client}_received"] = transform_ternary_data(eval(lines[cur_line+7][15:]))
            data[n][f"c{client}_sent"] = transform_ternary_data(eval(lines[cur_line+8][11:]))

count = 0
for _, item in data.items():
    if len(item) != 9: count += 1
print(f"Found {count} of {len(data)} training results!")

##########---------- Visulalisations ----------##########
def plot1():
    color = {True: "orange", False: "blue", np.nan:"black"}
    color_num = {1: "yellow", 2: "orange", 3: "red", 4: "purple", 5: "blue", 6: "lightblue", 7:"lightgreen", 8: "darkgreen", 9: "black"}
    color_iid = {1: "black", 2: "black", 3: "orange", 4: "orange", 5: "blue", 6: "blue", 7:"green", 8: "green", 9: "red"}
    line = {True: "-.", False: "-", np.nan:"-"}
    marker = {True: "x", False: "", np.nan:""}
    fig, ax = plt.subplots(figsize=(15, 10))
    for num, value in data.items():
        ax.plot(value["server_acc"], color=color_iid[num], linestyle=line[value["iid"]], label=value["name"])
    ax.set_ylabel("Accuracy")
    ax.set_yticks(np.arange(11)/10)
    ax.set_xlabel("Global Round")
    ax.set_xticks(np.arange(26))
    ax.set_title("server accuracy for different configurations")
    ax.grid()
    ax.legend()
    fig.show()

def plot1_clients(c_num=1):
    color_iid = {1: "black", 2: "black", 3: "orange", 4: "orange", 5: "blue", 6: "blue", 7:"green", 8: "green", 9: "red"}
    line = {True: "-.", False: "-", np.nan:"-"}
    fig, ax = plt.subplots(figsize=(15, 10))
    for num, value in data.items():
        if num==9:
            ax.plot(value["server_acc"], color=color_iid[num], linestyle=line[value["iid"]], label=value["name"])
            continue
        ax.plot(value[f"c{c_num}_acc"], color=color_iid[num], linestyle=line[value["iid"]], label=value["name"])
    ax.set_ylabel("Accuracy")
    ax.set_yticks(np.arange(11)/10)
    ax.set_xlabel("Global Round")
    ax.set_xticks(np.arange(26))
    ax.set_title(f"Client {c_num} accuracy for different configurations")
    ax.grid()
    ax.legend()
    fig.show()

def plot2():
    color_iid = {1: "black", 2: "black", 3: "orange", 4: "orange", 5: "blue", 6: "blue", 7:"green", 8: "green", 9: "red"}
    line = {True: "-.", False: "-", np.nan:"-"}
    fig, ax = plt.subplots(figsize=(15, 10))
    for num, value in data.items():
        if num == 9: continue
        ax.plot(np.cumsum(value["server_sent"][:,-1]), color=color_iid[num], linestyle=line[value["iid"]], label=value["name"])
    ax.set_ylabel("Data")
    #ax.set_yticks(np.arange(11)/10)
    ax.set_xlabel("Global Round")
    #ax.set_xticks(np.arange(26))
    ax.set_title("server sent data for different configurations")
    ax.grid()
    ax.legend()
    fig.show()

    fig, ax = plt.subplots(figsize=(15, 10))
    for num, value in data.items():
        if num == 9: continue
        ax.plot(np.cumsum(value["server_received"][:,-1]), color=color_iid[num], linestyle=line[value["iid"]], label=value["name"])
    ax.set_ylabel("Data")
    #ax.set_yticks(np.arange(11)/10)
    ax.set_xlabel("Global Round")
    #ax.set_xticks(np.arange(26))
    ax.set_title("server received data for different configurations")
    ax.grid()
    ax.legend()
    fig.show()

def plot3():
    sdata = []
    rdata = []
    for num, value in data.items():
        if num==9: continue
        sdata.append(np.sum(value["server_sent"][:,1]))
        rdata.append(np.sum(value["server_received"][:,1]))
    print(sdata, rdata)
    sdata = ((np.array(sdata) / np.array([sdata[0]]*8)) - 1)[1:]
    rdata = ((np.array(rdata) / np.array([rdata[0]]*8)) - 1)[1:]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(np.arange(7)-.35/2, sdata, color="blue", width=.35, label="sent")
    ax.bar(np.arange(7)+.35/2, rdata, color="orange", width=.35, label="received")
    ax.axhline(0, color="black", lw=2)
    ax.set_ylabel("Percentage")
    #ax.set_yticks(np.arange(11)/10)
    ax.set_xlabel("Configuration")
    ax.set_xticks(np.arange(7))
    ax.set_xticklabels(np.arange(2, 9))
    ax.set_title("Percentage of sent/received data compared to conf. 1 (F_F_F)")
    ax.grid()
    ax.legend()
    fig.tight_layout()
    fig.show()
    print()


##########---------- Tables ----------##########
def table1():
    df_acc = pd.DataFrame({"round": np.arange(1, data[1]["rounds"]+1)})
    df = pd.DataFrame({"agent": np.arange(1, 10)})
    end, x70, x80, x90, x95 = [], [], [], [], []
    for num, value in data.items():
        accs = value["server_acc"]
        cond = [accs[-1]]
        end.append(accs[-1])
        n70 = "-"
        n80 = "-"
        n90 = "-"
        n95 = "-"
        for i, acc in enumerate(accs):
            if n70=="-" and acc>.7: n70 = i+1
            if n80=="-" and acc>.8: n80 = i+1
            if n90=="-" and acc>.9: n90 = i+1
            if n95=="-" and acc>.95: n95 = i+1
        df_acc[num] = accs
        x70.append(n70)
        x80.append(n80)
        x90.append(n90)
        x95.append(n95)
    df["end"] = end
    df[">70"] = x70
    df[">80"] = x80
    df[">90"] = x90
    df[">95"] = x95
    return df

plot1()