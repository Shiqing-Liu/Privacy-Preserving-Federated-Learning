import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("/Users/fabrei/OneDrive - uni-bielefeld.de/Studium/Current/2-Project_FL/Federated-Learning/results/table_results.csv")
df_single = df[df.single==True]
df = df[df.single==False]

# add single learner to facet grid
df.loc[90] = [0,"CIFAR100",np.nan,np.nan,10000,"True","True","single_learner",0.019,0,0,0,True]
df.loc[91] = [0,"CIFAR100",np.nan,np.nan,10000,"True","False","single_learner",0.019,0,0,0,True]
df.loc[92] = [0,"CIFAR100",np.nan,np.nan,10000,"False","True","single_learner",0.019,0,0,0,True]
df.loc[93] = [0,"CIFAR100",np.nan,np.nan,10000,"False","False","single_learner",0.019,0,0,0,True]
df.loc[94] = [0,"CIFAR10",np.nan,np.nan,10000,"True","True","single_learner",0.337,0,0,0,True]
df.loc[95] = [0,"CIFAR10",np.nan,np.nan,10000,"True","False","single_learner",0.337,0,0,0,True]
df.loc[96] = [0,"CIFAR10",np.nan,np.nan,10000,"False","True","single_learner",0.337,0,0,0,True]
df.loc[97] = [0,"CIFAR10",np.nan,np.nan,10000,"False","False","single_learner",0.337,0,0,0,True]


g = sns.FacetGrid(df, col="personalized", row="ternary", hue="iid")#, height=2.5, aspect=2.5)
g.map(sns.lineplot, "dataset", "acc (server)")
g.set(ylim=(-0.1, 1), xlabel="Dataset", ylabel="Acc (Server)")
g.fig.suptitle("", fontsize=20)
g.fig.subplots_adjust(top=0.92, hspace=0.2)
g.add_legend()
g.legend.set_title("")
for ax in g.axes.flat:
    continue
plt.show()
fig, ax = plt.subplots()
ax = sns.lineplot(data=df_single, x="dataset", y="acc (server)")
ax.set_ylim(-0.1, 1)
plt.show()
