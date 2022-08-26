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
for col in ["ternary", "personalized", "iid"]: df[col] = [eval(x) if x!="single_learner" else x for x in df[col]]

fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(nrows=2, ncols=2)
palette = {True:"blue", False:"orange", "single_learner":"black"}

sub = df[(df.ternary==True) & (df.personalized==True)]
sns.lineplot(data=sub, x="dataset", y="acc (server)", hue="iid", palette=palette, ax=ax1)
ax1.set_title("Ternray=True | Person.=True")

sub = df[(df.ternary==True) & (df.personalized==False)]
sns.lineplot(data=sub, x="dataset", y="acc (server)", hue="iid", palette=palette, ax=ax2)
ax2.set_title("Ternray=True | Person.=False")

sub = df[(df.ternary==False) & (df.personalized==True)]
sns.lineplot(data=sub, x="dataset", y="acc (server)", hue="iid", palette=palette, ax=ax3)
ax3.set_title("Ternray=False | Person.=True")

sub = df[(df.ternary==False) & (df.personalized==False)]
sns.lineplot(data=sub, x="dataset", y="acc (server)", hue="iid", palette=palette, ax=ax4)
ax4.set_title("Ternray=False | Person.=False")


for ax in [ax1, ax2, ax3, ax4]:
    ax.set_ylim(-0.1, 1)
ax1.legend()
plt.show()

fig, ax = plt.subplots()
ax = sns.lineplot(data=df_single, x="dataset", y="acc (server)")
ax.set_ylim(-0.1, 1)
plt.show()
