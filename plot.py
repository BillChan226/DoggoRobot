import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

#df1 = pd.read_csv("./result/multiagents/figures")
df1 = pd.read_csv("result/multiagents/scores_20_3_agent")
df3 = pd.read_csv("result/multiagents/scores_40_3_agent")
df2 = pd.read_csv("result/multiagents/scores_60_3_agent")
# df1 = pd.read_csv("./result/multiagents/scores_60_4_agent")
# df5 = pd.read_csv("./result/multiagents/scores_60_6_agent")
# df0 = pd.read_csv("./result/multiagents/scores_60_10_agent")

#plt.figure(2)
# df1.loc[df1['scores'] < -500, 'scores'] /= 60
# df2.loc[df2['scores'] < -500, 'scores'] /= 60
# df3.loc[df3['scores'] < -500, 'scores'] /= 60
#df1.loc[df1['scores'] < -1000, 'scores'] += 300
# df1.loc[df1['col'] > 100, 'col'] -= 100
# df1.loc[df1['col'] > 70, 'col'] -= 30
# df2.loc[df2['col'] > 80, 'col'] -= 120
# df3.loc[df3['col'] > 80, 'col'] -= 120

# x0 = df0.iloc[:, 1]*1000
# y0 = df0.iloc[:, 2]
#y1 = gaussian_filter1d(y1, sigma=1)

# plot the data using matplotlib
# plt.plot(x0, y0, linewidth=2,label="1 agent")

x1 = df1.iloc[:, 1]*1000
y1 = df1.iloc[:, 2]
#y1 = gaussian_filter1d(y1, sigma=1)

# plot the data using matplotlib
plt.plot(x1, y1, linewidth=2,label="30%")


x2 = df2.iloc[:, 1]*1000
y2 = df2.iloc[:, 2]
#y2 = gaussian_filter1d(y2, sigma=1)

# plot the data using matplotlib
plt.plot(x2, y2, linewidth=2, label="60%")
#plt.legend("40%")

x3 = df3.iloc[:, 1]*1000
y3 = df3.iloc[:, 2]
#y3 = gaussian_filter1d(y3, sigma=1)

plt.plot(x3, y3, linewidth=2,label="90%")

# x4 = df4.iloc[:, 1]*1000
# y4 = df4.iloc[:, 2]
# #y3 = gaussian_filter1d(y3, sigma=1)

# plt.plot(x4, y4, linewidth=2,label="6 agents")

# x5 = df5.iloc[:, 1]*1000
# y5 = df5.iloc[:, 2]
# #y3 = gaussian_filter1d(y3, sigma=1)

# plt.plot(x5, y5, linewidth=2,label="10 agents")
#plt.legend("80%")

plt.legend()

#plt.plot(df1)
plt.title('Accumulated Scores w.r.t. Observability (3 agents)')
plt.xlabel('total time steps')
plt.ylabel('Scores')
plt.savefig("./test.png")