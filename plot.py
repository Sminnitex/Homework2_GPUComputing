import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("output/timeB0.csv")
df2 = pd.read_csv("output/timeB1.csv")
df3 = pd.read_csv("output/timeB2.csv")
df4 = pd.read_csv("output/timeB3.csv")
df5 = pd.read_csv("output/timeN0.csv")
df6 = pd.read_csv("output/timeN1.csv")
df7 = pd.read_csv("output/timeN2.csv")
df8 = pd.read_csv("output/timeN3.csv")

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

axes[0].plot(df['Dimensions'], df['Time'], label='O0', color='blue')
axes[0].plot(df2['Dimensions'], df2['Time'], label='O1', color='red')
axes[0].plot(df3['Dimensions'], df3['Time'], label='O2', color='green')
axes[0].plot(df4['Dimensions'], df4['Time'], label='O3', color='yellow')
axes[0].set_title('Block Matrix Transpose')
axes[0].legend()
axes[0].set_xlim(1537, 2048)
axes[0].set_ylim(0.3, 2)

# Plot second graph
axes[1].plot(df5['Dimensions'], df5['Time'], label='O0', color='blue')
axes[1].plot(df6['Dimensions'], df6['Time'], label='O1', color='red')
axes[1].plot(df7['Dimensions'], df7['Time'], label='O2', color='green')
axes[1].plot(df8['Dimensions'], df8['Time'], label='O3', color='yellow')
axes[1].set_title('Normal Matrix Transpose')
axes[1].legend()

plt.xlabel("Matrix n*n dimension")
plt.ylabel("Time (s)")
plt.tight_layout()
plt.xlim(1537, 2048)
plt.ylim(0.3, 2)
plt.show()