import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("output/timeShared0.csv")
df2 = pd.read_csv("output/timeShared1.csv")
df3 = pd.read_csv("output/timeShared2.csv")
df4 = pd.read_csv("output/timeShared3.csv")
df5 = pd.read_csv("output/timeGlobal0.csv")
df6 = pd.read_csv("output/timeGlobal1.csv")
df7 = pd.read_csv("output/timeGlobal2.csv")
df8 = pd.read_csv("output/timeGlobal3.csv")

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

axes[0].plot(df['Dimensions'], df['Time'], label='blk-grid size 64-14', color='blue')
axes[0].plot(df2['Dimensions'], df2['Time'], label='32-7', color='red')
axes[0].plot(df3['Dimensions'], df3['Time'], label='16-3', color='green')
axes[0].plot(df4['Dimensions'], df4['Time'], label='8-1', color='yellow')
axes[0].set_title('Shared Memory Matrix Transpose')
axes[0].legend()

# Plot second graph
axes[1].plot(df5['Dimensions'], df5['Time'], label='64-14', color='blue')
axes[1].plot(df6['Dimensions'], df6['Time'], label='32-7', color='red')
axes[1].plot(df7['Dimensions'], df7['Time'], label='16-3', color='green')
axes[1].plot(df8['Dimensions'], df8['Time'], label='8-1', color='yellow')
axes[1].set_title('Global Memory Matrix Transpose')
axes[1].legend()

plt.xlabel("Matrix n*n dimension")
plt.ylabel("Time (s)")
plt.tight_layout()
plt.show()

mean = np.mean(df8['Time'])
median = np.median(df8['Time'])
std_dev = np.std(df8['Time'])
min_val = np.min(df8['Time'])
max_val = np.max(df8['Time'])
sum = np.sum(df8['Time'])

print("Mean:", mean)
print("Median:", median)
print("Standard Deviation:", std_dev)
print("Min:", min_val)
print("Max:", max_val)
print("Sum: ", sum)

fig, axes = plt.subplots(2, 1, figsize=(10, 8))
bandwidth1 = 0.348492
bandwidth2 = ((19879305848 + 8695828316) / np.power(10, 9)) / 71.164095
bandwidth3 = ((19879305848 + 8695828316) / np.power(10, 9)) / 69.479736
bandwidth4 = ((19879305848 + 8695828316) / np.power(10, 9)) / 60.077699
bandwidth5 = 36.530886
bandwidth6 = ((19773088714 + 8642431561) / np.power(10, 9)) / 57.886146999999994
bandwidth7 = ((19773088714 + 8642431561) / np.power(10, 9)) / 74.12973099999999
bandwidth8 = ((19773088714 + 8642431561) / np.power(10, 9)) / 75.73757499999999
firstplot = [bandwidth1, bandwidth2, bandwidth3, bandwidth4]
secondplot = [bandwidth5, bandwidth6, bandwidth7, bandwidth8]

axes[0].plot(["64-14", "32-7", "16-3", "8-1"], firstplot, color='blue')
axes[0].set_title('Shared Memory Matrix Transpose')

# Plot second graph
axes[1].plot(["64-14", "32-7", "16-3", "8-1"], secondplot, color='blue')
axes[1].set_title('Global Memory Matrix Transpose')

plt.xlabel("Block and grid sizes")
plt.ylabel("Bandwidth GB/s")
plt.tight_layout()
plt.show()