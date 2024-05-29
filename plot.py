import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

axes[0].plot(["Block -O0", "Block -O1", "Block -O2", "Block -O3"], firstplot, color='blue')
axes[0].set_title('Block Matrix Transpose')

# Plot second graph
axes[1].plot(["Normal -O0", "Normal -O1", "Normal -O2", "Normal -O3"], secondplot, color='blue')
axes[1].set_title('Normal Matrix Transpose')

plt.xlabel("Optimization flag")
plt.ylabel("Bandwidth GB/s")
plt.tight_layout()
plt.show()