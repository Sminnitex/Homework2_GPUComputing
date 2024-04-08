import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("output/time.csv")
gmean= df["Time"]
gstd = df["Dimensions"]
plt.figure()
plt.plot(gstd, gmean)
plt.xlabel("Matrix n*n dimension")
plt.ylabel("Time (s)")
plt.title("Block matrix transpose -O1")
plt.xlim(513, 1024)
plt.ylim(0, 0.6)
plt.show()