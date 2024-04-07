import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("output/time.csv")
gmean= df.mean()
gstd = df.std()
gmean.plot(yerr=gstd,ylabel="time (s)", xlabel="log2(n)")
plt.show()