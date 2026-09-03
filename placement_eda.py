import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("/placement_predict_50k Dataset.csv")
print(df.head())
print(df.tail())
print(df.shape)
print(df.columns.tolist())
print(df.info())
print(df.describe())
print(df.sample(5))
plt.figure(figsize=(6,4))
sns.kdeplot(df["CodingTestScore"],fill=True)
plt.title("Coding Test Score Density")
plt.show()
plt.figure(figsize=(6,4))
sns.boxplot(x=df["CGPA"])
plt.title("CGPA Box Plot")
plt.show()
