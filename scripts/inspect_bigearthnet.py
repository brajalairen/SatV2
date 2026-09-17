import pandas as pd

path = "data/bigearthnet/txt/BigEarthNet.txt.parquet"

df = pd.read_parquet(path)

sample = df.iloc[0]

print("========== FIRST SAMPLE ==========")

for column in df.columns:
    print(f"\n{column}:")
    print(sample[column])