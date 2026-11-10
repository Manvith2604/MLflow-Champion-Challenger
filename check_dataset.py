import pandas as pd

# Load evaluation dataset
df = pd.read_csv("data/eval_inputs.csv")

# Display dataset
print(df)

# Display number of examples
print("\nTotal examples:", len(df))

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())