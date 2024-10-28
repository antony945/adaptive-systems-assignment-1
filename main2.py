import pandas as pd

main_column = "description"

# Read the Excel file
file_path = 'news1.csv'
df = pd.read_csv(file_path, delimiter=",")       

print(df.info())

# Pay attention that some rows may don't have description, so we'll not include those articles there
df = df.dropna(subset=[main_column]).reset_index(drop=True)

# print(df.info())
print(df.info())