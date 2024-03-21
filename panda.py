import pandas as pd

file = 'Book1.xlsx'

df = pd.read_excel(file)
for index, row in df.iterrows():
    if row is not None and row[1] != '00:01':
        try:
           if int(row[1]):
               print(row[1])
        except:
           continue