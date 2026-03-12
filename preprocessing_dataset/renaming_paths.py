import csv
import pandas as pd

df = pd.read_csv('C:/FYP/csvs/caxton_dataset_final.csv')

df['img_path'] = df['img_path'].str.replace('caxton_dataset', 'C:/FYP/full_dataset', regex=False)

df.to_csv('C:/FYP/phase1/full_dataset.csv', index=False)