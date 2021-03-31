from basic_attribution import position, cpa
from prep import prep_data
import pandas as pd


data = pd.read_csv('attribution data.csv')
full_gr, click_gr, view_gr = prep_data(data, 'channel', 'cookie', 'interaction')
p = position(full_gr, data['channel'].unique())

print(p)
