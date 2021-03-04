import pandas as pd
from prep import prep_data
from shapley import RwShap
from basic_attribution import first_click, last_click, last_non_direct_click, uniform


data = pd.read_csv('/Users/gleb/Documents/GitHub/RwAttribution/attribution data.csv')
full_gr, click_gr, view_gr = prep_data(data, 'channel', 'cookie', 'interaction', verbose=1)
# data = pd.read_csv('data_for_atribution.csv')
# full_gr, click_gr, view_gr = prep_data(data, 'channel_grouping', 'clientId', 'post_click_or_post_view')

# shap = RwShap(full_gr, 'path', 'cookie')
# print(shap.shapley_value(channels=data['channel'].unique()))

first_click(click_gr)