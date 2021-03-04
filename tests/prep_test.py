import pandas as pd

from prep import prep_data
from shapley import RwShap

mazda = pd.read_csv('/data_for_atribution.csv',
                    converters = {'clientId' : str})

full_gr, click_gr, view_gr = prep_data(mazda, 'channel_grouping', 'clientId', 'post_click_or_post_view')
# data = pd.read_csv('data_for_atribution.csv')
# full_gr, click_gr, view_gr = prep_data(data, 'channel_grouping', 'clientId', 'post_click_or_post_view')

unique_channels = mazda['channel_grouping'].unique()
# unique_channels = data['channel'].unique()
shap_cl = RwShap(click_gr, 'path', 'conversion')
print(shap_cl.shapley_value(channels=unique_channels))

