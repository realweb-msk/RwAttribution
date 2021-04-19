import pandas as pd

from tools import prep_data
from attribution import shap_and_freq, RwShapley

data = pd.read_csv('/data/alphabank.csv',
                   converters = {'client_id' : str})


click_gr = pd.read_csv('/Users/gleb/Desktop/click_gr.csv', converters = {'client_id' : str})
view_gr = pd.read_csv('/Users/gleb/Desktop/view_gr.csv', converters = {'client_id' : str})

# full_gr, click_gr, view_gr = prep_data(data, 'channel_grouping', 'clientId', 'post_click_or_post_view')
# data = pd.read_csv('data_for_atribution.csv')
# full_gr, click_gr, view_gr = prep_data(data, 'channel_grouping', 'clientId', 'post_click_or_post_view')

# unique_channels = data['channel_grouping'].unique()
# unique_channels = data['channel'].unique()
# shap_cl = RwShap(click_gr, 'path', 'conversion')
# print(shap_cl.shapley_value(channels=unique_channels))

unique_channels = data['channel_group'].unique()

shap_cl = RwShap(click_gr, 'path', 'conversion').shapley_value(channels=unique_channels)
shap_v = RwShap(view_gr, 'path', 'conversion').shapley_value(channels=unique_channels)

res = shap_and_freq(shap_cl, shap_v, data, "interaction_type", 'channel_group', 'channel_group',
                    'visit_start_time', 'client_id')
