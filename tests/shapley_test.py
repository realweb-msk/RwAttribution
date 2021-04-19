import pandas as pd
from attribution import RwShap, shap_and_freq
from tools import prep_data, compute_FIC

data = pd.read_csv("../data/attribution data.csv")
data['interaction'] = data['interaction'].apply(lambda x: "Impression" if x == "impression" else "Click")

unique_channels = data['channel'].unique()

full_gr, click_gr, view_gr = prep_data(data, "channel", 'cookie', 'interaction')

shap_cl = RwShap(click_gr, 'path', 'conversion').shapley_value(channels=unique_channels)
shap_v = RwShap(view_gr, 'path', 'conversion').shapley_value(channels=unique_channels)

fic = compute_FIC(data, 'interaction', 'channel', 'time', 'cookie')

res = shap_and_freq(shap_cl, shap_v, fic, "channel", 'channel_group')
print(res)

