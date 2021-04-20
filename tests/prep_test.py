import pandas as pd
from tools.prep import prep_data


# Load data
data = pd.read_csv("../data/attribution data.csv", converters={'cookie': False})

# Data prep
full_gr, click_gr, view_gr = prep_data(data, "channel", 'cookie', 'interaction')

print(full_gr)