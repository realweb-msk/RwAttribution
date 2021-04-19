from attribution.markov import RwMarkov
import pandas as pd


data_1 = pd.read_csv('../data/attribution data.csv', sep=',')
# data_2 = pd.read_csv('data_attrib_AB_credit-card.csv', sep=',', converters={'client_id' : str})

markov_1 = RwMarkov(data_1, 'channel', 'cookie', 'time', 'conversion', verbose=1)
# markov_1 = RwMarkov(data_2, 'channel_group', 'client_id', 'visit_start_time', 'session_with_conversion', verbose=1)
# print(markov_2.make_markov(250000, 0.09))


# conv = markov_1.conversions_per_channel(prep)
# print(markov_1.removal_effect(matr))#, conv))
print(markov_1.make_markov())


