from attribution.markov import RwMarkov
import pandas as pd


data_1 = pd.read_csv('../data/attribution data.csv', sep=',')

markov_1 = RwMarkov(data_1, 'channel', 'cookie', 'time', 'conversion', verbose=1)

print(markov_1.make_markov())


