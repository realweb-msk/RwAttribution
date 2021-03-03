from markov import RwMarkov
import pandas as pd


data = pd.read_csv('attribution data.csv', sep=',')

markov = RwMarkov(data, 'channel', 'cookie', 'time', 'conversion', verbose=1)
print(markov.make_markov(17639, 0.2))