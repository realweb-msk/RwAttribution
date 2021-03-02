import pandas as pd
import numpy as np
from collections import defaultdict


def markov_prep(df, id_col, order_col, channel_col, conv_col):

    """Preprocess data for markov chains"""

    df = df.sort_values([id_col, order_col])

    # Нужно для тестовых данных
    df['interaction_number'] = df.groupby(id_col).cumcount() + 1
    df_path = df.groupby(id_col)[channel_col].aggregate(
    lambda x: x.tolist()).reset_index()

    df_last_int = df.drop_duplicates(id_col, keep='last')[[id_col, conv_col]]
    df_path = df_path.merge(df_last_int, how = 'left',
                                           left_on = id_col, right_on = id_col)

    # TODO: Для нормальных данных


    # Добавим начало и конец цепочки
    df_path[channel_col].apply(lambda x: x.insert(0, "Start"))
    df_path.query(f'{conv_col} == 0')[channel_col].apply(lambda x: x.append('Null'))
    df_path.query(f'{conv_col} == 1')[channel_col].apply(lambda x: x.append('Conversion'))

    return df_path


def transitions(path_list, unique_channels):
    """
    Computes ALL transition points in paths
    """
    unique_channels = np.append(unique_channels, ['Start', 'Conversion', 'Null'])
    states = {x + '>' + y: 0 for x in unique_channels for y in unique_channels}

    # Ugly cycles
    # TODO: Try to optimize somehow
    for state in unique_channels:
        if state not in ['Conversion', 'Null']:
            for current_path in path_list:
                if state in current_path:
                    indices = [i for i, s in enumerate(current_path) if state in s]
                    for col in indices:
                        states[current_path[col] + '>' + current_path[col + 1]] += 1

    return states



