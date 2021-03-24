import pandas as pd
import numpy as np
from collections import defaultdict

class RwMarkov():

    def __init__(self, df, channel_col, id_col, order_col, conv_col, verbose=0):
        self.df = df
        self.channel_col = channel_col
        self.unique_channels = np.append(df[channel_col].unique(), ['Start', 'Conversion', 'Null'])
        self.id_col = id_col
        self.order_col = order_col
        self.conv_col = conv_col
        self.verbose = verbose


    def markov_prep(self):

        """Preprocess data for markov chains"""

        if self.verbose > 0:
            print("started markov_prep")

        id_col = self.id_col
        order_col = self.order_col
        channel_col = self.channel_col
        conv_col = self.conv_col

        df = self.df.sort_values([id_col, order_col])

        # Нужно для тестовых данных
        df['interaction_number'] = df.groupby(id_col).cumcount() + 1
        df_path = df.groupby(id_col)[channel_col].aggregate(lambda x: x.tolist()).reset_index()

        df_last_int = df.drop_duplicates(id_col, keep='last')[[id_col, conv_col]]
        df_path = df_path.merge(df_last_int, how = 'left',
                                               left_on = id_col, right_on = id_col)


        # Добавим начало и конец цепочки
        df_path[channel_col].apply(lambda x: x.insert(0, "Start"))
        df_path.query(f'{conv_col} == 0')[channel_col].apply(lambda x: x.append('Null'))
        df_path.query(f'{conv_col} == 1')[channel_col].apply(lambda x: x.append('Conversion'))

        if self.verbose > 0:
            print("markov_prep is done")

        return df_path

    def transitions(self, df_prep):
        """
        Computes ALL transition points in paths
        """

        if self.verbose > 0:
            print("started transitions")

        path_list = df_prep[self.channel_col]
        unique_channels = self.unique_channels
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

        if self.verbose > 0:
            print("transitions is done")
        return states




    def prob(self, trans_dict):

        """
        Computes probabilities of transitions between ALL possible states.

        Probability defined as (number of transitions to next state) / (number of transitions in current state)
        """

        if self.verbose > 0:
            print("started prob")

        unique_channels = self.unique_channels
        trans_prob = defaultdict(dict)

        # TODO: try to optimize
        for state in unique_channels:
            if state not in ['Conversion', 'Null']:
                cnt = 0
                index = [i for i, s in enumerate(trans_dict) if state + '>' in s]
                for col in index:
                    if trans_dict[list(trans_dict)[col]] > 0:
                        cnt += trans_dict[list(trans_dict)[col]]
                for col in index:
                    if trans_dict[list(trans_dict)[col]] > 0:
                        state_prob = float((trans_dict[list(trans_dict)[col]])) / float(cnt)
                        trans_prob[list(trans_dict)[col]] = state_prob


        if self.verbose > 0:
            print("prob is done")
        return trans_prob


    def make_matrix(self, transition_prob):
        """
        Makes squared n x n matrix, where n is number of unique_channels

        Values of this matrix are probabilities of transition between two states
        Most interesting are transitions to Conversion
        """

        if self.verbose > 0:
            print("started make_matrix")
        matrix = pd.DataFrame()

        for channel in self.unique_channels:
            matrix[channel] = 0.00
            matrix.loc[channel] = 0.00
            matrix.loc[channel][channel] = 1.0 if channel in ['Conversion', 'Null'] else 0.0

        for key, value in transition_prob.items():
            origin, destination = key.split('>')
            matrix.at[origin, destination] = value

        if self.verbose > 0:
            print("make_matrix is done")
        return matrix


    def removal_effect(self, matrix, base_conversion_rate):
        """
        Computes removal effect of each channel in transitions graph
        Miracle of Linear Algebra
        """

        if self.verbose > 0:
            print("started removal_effect")
        removal_effect_dict = {}
        channels = [channel for channel in matrix.columns if channel not in ['Start', 'Null', 'Conversion']]

        # Дропаем определенный канал и считаем сумму оставшихся вероятностей
        # Это и есть наш removal effect
        for channel in channels:
            removal_df = matrix.drop(channel, axis=1).drop(channel, axis=0)
            for column in removal_df.columns:
                row_sum = np.sum(list(removal_df.loc[column]))
                null_pct = float(1) - row_sum
                if null_pct != 0:
                    removal_df.loc[column]['Null'] = null_pct
                removal_df.loc['Null']['Null'] = 1.0

            # Разделяем на пути, ведущие и не ведущие к конверсии
            removal_to_conv = removal_df[
                ['Null', 'Conversion']].drop(['Null', 'Conversion'], axis=0)
            removal_to_null = removal_df.drop(
                ['Null', 'Conversion'], axis=1).drop(['Null', 'Conversion'], axis=0)

            # Пошли чудеса линала, чтобы эффективно посчитать removal effect
            removal_inv_diff = np.linalg.inv(
                np.identity(
                    len(removal_to_null.columns)) - np.asarray(removal_to_null))
            removal_dot = np.dot(removal_inv_diff, np.asarray(removal_to_conv))
            removal_conversion = pd.DataFrame(removal_dot,
                                       index=removal_to_conv.index)[[1]].loc['Start'].values[0]
            removal_effect = 1 - removal_conversion / base_conversion_rate
            removal_effect_dict[channel] = removal_effect

        if self.verbose > 0:
            print("removal_effect is done")
        return removal_effect_dict


    def make_markov(self, total_conversions, base_cr):

        """Final func to make markov-chain attribution model"""

        # Обработанные данные
        df_prep = self.markov_prep()
        # Все переходы
        transitions_dict = self.transitions(df_prep)
        # Вероятности перехода между состояниями
        probs = self.prob(transitions_dict)
        # Матрица
        matrix = self.make_matrix(probs)
        # Посчитали removal effect
        removal = self.removal_effect(matrix, base_cr)


        removal_sum = sum(removal.values())

        return {k: (v / removal_sum) * total_conversions for k, v in removal.items()}





