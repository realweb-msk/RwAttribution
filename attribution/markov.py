from collections import Counter
import pandas as pd
import numpy as np
from collections import defaultdict
from tools.exceptions import MissInputData
from tqdm import tqdm


class RwMarkov:
    """
    Class for markov chain attribution model
    """

    def __init__(self, df, channel_col, conv_col, id_col=None, order_col=None, conv_cnt=None, cm_full_path=False,
                 verbose=0, unique_channels=None, drop_direct=False, direct_name=None):
        """
        :param df: (pd.DataFrame), dataframe with paths You can find expected input data format here:
        https://github.com/realweb-msk/RwAttribution#readme
        :param channel_col: (str), name of column with channel(or path) data
        :param conv_col: (str), name of column with conversion flag
        :param id_col: (str, optional, default=None), name of column with some id (clientId, conversionId, etc.).
         Must be specified when cm_full_path=False
        :param order_col: (str, optional, default=None), name of column with order
         (time, precomputed interaction number, etc.) Must be specified when cm_full_path=False
        :param conv_cnt: (str, optional, default=None), name of column with total number of path's occurrences.
         Must be specified when cm_fill_path=True
        :param cm_full_path: (bool, optional, default=False), whether to preprocess and compute transitions
         based on Campaign Manager Full paths report. You can find an example of SQL query in sql.cm_full_path_prep
        :param verbose: (int), when verbose > 1 print progress
        :param unique_channels: (iterable), list(or other iterable) of channels in paths. Must be not None when
         cm_full_path=True
        :param drop_direct: (bool, optional, default=False), if True then drops "Direct" from touchpoints with any other
            touchpoint
        :param direct_name: (str, optional, default=None), name of "Direct" touchpoint in your paths,
            must be not None when null_direct=True
        """
        self.df = df
        self.channel_col = channel_col
        if cm_full_path:
            if unique_channels is not None:
                self.unique_channels = np.append(unique_channels, ['Start', 'Conversion', 'Null'])
            else:
                print("When cm_full_path=True, unique_channels must be not None")
                raise MissInputData
        else:
            self.unique_channels = np.append(df[channel_col].unique(), ['Start', 'Conversion', 'Null'])
        self.id_col = id_col
        self.order_col = order_col
        self.conv_col = conv_col
        self.conv_cnt = conv_cnt
        self.cm_full_path = cm_full_path
        self.verbose = verbose
        self.drop_direct = drop_direct
        if self.drop_direct:
            if self.direct_name is None:
                print("When drop_direct=True, direct_name must be not None")
                raise MissInputData
            else:
                self.direct_name = direct_name

    @staticmethod
    def dropper(path, sep, direct_name, mode='string'):

        if mode == 'string':
            replace_strs = [sep+direct_name, direct_name+sep]
            if direct_name in path and len(Counter(path.split(sep))) > 1:
                new_path = str.replace(path, replace_strs[0], '')
                new_path = str.replace(new_path, replace_strs[1], '')

                return new_path

        elif mode == 'list':
            if direct_name in path and len(Counter(path)) > 1:
                for i in range(Counter(path)[direct_name]):
                    path.remove(direct_name)

                return path

        return path

    def markov_prep(self, sep="^"):
        """
        Preprocess data for markov chains with respect for cm_full_path
        :param sep: (str, optional, default='^'), character that separates touchpoints in paths
        :raises MissInputData: when one of the *args is not defined
        """

        if self.verbose > 0:
            print("Started markov_prep")

        if self.cm_full_path:
            if self.conv_cnt is None:
                print("When cm_full_path is True conv_cnt must be provided")
                raise MissInputData

            channel_col = self.channel_col
            conv_col = self.conv_col
            df_path = self.df.copy()

            if self.drop_direct:
                print('was here')
                df_path[channel_col] = df_path[channel_col].apply(lambda x: self.dropper(x, sep, self.direct_name))
                return df_path
            df_path[channel_col] = df_path[channel_col].apply(lambda x: x.split(sep))



            df_path[channel_col].apply(lambda x: x.insert(0, "Start"))
            df_path.query(f'{conv_col} == 0')[channel_col].apply(lambda x: x.append('Null'))
            df_path.query(f'{conv_col} == 1')[channel_col].apply(lambda x: x.append('Conversion'))

        else:
            if self.id_col is None and self.order_col is None:
                print("When cm_full_path is False id_col and order_col must be provided")
                raise MissInputData
            id_col = self.id_col
            order_col = self.order_col
            channel_col = self.channel_col
            conv_col = self.conv_col

            df = self.df.sort_values([id_col, order_col])

            # Нужно для тестовых данных
            df['interaction_number'] = df.groupby(id_col).cumcount() + 1
            # return df
            df_path = df.groupby(id_col)[channel_col].aggregate(lambda x: x.tolist()).reset_index()

            df_last_int = df.drop_duplicates(id_col, keep='last')[[id_col, conv_col]]
            df_path = df_path.merge(df_last_int, how='left',
                                                   left_on=id_col, right_on = id_col)

            if self.drop_direct:
                df_path[channel_col].apply(lambda x: self.dropper(x, sep, self.direct_name, mode='list'))

            # Добавим начало и конец цепочки
            df_path[channel_col].apply(lambda x: x.insert(0, "Start"))
            df_path.query(f'{conv_col} == 0')[channel_col].apply(lambda x: x.append('Null'))
            df_path.query(f'{conv_col} == 1')[channel_col].apply(lambda x: x.append('Conversion'))

        if self.verbose > 0:
            print("markov_prep is done")

        return df_path


    def transitions(self, df_prep):
        """
        Computes ALL transition points in paths with respect to cm_full_path
        :param df_prep: result of self.markov_prep method
        """

        if self.verbose > 0:
            print("started transitions")

        path_list = df_prep[self.channel_col]
        conv_list = df_prep[self.conv_cnt] if self.cm_full_path else None
        unique_channels = self.unique_channels
        states = {x + '>' + y: 0 for x in unique_channels for y in unique_channels}

        # Ugly cycles
        # TODO: Try to optimize somehow
        for state in tqdm(unique_channels):
            if state not in ('Conversion', 'Null'):
                # Делим на предобработку для обычных данных и для данных из cm_full_path
                if self.cm_full_path:
                    for current_path, current_conv in zip(path_list, conv_list):
                        if state in current_path:
                            indices = (i for i, s in enumerate(current_path) if state in s)
                            for col in indices:
                                states[current_path[col] + '>' + current_path[col + 1]] += current_conv

                else:
                    for current_path in path_list:
                        if state in current_path:
                            indices = (i for i, s in enumerate(current_path) if state in s)
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
        for state in tqdm(unique_channels):
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

    # TODO: Update docstring
    def calc_conversions(self, prep):
        """
        Computes basic conversion rate based on prepared data with

        :param prep: result of self.markov_prep method
        """

        try:
            if not self.cm_full_path:
                total_conversions = sum(path.count('Conversion') for path in prep[self.channel_col].tolist())
                return total_conversions / len(prep[self.id_col])

            total_conversions = prep[prep[self.conv_col] == 1][self.conv_cnt].sum()
            return total_conversions / prep[self.conv_cnt].sum()

        except ZeroDivisionError:
            print("Input data is empty! Division by zero error")

    # TODO: Update docstring
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


    def make_markov(self):
        """Final method to make markov-chain attribution model"""

        # Обработанные данные
        df_prep = self.markov_prep()
        # Все переходы
        transitions_dict = self.transitions(df_prep)
        # Вероятности перехода между состояниями
        probs = self.prob(transitions_dict)
        # Матрица
        matrix = self.make_matrix(probs)
        # Считаем базовый CR
        base_conv_rate = self.calc_conversions(df_prep)
        if self.verbose > 0:
            print("Base conversion rate:", base_conv_rate)
        # Посчитали removal effect
        removal = self.removal_effect(matrix, base_conv_rate)

        removal_sum = sum(removal.values())
        return {k: (v / removal_sum) for k, v in removal.items()}





