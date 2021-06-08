from itertools import permutations, combinations, product
import numpy as np
from collections import defaultdict
import pandas as pd
import plotly.express as px


class RwShap():
    def __init__(self, df, channel_col_name, conv_col_name, sep="^"):
        self.df = df
        self.channel_col_name = channel_col_name
        self.conv_col_name = conv_col_name
        self.sep = sep


    # Считаем все возможные комбинации БЕЗ ПОВТОРОВ
    def comb(self, l):
        res = [list(j) for i in range(len(l)) for j in combinations(l, i+1)]
        return res


    # Считаем все возможные комбинации С ПОВТОРАМИ
    def comb_full(self, vals, max_len = None):

        """
        Finds ALL possible combinations of set with repetitions with the specified length

        **NOTE** Be careful with large sets and max length, because number of possible combinations is n^max_len

        Inputs:
        - vals (iterable) - unique set of values for combinations
        - max_len (int) - maximum length of combination


        Outputs: list with all possible combinations
        """

        res = []

        for l in range(1, max_len + 1, 1):
            res.extend([p for p in product(vals, repeat=l)])

        return res


    # Все возможные подмножества
    def subs(self, s, with_repetitions):
        if len(s) == 1:
            return s
        else:
            sub_channels = []
            for i in range(1, len(s) + 1):
                sub_channels.extend(map(list, combinations(s, i)))

        if with_repetitions:
            return list(map(self.sep.join, sub_channels))

        return list(map(self.sep.join, map(sorted, sub_channels)))


    # Вклад
    def impact(self, A, C_values, with_repetitions):
        '''
        Computes impact of coalition (channel combination)

        Input:
        - A (iterable) : coalition of channels.
        - C_values (dictionary): containins the number of conversions that each subset of channels has given
        '''
        subsets_of_A = self.subs(A, with_repetitions=with_repetitions)
        worth_of_A = 0
        for subset in subsets_of_A:
            if subset in C_values:
                worth_of_A += C_values[subset]
        return worth_of_A


    # Считаем вектор Шэпли
    def shapley_value(self, max_path_len=1, with_repetitions=True, channels=None, plot=True):

        # TODO: change docstring
        '''
        Calculates shapley values:
        Input:
        - df (pandas.DataFrame): A dataframe with the two columns: channels(path) and conversion amnt

        - col_name: A string that is the name of the column with conversions
                ***NOTE*** Channels should be sorted alphabetically. In this analysis path "Google, Email" is the same as "Email, Google"
                Thus they should be combined in "Google, Email"

                ***NOTE*** The growth of combinations is exponential 2^(n), so it'll work too slow for large amnts of combinations
        '''

        df = self.df

        c_values = df.set_index(self.channel_col_name).to_dict()[self.conv_col_name]
        # Test feature
        # if fic is not None:
        #     for k, conv in c_values.items():
        #         path = k.split(self.sep)
        #         for k_, w in fic.items():

        if channels is None:
            df['channels'] = df[self.channel_col_name].apply(lambda x: x if len(x.split(self.sep)) == 1 else np.nan)

        if with_repetitions:
            # Максимальная длина цепочки должна быть < число каналов
            if len(channels) <= max_path_len:
                return "Maximum path length can't be larger than unique channels amnt"

        v_values = {}

        if with_repetitions:
            for A in self.comb_full(channels, max_path_len):
                v_values[self.sep.join(A)] = self.impact(A, c_values, with_repetitions)

        else:
            for A in self.comb(channels):
                v_values[self.sep.join(sorted(A))] = self.impact(A, c_values, with_repetitions)


        n = len(channels)
        shapley_values = defaultdict(int)

        for channel in channels:
            for A in v_values.keys():

                if channel not in A.split(self.sep):

                    cardinal_A = len(A.split(self.sep))
                    A_with_channel = A.split(self.sep)

                    if with_repetitions and cardinal_A < max_path_len:
                        A_with_channel.append(channel)

                    A_with_channel = self.sep.join(A_with_channel)

                    # Weight = |S|!(n-|S|-1)!/n!
                    weight = (np.math.factorial(cardinal_A) *
                              np.math.factorial(n-cardinal_A - 1) / np.math.factorial(n))

                    # Marginal contribution = v(S U {i})-v(S)
                    contrib = (v_values[A_with_channel] - v_values[A])
                    shapley_values[channel] += weight * contrib

            # Add the term corresponding to the empty set
            shapley_values[channel] += v_values[channel] / n

        sh_df = (
            pd.DataFrame(data=shapley_values.values(), index=shapley_values.keys())
            .reset_index()
            # TODO : change index : 'channel_group' to smth general
            .rename(columns={'index': 'channel_group', 0: 'weight'})
                )
        sh_df['weight_rel'] = sh_df['weight'] / sh_df['weight'].sum()

        return sh_df

def shap_and_freq(sh_clicks, sh_impr, FIC_data, FIC_on, shap_on):
    """
    Combines FIC and Shapley Value attribution

    :param sh_clicks: (pd.DataFrame, optional, default=None), if separated=True must be specified, result
    of tools.prep.compute_FIC for ONLY "Click" interactions

    :param sh_impr: (pd.DataFrame, optional, default=None), if separated=True must be specified, result
    of tools.prep.compute_FIC for ONLY "Impression" interactions

    :param FIC_data: (tuple of pandas.DataFrames), result of tools.prep.compute_FIC, dataframe with FIC for each channel
    for clicks and impressions

    :param FIC_on: (str), name of column in FIC DataFrame to merge with shapley data

    :param shap_on: (str), name of column in sh_data DataFrame to merge with shapley data


    :return:
    """

    freq_c, freq_i = FIC_data
    try:
        if len(freq_c) > 0:
            click = sh_clicks.merge(freq_c, left_on=shap_on, right_on=FIC_on)
        else:
            click = pd.DataFrame()
        if len(freq_i) > 0:
            view = sh_impr.merge(freq_i, left_on=shap_on, right_on=FIC_on)
        else:
            view = pd.DataFrame()

        if len(freq_c) > 0 and len(freq_i) > 0:
            total = (click
                     .merge(view, how='outer', left_on=shap_on, right_on=shap_on)
                     .fillna(0)
                     )
        elif len(freq_c) > 0 and len(freq_i) == 0:
            click['total_weight'] = click['weight'] * click['FIC']
            click['total_weight'] = click['total_weight'] / click['total_weight'].sum()
            return click[[shap_on, 'total_weight']]

        elif len(freq_c) == 0 and len(freq_i) > 0:
            view['total_weight'] = view['weight'] * view['FIC']
            view['total_weight'] = view['total_weight'] / view['total_weight'].sum()
            return view[[shap_on, 'total_weight']]

    except AttributeError:
        print('Input has incorrect data type')
        raise

    total['total_weight'] = total['weight_x'] * total['FIC_x'] + total['weight_y'] * total['FIC_y']

    total['total_weight'] = total['total_weight'] / total['total_weight'].sum()

    return total[[shap_on, 'total_weight']]




