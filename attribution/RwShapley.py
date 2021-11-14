from itertools import combinations, product
import numpy as np
from collections import defaultdict
import pandas as pd
from tools.exceptions import MissInputData
from tqdm import tqdm


class RwShap:
    def __init__(self, df, channel_col_name, conv_col_name, sep="^"):
        self.df = df
        self.channel_col_name = channel_col_name
        self.conv_col_name = conv_col_name
        self.sep = sep


    @staticmethod
    def comb(l, max_path_len):
        """
        Computes combinations without repetitions for elements in set
        :param l: (iterable) - Initial set
        :param max_path_len: (int) - Maximum cardinality of subset with combinations

        :return: list with combinations
        """
        res = [list(j) for i in range(max_path_len) for j in combinations(l, i+1)]
        return res


    @staticmethod
    def comb_full(vals, max_len=None):
        """
        Finds ALL possible combinations of set with repetitions with the specified length

        **NOTE** Be careful with large sets and max length, because number of possible combinations is n^max_len

        :param vals: (iterable), Unique set of values for combinations
        :param max_len: (int), Maximum length of combination

        :return: list with all possible combinations
        """

        res = []

        for l in range(1, max_len + 1, 1):
            res.extend([p for p in product(vals, repeat=l)])

        return res


    def subs(self, s, with_repetitions):
        """
        Finds ALL subsets of set s
        :param s: (iterable), Initial set
        :param with_repetitions: (bool), Whether subsets can contain repeating values
        :return: list with subsets
        """
        if len(s) == 1:
            return s
        else:
            sub_channels = []
            for i in range(1, len(s) + 1):
                sub_channels.extend(map(list, combinations(s, i)))

        if with_repetitions:
            return list(map(self.sep.join, sub_channels))

        return list(map(self.sep.join, map(sorted, sub_channels)))


    def impact(self, A, C_values, with_repetitions):
        """
        Computes impact of coalition (channel combination)

        :param A: (iterable), Coalition of channels.
        :param C_values: (dictionary), Contains the number of conversions that each subset of channels has given
        """

        subsets_of_A = self.subs(A, with_repetitions=with_repetitions)
        worth_of_A = 0
        for subset in subsets_of_A:
            if subset in C_values:
                worth_of_A += C_values[subset]
        return worth_of_A


    def shapley_value(self, max_path_len=1, with_repetitions=True, channels=None):

        """
        Computes Shapley values
        ***NOTE*** The growth of combinations is exponential 2^(n),
        so it'll work too slow for large amnts of combinations

        :param max_path_len: - maximum length of considered path, can not be greater than number of unique_channels
        :param with_repetitions: (bool), Whether subsets can contain repeating values
        :param channels: (iterable), Set of all channels in paths. If not provided will be found from
        self.df[self.channel_col_name]
        """

        df = self.df

        c_values = df.set_index(self.channel_col_name).to_dict()[self.conv_col_name]

        if channels is None:
            df[self.channel_col_name] = df[self.channel_col_name].apply(lambda x: x if len(x.split(self.sep)) == 1 else np.nan)

        if with_repetitions:
            # Максимальная длина цепочки должна быть < число каналов
            if len(channels) <= max_path_len:
                return "Maximum path length can't be larger than unique channels amnt"

        v_values = {}

        if with_repetitions:
            for A in self.comb_full(channels, max_path_len):
                v_values[self.sep.join(A)] = self.impact(A, c_values, with_repetitions)

        else:
            for A in self.comb(channels, max_path_len):
                v_values[self.sep.join(sorted(A))] = self.impact(A, c_values, with_repetitions)

        n = len(channels)
        shapley_values = defaultdict(int)

        for channel in tqdm(channels):
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


def shap_and_freq(FIC_data, FIC_on, shap_on, sh_clicks=None, sh_impr=None, sh_total=None):
    """
    Combines FIC and Shapley Value attribution

    :param FIC_data: (pandas.DataFrame or tuple of pandas.DataFrames), result of tools.prep.compute_FIC, dataframe with
     FIC for each channel for clicks and impressions or for all interaction types
    :param FIC_on: (str), name of column in FIC DataFrame to merge with shapley data
    :param shap_on: (str), name of column in sh_data DataFrame to merge with shapley data
    :param sh_clicks: (pd.DataFrame, optional, default=None), if FIC_data is tuple with FIC for both impressions and
    clicks data, must be not None, result of tools.prep.compute_FIC for ONLY "Click" interactions
    :param sh_impr: (pd.DataFrame, optional, default=None), if FIC_data is tuple with FIC for both impressions and click
    data, must be not None, result of tools.prep.compute_FIC for ONLY "Impression" interactions
    :param sh_total: (pd.DataFrame, optional, default=None), if FIC_data is pd.DataFrame with FIC for total data,
    must be not None

    :return:
    """

    if isinstance(FIC_data, tuple):
        if sh_clicks is None or sh_impr is None:
            raise MissInputData("When clicks and impressions FIC are provided, sh_clicks and sh_impr must be not None")
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

    elif isinstance(FIC_data, pd.DataFrame):
        if sh_total is None:
            raise MissInputData("When full FIC is provided, sh_total must be not None")

        total = sh_total.merge(FIC_data, left_on=shap_on, right_on=FIC_on)
        total['total_weight'] = total['weight'] * total['FIC']
        total['total_weight'] = total['total_weight'] / total['total_weight'].sum()
        return total[[shap_on, 'total_weight']]
