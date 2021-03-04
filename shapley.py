from itertools import permutations, combinations, product
import numpy as np
from collections import defaultdict
import pandas as pd


class RwShap():
    def __init__(self, df, channel_col_name, conv_col_name):
        self.df = df
        self.channel_col_name = channel_col_name
        self.conv_col_name = conv_col_name


    # Считаем все возможные комбинации БЕЗ ПОВТОРОВ
    def comb(self, l):
        res = [list(j) for i in range(len(l)) for j in combinations(l, i+1)]
        return res


    # Считаем все возможные комбинации С ПОВТОРАМИ
    def comb_full(self, vals, max_len = None):

        """
        Finds ALL possible combinations of set with repitations with the specified length

        **NOTE** Be carefull with large sets and max length, because number of possible combinations is n^max_len

        Inputs:
        - vals (iterable) - unique set of values for combinations
        - max_len (int) - maximum length of combination


        Outputs: list with all possible combinations
        """

        res = []

        for l in range(1, max_len + 1, 1):
            res.extend([p for p in product(vals, repeat = l)])

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
            return list(map("_".join, sub_channels))

        return list(map("_".join, map(sorted, sub_channels)))




    # Вклад
    def impact(self, A, C_values, with_repetitions):
        '''
        Computes impact of coalition (channel combination)

        Input:
        - A (iterable) : coalition of channels.
        - C_values (dictionary): containins the number of conversions that each subset of channels has given
        '''
        subsets_of_A = self.subs(A, with_repetitions = with_repetitions)
        worth_of_A = 0
        for subset in subsets_of_A:
            if subset in C_values:
                worth_of_A += C_values[subset]
        return worth_of_A



    # Считаем вектор Шэпли
    def shapley_value(self, max_path_len = 1, with_repetitions = True, channels = None):

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

        if channels is None:
            df['channels'] = df[self.channel_col_name].apply(lambda x: x if len(x.split("_")) == 1 else np.nan)
            channels = list(df['channels'].dropna().unique())

        else:
            channels = channels

        #print(channels)


        # В наших цепочках каналы могут повторяться
        if with_repetitions:

            # Максимальная длина цепочки должна быть < число каналов
            if len(channels) <= max_path_len:
                return "Maximum path length can't be larger than unique channels amnt"

            v_values = {}

            for A in self.comb_full(channels, max_path_len):
                #print(A)
                v_values['_'.join(A)] = self.impact(A, c_values, with_repetitions)


            #print(v_values)

            n = len(channels)
            #print(n)
            shapley_values = defaultdict(int)

            for channel in channels:
                for A in v_values.keys():
                    #print(A)

                    if channel not in A.split("_"):
                        #print(channel)

                        cardinal_A = len(A.split("_"))
                        A_with_channel = A.split("_")

                        if cardinal_A < max_path_len:
                            A_with_channel.append(channel)
                        #print(A_with_channel)

                        #A_with_channel = "_".join(sorted(A_with_channel))
                        #print(A_with_channel)
                        A_with_channel = "_".join(A_with_channel)

                        # Weight = |S|!(n-|S|-1)!/n!
                        weight = (np.math.factorial(cardinal_A) *
                                  np.math.factorial(n-cardinal_A - 1) / np.math.factorial(n))

                        # Marginal contribution = v(S U {i})-v(S)
                        contrib = (v_values[A_with_channel] - v_values[A])
                        shapley_values[channel] += weight * contrib

                # Add the term corresponding to the empty set
                shapley_values[channel] += v_values[channel] / n

            sh_df = (
                pd.DataFrame(data = shapley_values.values(), index = shapley_values.keys())
                .reset_index()
                .rename(columns = {'index' : 'channel_grouping', 0 : 'weight'})
                    )
            sh_df['weight'] = sh_df['weight'] / sh_df['weight'].sum()

            return(sh_df)


        # Берем только уникальные каналы, причем последовательность каналов
        # Должна быть отсортирована в алфавитном порядке
        else:
            v_values = {}

            for A in self.comb(channels):
                #print(A)
                v_values['_'.join(sorted(A))] = self.impact(A, c_values, with_repetitions)



            n = len(channels)
            shapley_values = defaultdict(int)


            for channel in channels:
                for A in v_values.keys():
                    #print(A)

                    if channel not in A.split("_"):
                        #print(channel)

                        cardinal_A = len(A.split("_"))
                        A_with_channel = A.split("_")
                        A_with_channel = "_".join(sorted(A_with_channel))
                        #print(A_with_channel)

                        # Weight = |S|!(n-|S|-1)!/n!
                        weight = (np.math.factorial(cardinal_A) *
                                  np.math.factorial(n-cardinal_A - 1) / np.math.factorial(n))

                        # Marginal contribution = v(S U {i})-v(S)
                        contrib = (v_values[A_with_channel] - v_values[A])
                        shapley_values[channel] += weight * contrib

                # Add the term corresponding to the empty set
                shapley_values[channel] += v_values[channel] / n

            sh_df = (
                pd.DataFrame(data = shapley_values.values(), index = shapley_values.keys())
                .reset_index()
                .rename(columns = {'index' : 'channel_grouping', 0 : 'weight'})
                    )

            sh_df['weight'] = sh_df['weight'] / sh_df['weight'].sum()

            return sh_df



