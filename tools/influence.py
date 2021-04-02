import numpy as np
from tools.exceptions import *
from gensim.models import word2vec



def channels_diff(channel_type, cost_dict, new_cost, mode="fixed", weights=None):
    # TODO: Update docstring
    """
    Первый случай: бесплатные каналы просто не меняются, т.е. внешние факторы(косты, охват и пр.) при изменении цепочек
    остаются такими же как они и есть. Изначально бесплатным каналам присваивается среднее среди всех платных каналов.

    Второй случай: при изменении внешних факторов (например увеличении бюджета на медийные размещения) влиянеие
    внешних факторов на бесплатные каналы меняются в том же направлении, но возможно с меньшей быстротой.
    Изначально бесплатным каналам присваивается среднее среди всех платных каналов

    Третий случай: WORD EMBEDDINGS, KEKW

    :param:
    - channel_type: dict, Dictionary where each channel is matched with one of the following types:
    "FREE", "PAID", "RETARGETING", "MOBILE"

         "FREE": free advertisement i.e. organic, direct, etc.
         "RETARGETING": paid retargeting
         "MOBILE": paid mobile ads
         "PAID": all other paid channels

    :return:
    - initial_cost_dict: dict, distribution of external for ALL(including free channels) before change,
    - new_cost_dict: dict, distribution of external for ALL(including free channels) after change
    """

    def sigmoid(x):
        """Returns sigmoid of x: 2/(1+exp(-0.5x))-1"""
        return 1/(1+np.exp(-0.5*x)-1)

    channels = channel_type.keys()
    new_cost_dict = {}
    n_paid = 0
    sum_paid_new = 0
    sum_paid_old = 0

    try:
        if mode not in ['fixed', 'linear', 'non-linear', 'weighted']:
            raise NonListedValue

        for channel in channels:
            # free channels do not change at all
                if mode == 'fixed':
                    if channel_type[channel] != "FREE":
                        new_cost_dict[channel] = new_cost[channel]
                        sum_paid_old += cost_dict[channel]
                        sum_paid_new += cost_dict[channel]
                        n_paid += 1

                    else:
                        new_cost_dict[channel] = 0
                        cost_dict[channel] = 0

                if mode == 'linear':
                    if channel_type[channel] != "FREE":
                        new_cost_dict[channel] = new_cost[channel]
                        sum_paid_new += new_cost[channel]
                        sum_paid_old += cost_dict[channel]
                        n_paid += 1

                    else:
                        new_cost_dict[channel] = 0
                        cost_dict[channel] = 0

                if mode == 'non-linear':
                    if channel_type[channel] != "FREE":
                        new_cost_dict[channel] = new_cost[channel]
                        sum_paid_new += sigmoid((new_cost[channel] - cost_dict[channel]) / cost_dict[channel]) * cost_dict[channel]
                        sum_paid_old += cost_dict[channel]
                        n_paid += 1

                    else:
                        new_cost_dict[channel] = 0
                        cost_dict[channel] = 0

                if mode == 'weighted':
                    try:
                        if weights is None:
                            raise MissInputData

                        if channel_type[channel] != "FREE":
                            new_cost_dict[channel] = new_cost[channel]
                            sum_paid_new += new_cost[channel] * weights[channel]
                            sum_paid_old += cost_dict[channel]
                            n_paid += 1

                        else:
                            new_cost_dict[channel] = 0
                            cost_dict[channel] = 0

                    except MissInputData:
                        print('When option mode == "weighted" weights can not be None')
                        print()


        new_cost_dict = {channel: v if v != 0 else sum_paid_new/n_paid for channel, v in new_cost_dict.items()}
        initial_cost_dict = {channel: v if v != 0 else sum_paid_old/n_paid for channel, v in cost_dict.items()}

        return initial_cost_dict, new_cost_dict

    except NonListedValue:
        print('parameter mode should be one of the following values: "fixed", "linear", "non-linear", "weighted"')
        print()


def embeddings_similarity(corpus, unique_channels, path_col='path', sep='^'):
    """
    Идея: конверсионные цепочки это тексты, а каналы это слова. Поэтому силу схожести каналов можно определить как
    косинусную схожесть для двух векторов, полученных при помощи эмбеддингов.

    Функция создает эмбеддинги на основе данных цепочек, для каждого канала возвращает список наиболее схожие каналы
    с величиной косинусной схожести. Основана на либе gensim https://github.com/RaRe-Technologies/gensim

    :param:
    - corpus: pandas.DateFrame, dataframe with paths. It is recommended to use prep_data function from tools module

    - unique_channels: iterable, iterable with unique channels in conversion paths

    - path_col: str, optional, name of column with paths

    - sep: str, optional, character to separate channels in paths

    :return: similar_channels: dict, where each channel is matched with the most similar channels and score
    of this similarity. Score takes value from [-1; 1]
    """

    corpus['text'] = corpus[path_col].apply(lambda x: x.split(sep))
    embedding_model = word2vec.Word2Vec(corpus['text'], vector_size=200, epochs=5, window=5, workers=4)
    similar_channels = {}

    for channel in unique_channels:
        similar_channels[channel] = embedding_model.wv.most_similar(channel, topn=5)

    return similar_channels


def linear_change(df, cost_dict, new_cost, conv_col='conversion', path_col='path', sep='^'):
    """
    Simple idea: more money we spend to some channel, more often it appears in conv paths

    :param:
    - df : pandas.DataFrame, DataFrame with paths and their counter, it is recommended to use prep.prep_data first.

    - cost_dict : dict, Dictionary with unique channels and their cost

    - new_cost : dict, Dictionary with new budget distribution

    - conv_col: str, Name of column with conversions/number path amnt

    - path_col: str, Name of column with paths

    - sep: str, Character that used for separation channels in paths

    :returns: df with two new columns: "prob" - probability of particular path occurs,
    conv_col+"new" - path counter with new budget distribution

    """

    channels = cost_dict.keys()
    cost_unit = {}
    cnt = {channel: 0 for channel in channels}

    for path, conv in zip(df[path_col], df[conv_col]):
        for channel in path.split(sep):
            cnt[channel] += conv

    # divide cost to conv
    for channel in channels:
        cost_unit[channel] = cost_dict[channel] / cnt[channel]

    # compute how likely each path with channel occurs
    df['prob'] = df[conv_col] / df[conv_col].sum()

    # new cost is distributed with the same ratio between current paths

    new_conv = {channel: int((new_cost[channel] - cost_dict[channel]) / cost_unit[channel])
                for channel in channels}

    res = []

    for path, conv, prob in zip(df[path_col], df[conv_col], df['prob']):
        conv_new = conv
        for channel in path.split(sep):
            conv_new += prob*new_conv[channel]
            new_conv[channel] -= prob*new_conv[channel]

        res.append(conv_new)

    df[conv_col+'_new'] = res
    df[conv_col+'_new'] = df[conv_col+'_new'].astype('int')
    df = df[df[conv_col+'_new'] >= 0]

    return df


def sigmoid_change(df, cost_dict, new_cost, conv_col='conversion', path_col='path', sep='^',
                   change_rate=2.5, shift=None):
    """
    Main idea: influence of external factors on conversion paths are distributed like sigmoid func
    https://en.wikipedia.org/wiki/Sigmoid_function

    By default we assume that external factors are in the middle of sigmoid func (x=0), this can be changed for each
    channel with shift parameter

    Sigmoid has following form: 2/(1+exp(-change_rate*x))-1

    :param:
    - df : pandas.DataFrame, DataFrame with paths and their counter, it is recommended to use prep.prep_data first.

    - cost_dict : dict, Dictionary with unique channels and their cost

    - new_cost : dict, Dictionary with new budget distribution

    - conv_col: str (default: "conversion"), Name of column with conversions/number path amnt

    - path_col: str (default: "path"), Name of column with paths

    - sep: str, Character that used for separation channels in paths

    - change_rate: float, int, Parameter for sigmoid function

    - shift: dict (default: None), Dictionary with values from interval [-1; 1] for every channel. This values indicate starting point
    at sigma function.

    :return:
    """

    channels = cost_dict.keys()
    cnt = {channel: 0 for channel in channels}

    for path, conv in zip(df[path_col], df[conv_col]):
        for channel in path.split(sep):
            cnt[channel] += conv

    # compute how likely each path with channel occurs
    df['prob'] = df[conv_col] / df[conv_col].sum()

    # normalized cost for sigmoid
    cost_norm = {channel: cost / cost for channel, cost in cost_dict.items()}
    new_cost_norm = {channel: new_cost / cost for channel, new_cost, cost in
                     zip(new_cost.keys(), new_cost.values(), cost_dict.values())}

    cost_delta = {channel: new_cost-cost for channel, cost, new_cost in
                  zip(cost_norm.keys(), cost_norm.values(), new_cost_norm.values())}

    # Custom sigmoid for conversion paths
    def sigmoid(x, shift):
        return 2/(1+np.exp(-change_rate*(x+shift)))-1

    if shift is None:
        shift = {channel: 0 for channel in channels}

    # compute new conversions based on sigmoid function
    new_conv = {channel: cnt[channel]*sigmoid(cost_delta[channel], shift[channel]) for channel in channels}

    res = []

    for path, conv, prob in zip(df[path_col], df[conv_col], df['prob']):
        conv_new = conv
        for channel in path.split(sep):
            conv_new += prob*new_conv[channel]
            new_conv[channel] -= prob*new_conv[channel]

        res.append(conv_new)

    df[conv_col+'_new'] = res
    df[conv_col+'_new'] = df[conv_col+'_new'].astype('int')
    df = df[df[conv_col+'_new'] >= 0]

    return df


# TODO:
def other_dist():
    """
    Предполагаем, что зависимость кол-ва конверсий(цепей) не прямая, а подчиняется какому то распределению
    Соотв на основе этого строим вероятности, семплируем и т.д.
    Должно быть чем то похоже на create_new_chain

    :return:
    """

# TODO:
def create_new_chain():
    """
     Идея: мы считаем в каком месте и с какой частотой появляется канал и на основе таких данных вероятностно
     создаем новые цепи, которые, быть может будут совпадать с имеющимися

     Мб тут стоит использовать марковские цепи:
    """