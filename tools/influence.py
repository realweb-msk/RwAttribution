import numpy as np
from tools.exceptions import *
from gensim.models import word2vec


def channels_diff(channel_type, cost_dict, new_cost, mode="fixed", weights=None):
    """
    Function for calculation how changing cost, reach, etc. in some channels will affect other

    :param channel_type: (dict), Dictionary where each channel is matched with one of the following types:
    "FREE", "PAID CONTEXT", "RETARGETING", "MOBILE", "PAID SOCIALS", "PAID DISPLAY", "OTHER"

         "FREE": free advertisement i.e. organic, direct, etc.
         "RETARGETING": paid retargeting
         "MOBILE": paid mobile ads
         "PAID CONTEXT": paid context ads
         "PAID SOCIALS": paid social networks
         "PAID DISPlAY": paid display ads
         "OTHER": other ad types

    :param cost_dict: (dict), Dictionary with non-FREE channels and their initial cost
    (or some other changing parameter, e.g. reach)
    :param new_cost: (dict), Dictionary with non-FREE channels and their cost (or some other changing parameter,
     e.g. reach) in the future, e.g. after budget optimisation
    :param mode: (str, optional, default="fixed"), Mode, which will be used for computing new influence.
    Should be one of the following values:
    - "fixed": FREE channels do not change at all, their initial and after cost is avg of non-FREE channels
    - "linear": FREE channels change linear to corresponding changes in non-FREE channels, e.g. if
    non-FREE channels cost increased by 10%, FREE channel after cost will be increased by 10%
    - "non-linear": similar to "linear", but cost growth is defined by sigmoid function, not linear
    - "weighted_free": After cost of FREE channels computed as weighted average of non-FREE channels
    - "weighted_full": ALL channel's after cost is dependent on weights
    :param weights: (optional, default=None), Weights dict. Must be not None when mode is set to
    "weighted_free" or "wighted_full".
    When mode="weighted_free" weights dict expected to be in the following format:
            {'some_channel': 0.123, # 0.123 and 0.42 are channel weights
             'other_channel': 0.42
            }

    When mode="weighted_full" weights dict expected to be in the following format:
            # In this case input wights should be in format of dict which values are also dicts
            {'some_channel': {'channel_A: 0.123, 'channel_B': 0.42},
             'other_channel': {'channel_C': 0.321, 'channel_A': 0.12}
            }

    :return: - initial_cost_dict: dict, distribution of external for ALL(including free channels) before change,
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

        if mode not in ('fixed', 'linear', 'non-linear', 'weighted_free', 'weighted_full'):
            raise ValueError("mode must be one of 'fixed', 'linear', 'non-linear', 'weighted_free', 'weighted_full',",
                             f"got {mode}")

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

            elif mode == 'linear':
                if channel_type[channel] != "FREE":
                    new_cost_dict[channel] = new_cost[channel]
                    sum_paid_new += new_cost[channel]
                    sum_paid_old += cost_dict[channel]
                    n_paid += 1

                else:
                    new_cost_dict[channel] = 0
                    cost_dict[channel] = 0

            elif mode == 'non-linear':
                if channel_type[channel] != "FREE":
                    new_cost_dict[channel] = new_cost[channel]
                    sum_paid_new += sigmoid((new_cost[channel] - cost_dict[channel]) / cost_dict[channel]) * cost_dict[channel]
                    sum_paid_old += cost_dict[channel]
                    n_paid += 1

                else:
                    new_cost_dict[channel] = 0
                    cost_dict[channel] = 0

            if mode == 'weighted_free':
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

            elif mode == 'weighted_full':
                if weights is None:
                    raise MissInputData
                    # In this case input wights should be in format of dict which values are also dicts
                if channel in weights:
                    for k, v in weights.items():
                        # In case FREE channel is in weights dictionary
                        try:
                            new_cost_dict[k] = cost_dict[k]
                        finally:
                            new_cost_dict[k] = 0

                        for k_, v_ in v.items():
                            # If weight is negative and cost are decreasing, we assume there is no change
                            if not (v_ < 0 and new_cost[k_] < cost_dict[k_]):
                                new_cost_dict[k] += (new_cost[k_] - cost_dict[k_]) * v_
                else:
                    if channel_type[channel] != 'FREE':
                        new_cost_dict[channel] = new_cost[channel]
                        sum_paid_new += new_cost[channel]
                        sum_paid_old += cost_dict[channel]
                        n_paid += 1

                    else:
                        new_cost_dict[channel] = 0
                        cost_dict[channel] = 0

        new_cost_dict = {channel: v if v != 0 else sum_paid_new/n_paid for channel, v in new_cost_dict.items()}
        initial_cost_dict = {channel: v if v != 0 else sum_paid_old/n_paid for channel, v in cost_dict.items()}

        return initial_cost_dict, new_cost_dict

    except NonListedValue:
        print('parameter mode should be one of the following values: "fixed", "linear", "non-linear", "weighted"')
        print()
        raise NonListedValue

    except MissInputData:
        print('When option mode == "weighted_free" weights can not be None')
        print()
        raise MissInputData


def embeddings_similarity(corpus, unique_channels, w2v=None, top_n=None, path_col='path', sep='^', **kwargs):
    """
    Обучаем эмбеддинги на основе последовательности каналов в конверсионных цепочках. Полученные вектора лежат на
    единичной гиперсфере, поэтому схожесть каналов можно определить через косинусную меру.

    Функция создает эмбеддинги на основе данных цепочек, для каждого канала возвращает список наиболее схожие каналы
    с величиной косинусной схожести. Основана на либе gensim https://github.com/RaRe-Technologies/gensim

    :param corpus: pandas.DateFrame, dataframe with paths. It is recommended to use prep_data function from tools module
    :param unique_channels: iterable, iterable with unique channels in conversion paths
    :param w2v: optional, pretrained w2v.Word2Vec model
    :param top_n: optional, number of most similar channels to return
    :param path_col: str, optional, name of column with paths
    :param sep: str, optional, character to separate channels in paths
    :return: similar_channels: dict, where each channel is matched with the most similar channels and score
    of this similarity. Score takes value from [-1; 1]
    """

    corpus['text'] = corpus[path_col].apply(lambda x: x.split(sep))
    if w2v is None:
        embedding_model = word2vec.Word2Vec(corpus['text'], **kwargs)

    else:
        try:
            assert isinstance(w2v, word2vec.Word2Vec)
            embedding_model = word2vec.Word2Vec.load(w2v)

        except AssertionError:
            raise AssertionError(f"w2v model must be pretrained gensim.word2vec.Word2Vec model, got {type(w2v)} obj")

    similar_channels = {}

    if top_n is None:
        top_n = len(unique_channels )

    for channel in unique_channels:
        similar_channels[channel] = embedding_model.wv.most_similar(channel, topn=top_n)

    return similar_channels, embedding_model


def linear_change(df, cost_dict, new_cost, conv_col='conversion', path_col='path', sep='^'):
    """
    Simple idea: more money we spent on channel, more likely it will occur in conversion paths.
    Простая идея, чем больше мы изменяем влияние (например расходуем больше денег) на какой либо канал, тем чаще
    он появляется в конверсионных цепочках

    :param df : pandas.DataFrame, DataFrame with paths and their counter, it is recommended to use prep.prep_data first.
    :param cost_dict : dict, Dictionary with unique channels and their cost
    :param new_cost : dict, Dictionary with new budget distribution
    :param conv_col: str, Name of column with conversions/number path amnt
    :param path_col: str, Name of column with paths
    :param sep: str, Character that used for separation channels in paths

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

    :param df: pandas.DataFrame, DataFrame with paths and their counter, it is recommended to use prep.prep_data first.
    :param cost_dict: dict, Dictionary with unique channels and their cost
    :param new_cost: dict, Dictionary with new budget distribution
    :param conv_col: str (default: "conversion"), Name of column with conversions/number path amnt
    :param path_col: str (default: "path"), Name of column with paths
    :param sep: str, Character that used for separation channels in paths
    :param change_rate: float, int, Parameter for sigmoid function
    :param shift: dict (default: None), Dictionary with values from interval [-1; 1] for every channel.
    This values indicate starting point at sigma function.

    :return: pandas.DataFrame
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
