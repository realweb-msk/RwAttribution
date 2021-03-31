import pandas as pd
import numpy as np


def linear_change(df, cost_dict, new_cost, conv_col='conversion', path_col='path', sep='^'):

    """
    Simple idea: more money we spend to some channel, more often it appears in conv paths

    :param:
    df : pandas.DataFrame, DataFrame with paths and their counter, it is recommended to use prep.prep_data first.

    cost_dict : dict, Dictionary with unique channels and their cost

    new_cost : dict, Dictionary with new budget distribution

    conv_col: str, Name of column with conversions/number path amnt

    path_col: str, Name of column with paths

    sep: str, Character that used for separation channels in paths

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

# TODO: Добавить сдвиг ачального значения относительно нуля
def sigmoid_change(df, cost_dict, new_cost, conv_col='conversion', path_col='path', sep='^',
                   change_rate=2.5, sup=1, inf=0):
    """
    Main idea: influence of external factors on conversion paths are distributed like sigmoid func
    https://en.wikipedia.org/wiki/Sigmoid_function

    By default we assume that external factors are in the middle of sigmoid func (x=0)

    Sigmoid has following form: 2/(1+exp(-change_rate*x))-1

    :param:
    df : pandas.DataFrame, DataFrame with paths and their counter, it is recommended to use prep.prep_data first.

    cost_dict : dict, Dictionary with unique channels and their cost

    new_cost : dict, Dictionary with new budget distribution

    conv_col: str, Name of column with conversions/number path amnt

    path_col: str, Name of column with paths

    sep: str, Character that used for separation channels in paths

    change_rate: float, int, Parameter for sigmoid function


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
    def sigmoid(x):
        return 2/(1+np.exp(-change_rate*x))-1

    new_conv = {channel: cnt[channel]*sigmoid(cost_delta[channel]) for channel in channels}

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