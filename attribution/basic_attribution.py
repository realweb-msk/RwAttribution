from collections import Counter
import plotly.express as px
from tools.prep import dict_to_frame


def last_click(df, path_col='path', conv_col='conversion', sep='^', plot=True):
    """
    Preforms last-click attribution model

    :param df: (pd.DataFrame), dataframe ONLY with clicks data, if tools.prep_data was used then
     it is "click_gr" dataframe
    :param path_col: (str, optional default='path'), name of column with path data
    :param conv_col: (str, optional default='conversion'), name of column with conversion data
    :param sep: (str, optional default='^'), channel separator used in paths
    :param plot: (bool, default = True), whether to plot the data. Plot will contain normalized data

    :return:
    pandas.DataFrame with channels an their attribution score (non-normalized)
    """
    df_ = df.copy()
    new_name = path_col+'_new'
    df_[new_name] = df_[path_col].apply(lambda x: x.split(sep)[-1])

    plt_data = df_.groupby(new_name, as_index=False).agg({conv_col : 'sum'})
    plt_data[conv_col+'_rel'] = plt_data[conv_col] / plt_data[conv_col].sum()

    if plot:
        fig = px.bar(plt_data, y=new_name, x=conv_col+'_rel', title='Last Click Model', orientation='h')
        fig.update_xaxes(title_text='Доля от всех конверсий')
        fig.update_yaxes(title_text='Группа каналов')
        fig.show()

    return plt_data


def last_non_direct_click(df, path_col='path', conv_col='conversion', sep='^', direct_name=None, plot=True):
    """
    Parforms LNDC Attribution model

    :param df: (pd.DataFrame), dataframe ONLY with clicks data, if tools.prep_data was used then
     it is "click_gr" dataframe
    :param path_col: (str, optional default='path'), name of column with path data
    :param conv_col: (str, optional default='conversion'), name of column with conversion data
    :param sep: (str, optional default='^'), channel separator used in paths
    :param direct_name: (str, optional default=None), name of direct channel in paths.
    If not None the following transformation will be applied to paths:
    For every path, if there are any channel but direct_name, direct_name will be deleted from path
    :param plot: (bool, default = True), whether to plot the data. Plot will contain normalized data
    :return:
    """

    def non_direct_index(path_list, direct_name=direct_name):
        # reverse path list cause we are looking from last interaction
        path_list.reverse()
        idx = next((x[0] for x in enumerate(path_list) if x[1] != direct_name), None)
        if idx is None:
            return direct_name

        return path_list[idx]

    new_name = path_col + '_new'
    df_ = df.copy()
    if direct_name is not None:
        df_[new_name] = df[path_col]\
            .apply(lambda x: x.split(sep))\
            .apply(lambda x: non_direct_index(x, direct_name))

    else:
        df_[new_name] = df[path_col]\
            .apply(lambda x: x.split(sep))

    plt_data = df_.groupby(new_name, as_index=False).agg({conv_col: 'sum'})

    if plot:
        fig = px.bar(plt_data, y=new_name, x=conv_col, title='Last non-direct click Model', orientation='h')
        fig.update_xaxes(title_text='Доля от всех конверсий')
        fig.update_yaxes(title_text='Группа каналов')
        fig.show()

    return plt_data


def first_click(df, path_col='path', conv_col='conversion', sep='^', plot=True):
    """
    Preforms first-click attribution model

    :param df: (pd.DataFrame) dataframe ONLY with clicks data, if prep_data was used then it is "click_gr" dataframe
    :param path_col: (str, optional default = 'path') name of column with path data
    :param conv_col: (str, optional default = 'conversion') name of column with conversion data
    :param sep: (str, optional default='^'), channel separator used in paths
    :param plot: (bool, optional default = True) whether to plot the data. Plot will contain normalized data

    :returns
    - pandas.DataFrame with channels an their attribution score (not-normalized)
    """

    df_ = df.copy()
    new_name = path_col + '_new'
    df_[new_name] = df_[path_col].apply(lambda x: x.split(sep)[0])

    plt_data = df_.groupby(new_name, as_index = False).agg({conv_col : 'sum'})
    plt_data[conv_col+'_rel'] = plt_data[conv_col] / plt_data[conv_col].sum()

    if plot:
        fig = px.bar(plt_data, y=new_name, x=conv_col+'_rel', title='First click model', orientation='h')
        fig.update_xaxes(title_text='Доля от всех конверсий')
        fig.update_yaxes(title_text='Группа каналов')
        fig.show()

    return plt_data


def uniform(df, unique_channels, path_col='path', conv_col='conversion', plot=True, as_frame=False,
            keys_col_name=None, values_col_name=None):
    """
    Preforms last-click attribution model

    :param df: (pd.DataFrame) dataframe with all data, if prep_data was used then it is "full_gr" dataframe
    :param unique_channels: (iterable) list(or other iterable) of channels in paths
    :param path_col: (str, default='path') name of column with path data
    :param conv_col: (str, default='conversion') name of column with conversion data
    :param plot: (bool, default=True) whether to plot the data. Plot will contain normalized data
    :param as_frame: (bool, optional, default=False) whether to return data as pandas.DataFrame
    :param keys_col_name: (str, optional, default=None) must be specified if as_frame is set to True
    :param values_col_name: (str, optional, default=None) must be specified if as_frame is set to True
    
    :returns
    - dictionary with channels an their attribution score (not-normalized) if as_frame=False
    - pandas.DataFrame with channels an their attribution score (not-normalized) if as_frame=True
    """
    df_ = df.copy()
    d = {}
    for channel in unique_channels:
        df_[channel] = df_[path_col].apply(lambda x: channel in x)

        d[channel] = df_[df_[channel]][conv_col].sum()

    d_ = {k: v/sum(d.values()) for k, v in zip(d.keys(), d.values())}

    if plot:
        fig = px.bar(y = d_.keys(), x=d_.values(), title='Uniform model', orientation='h')
        fig.update_xaxes(title_text = 'Доля от всех конверсий')
        fig.update_yaxes(title_text = 'Группа каналов')
        fig.show()

    if as_frame:
        if keys_col_name is not None and values_col_name is not None:
            return dict_to_frame(d, keys_col_name, values_col_name)
        else:
            keys_col_name = path_col + '_new'
            values_col_name = conv_col
            return dict_to_frame(d, keys_col_name, values_col_name)

    return d


def linear(df, path_col='path', conv_col='conversion', plot=True, as_frame=False,
            keys_col_name=None, values_col_name=None, sep='^'):

    """
    Performs linear attribution model

    :param df: (pd.DataFrame) dataframe with all data, if prep_data was used then it is "full_gr" dataframe
    :param path_col: (str, default='path') name of column with path data
    :param conv_col: (str, default='conversion') name of column with conversion data
    :param plot: (bool, default=True) whether to plot the data. Plot will contain normalized data
    :param as_frame: (bool, optional, default=False) whether to return data as pandas.DataFrame
    :param keys_col_name: (str, optional, default=None) must be specified if as_frame is set to True
    :param values_col_name: (str, optional, default=None) must be specified if as_frame is set to True
    :param sep: (str, optional default='^'), channel separator used in paths
    """
    d = {}
    for row in df.iterrows():
        cnt = Counter(row[1][path_col].split(sep)).keys()
        for channel in cnt:
            if channel in d.keys():
                d[channel] += row[1][conv_col] / len(cnt)
            else:
                d[channel] = row[1][conv_col] / len(cnt)

    if plot:
        d_ = {k: v/ sum(d.values()) for k, v in d.items()}
        fig = px.bar(y=d_.keys(), x=d_.values(), title='Linear model', orientation='h')
        fig.update_xaxes(title_text='Доля от всех конверсий')
        fig.update_yaxes(title_text='Группа каналов')
        fig.show()

    if as_frame:
        if keys_col_name is not None and values_col_name is not None:
            return dict_to_frame(d, keys_col_name, values_col_name)
        else:
            keys_col_name = path_col + '_new'
            values_col_name = conv_col
            return dict_to_frame(d, keys_col_name, values_col_name)

    return d


def time_decay(df, unique_channels, path_col='path', path_len='path_len', conv_col='conversion', sep='^',
               as_frame=False, plot=True, recent=True, keys_col_name=None, values_col_name=None):
    """
    Preforms time decay attribution model
    when recent = True then more recent channel is more valuable
    when recent = False then less recent channel is more valuable

    :param df: (pd.DataFrame) dataframe with all data, if prep_data was used then it is "full_gr" dataframe
    :param unique_channels: (iterable) list(or other iterable) of channels in paths
    :param path_col: (str, default = 'path') name of column with path data
    :param conv_col: (str, default = 'conversion') name of column with conversion data
    :param plot: (bool, default = True) whether to plot the data. Plot will contain normalized data
    :param as_frame: (bool, optional, default=False) whether to return data as pandas.DataFrame
    :param keys_col_name: (str, optional, default=None) must be specified if as_frame is set to True
    :param values_col_name: (str, optional, default=None) must be specified if as_frame is set to True
    :param sep: (str, optional default='^'), channel separator used in paths

    :returns
    - dictionary with channels an their attribution score (not-normalized)
    """

    d = {channel: 0 for channel in unique_channels}
    for path, path_len, conversions in zip(df[path_col], df[path_len], df[conv_col]):
        L = path_len

        cur_path = path.split(sep)
        if recent:
            for channel in cur_path:
                d[channel] += int(conversions / L)
                L -= 1
        else:
            for channel in reversed(cur_path):
                d[channel] += int(conversions / L)
                L -= 1

    if plot:
        d_ = {k: v/sum(d.values()) for k, v in zip(d.keys(), d.values())}
        fig = px.bar(y=d_.keys(), x=d_.values(), title='Time decay model', orientation='h')
        fig.update_xaxes(title_text='Доля от всех конверсий')
        fig.update_yaxes(title_text='Группа каналов')
        fig.show()

    if as_frame:
        if keys_col_name is not None and values_col_name is not None:
            return dict_to_frame(d, keys_col_name, values_col_name)
        else:
            keys_col_name = path_col + '_new'
            values_col_name = conv_col
            return dict_to_frame(d, keys_col_name, values_col_name)

    return d


def position(df, unique_channels, path_col='path', conv_col='conversion', sep='^', plot=True, as_frame=False,
             keys_col_name=None, values_col_name=None):
    """
    Preforms position_based attribution model
    In position based model 30% of conversions are attributed to the first and last touchpoints
    and 40% are evenly attributed to other

    :param df: (pd.DataFrame) dataframe with all data, if prep_data was used then it is "full_gr" dataframe
    :param unique_channels: (iterable) list(or other iterable) of channels in paths
    :param path_col: (str, default = 'path') name of column with path data
    :param conv_col: (str, default = 'conversion') name of column with conversion data
    :param plot: (bool, default = True) whether to plot the data. Plot will contain normalized data
    :param as_frame: (bool, optional, default=False) whether to return data as pandas.DataFrame
    :param keys_col_name: (str, optional, default=None) must be specified if as_frame is set to True
    :param values_col_name: (str, optional, default=None) must be specified if as_frame is set to True
    :param sep: (str, optional default='^'), channel separator used in paths

    :returns
    - dictionary with channels an their attribution score (not-normalized)
    """

    d = {channel : 0 for channel in unique_channels}

    for path, conversions in zip(df[path_col], df[conv_col]):
        cur_path = path.split(sep)
        L = len(cur_path)

        for pos in range(L):
            if pos == 0:
                d[cur_path[pos]] += conversions * 0.3

            elif pos == L-1 and pos != 0:
                d[cur_path[pos]] += conversions * 0.3

            else:
                d[cur_path[pos]] += int((conversions * 0.4) / (L-2))

    if plot:
            d_ = {k: v/sum(d.values()) for k, v in zip(d.keys(), d.values())}
            fig = px.bar(y=d_.keys(), x=d_.values(), title='Position based model', orientation='h')
            fig.update_xaxes(title_text='Доля от всех конверсий')
            fig.update_yaxes(title_text='Группа каналов')
            fig.show()

    if as_frame:
        if keys_col_name is not None and values_col_name is not None:
            return dict_to_frame(d, keys_col_name, values_col_name)
        else:
            keys_col_name = path_col + '_new'
            values_col_name = conv_col
            return dict_to_frame(d, keys_col_name, values_col_name)

    return d


def cpa(conversions, conversion_col, costs, cost_col, col_to_join):
    """
    Computes CPA
    :param conversions: pd.DataFrame with conversions data
    :param costs: pd.DataFrame with cost data
    :param col_to_join: string name of column to join costs and attribution data
    :return: pandas.DataFrame with channel_col, total_conversions, total_cost and CPA
    """
    df = conversions.merge(costs, left_on=col_to_join, right_on=col_to_join)
    df['cpa'] = df[cost_col]/ df[conversion_col]

    return df

