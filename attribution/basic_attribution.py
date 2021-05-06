import pandas as pd
import plotly.express as px
from tools.prep import dict_to_frame
from tools.exceptions import MissInputData


def last_click(df, path_col='path', conv_col='conversion', plot=True):
    """
    Preforms last-click attribution model
    :param
    - df (pd.DataFrame) : dataframe ONLY with clicks data, if prep_data was used then it is "click_gr" dataframe
    - path_col (str, default = 'path') : name of column with path data
    - conv_col (str, default = 'conversion') : name of column with conversion data
    - plot (bool, default = True) : whether to plot the data. Plot will contain normalized data

    :returns
    - pandas.DataFrame with channels an their attribution score (not-normalized)
    """
    df_ = df.copy()
    df_['first'] = df_[path_col].apply(lambda x: x.split('^')[-1])

    plt_data = df_.groupby('first', as_index = False).agg({conv_col : 'sum'})
    plt_data[conv_col+'_rel'] = plt_data[conv_col] / plt_data[conv_col].sum()


    if plot:
        fig = px.bar(plt_data, y = 'first', x = conv_col+'_rel', title = 'Last Click Model', orientation = 'h')
        fig.update_xaxes(title_text = 'Доля от всех конверсий')
        fig.update_yaxes(title_text = 'Группа каналов')
        fig.show()

    return plt_data


# TODO: Переписать Last-non-direct-click для данных помимо мазды
def last_non_direct_click(df):
    not_direct = df.query('channel_grouping != "Прямой" and session_with_conversion == True \
    and post_click_or_post_view == "click"')
    direct = df.query('channel_grouping == "Прямой" and session_with_conversion == True')
    dir_cid = direct['clientId'].unique()

    _ = (df
        .query('''session_with_conversion == False and post_click_or_post_view == "click" \
               and channel_grouping != "Прямой" and clientId in @dir_cid''')
        .groupby('clientId', as_index = False)
        .agg({'time' : 'max'})
       )

    test = (df
            .merge(_, how = 'left', left_on = 'clientId', right_on = 'clientId').query('time_x == time_y')
            .drop('time_y', axis = 1)
            .rename(columns = {'time_x' : 'time'})
            .query('post_click_or_post_view == "click"')
           )



    res = pd.concat([not_direct, test])
    res = pd.concat([res, direct[~direct['clientId'].isin(test['clientId'].unique())]])

    plt_data = res.groupby('channel_grouping', as_index = False).agg({'clientId' : 'nunique'})

    plt_data['clientId'] = plt_data['clientId'] / plt_data['clientId'].sum()

    fig = px.bar(plt_data, y = 'channel_grouping', x = 'clientId', title = 'Last non-direct click Model', orientation = 'h')
    fig.update_xaxes(title_text = 'Доля от всех конверсий')
    fig.update_yaxes(title_text = 'Группа каналов')
    fig.show()

    return


def first_click(df, path_col = 'path', conv_col='conversion', plot=True):
    """
    Preforms first-click attribution model
    :param
    - df (pd.DataFrame) : dataframe ONLY with clicks data, if prep_data was used then it is "click_gr" dataframe
    - path_col (str, default = 'path') : name of column with path data
    - conv_col (str, default = 'conversion') : name of column with conversion data
    - plot (bool, default = True) : whether to plot the data. Plot will contain normalized data

    :returns
    - pandas.DataFrame with channels an their attribution score (not-normalized)
    """

    df_ = df.copy()
    df_['first'] = df_[path_col].apply(lambda x: x.split('^')[0])

    plt_data = df_.groupby('first', as_index = False).agg({conv_col : 'sum'})
    plt_data[conv_col+'_rel'] = plt_data[conv_col] / plt_data[conv_col].sum()


    if plot:
        fig = px.bar(plt_data, y = 'first', x = conv_col+'_rel', title = 'First click model', orientation = 'h')
        fig.update_xaxes(title_text = 'Доля от всех конверсий')
        fig.update_yaxes(title_text = 'Группа каналов')
        fig.show()

    return plt_data


def uniform(df, unique_channels, path_col = 'path', conv_col='conversion', plot=True, as_frame=False,
            keys_col_name=None, values_col_name=None):
    """
    Preforms last-click attribution model
    :param
    - df (pd.DataFrame) : dataframe with all data, if prep_data was used then it is "full_gr" dataframe
    - unique_channels (iterable) : list(or other iterable) of channels in paths
    - path_col (str, default='path') : name of column with path data
    - conv_col (str, default='conversion') : name of column with conversion data
    - plot (bool, default=True) : whether to plot the data. Plot will contain normalized data
    - as_frame (bool, optional, default=False) : whether to return data as pandas.DataFrame
    - keys_col_name (str, optional, default=None): must be specified if as_frame is set to True
    - values_col_name (str, optional, default=None): must be specified if as_frame is set to True
    :returns
    - dictionary with channels an their attribution score (not-normalized) if as_frame=False
    - pandas.DataFrame with channels an their attribution score (not-normalized) if as_frame=True
    """
    df_ = df.copy()
    d = {}
    for channel in unique_channels:
        df_[channel] = df_[path_col].apply(lambda x: channel in x)
#         df_[channel + '_len'] =

        d[channel] = df_[df_[channel] == True][conv_col].sum()

    d_ = {k : v/sum(d.values()) for k, v in zip(d.keys(), d.values())}

    if plot:
        fig = px.bar(y = d_.keys(), x = d_.values(), title = 'Uniform model', orientation = 'h')
        fig.update_xaxes(title_text = 'Доля от всех конверсий')
        fig.update_yaxes(title_text = 'Группа каналов')
        fig.show()

    if as_frame == True:
        try:
            if keys_col_name is not None and values_col_name is not None:
                return dict_to_frame(d, keys_col_name, values_col_name)
            else:
                raise MissInputData
        except MissInputData:
            print("When as_frame is set to True keys_col_name and values_col_name must be specified")



def time_decay(df, unique_channels, path_col='path', path_len='path_len', conv_col='conversion', as_frame=False,
               plot=True, recent=True, keys_col_name=None, values_col_name=None):
    """
    Preforms time decay attribution model
    when recent = True then more recent channel is more valuable
    when recent = False then less recent channel is more valuable

    :param
    - df (pd.DataFrame) : dataframe with all data, if prep_data was used then it is "full_gr" dataframe
    - unique_channels (iterable) : list(or other iterable) of channels in paths
    - path_col (str, default = 'path') : name of column with path data
    - conv_col (str, default = 'conversion') : name of column with conversion data
    - plot (bool, default = True) : whether to plot the data. Plot will contain normalized data

    :returns
    - dictionary with channels an their attribution score (not-normalized)
    """

    d = {channel : 0 for channel in unique_channels}
    for path, path_len, conversions in zip(df[path_col], df[path_len], df[conv_col]):
        L = path_len

        cur_path = path.split('^')
        if recent:
            for channel in cur_path:
                d[channel] += int(conversions / L)
                L -= 1
        else:
            for channel in reversed(cur_path):
                d[channel] += int(conversions / L)
                L -= 1

    if plot == True:
        d_ = {k: v/sum(d.values()) for k, v in zip(d.keys(), d.values())}
        fig = px.bar(y=d_.keys(), x=d_.values(), title='Time decay model', orientation='h')
        fig.update_xaxes(title_text='Доля от всех конверсий')
        fig.update_yaxes(title_text='Группа каналов')
        fig.show()

    if as_frame == True:
        try:
            if keys_col_name is not None and values_col_name is not None:
                return dict_to_frame(d, keys_col_name, values_col_name)
            else:
                raise MissInputData
        except MissInputData:
            print("When as_frame is set to True keys_col_name and values_col_name must be specified")

    return d


def position(df, unique_channels, path_col='path', conv_col='conversion', plot=True, as_frame=False,
             keys_col_name=None, values_col_name=None):
    """
    Preforms position_based attribution model
    In position based model 30% of conversions are attributed to the first and last touchpoints
    and 40% are evenly attributed to other

    :param
    - df (pd.DataFrame) : dataframe with all data, if prep_data was used then it is "full_gr" dataframe
    - unique_channels (iterable) : list(or other iterable) of channels in paths
    - path_col (str, default = 'path') : name of column with path data
    - conv_col (str, default = 'conversion') : name of column with conversion data
    - plot (bool, default = True) : whether to plot the data. Plot will contain normalized data

    :returns
    - dictionary with channels an their attribution score (not-normalized)
    """

    d = {channel : 0 for channel in unique_channels}

    for path, conversions in zip(df[path_col], df[conv_col]):
        cur_path = path.split('^')
        L = len(cur_path)

        for pos in range(L):
            if pos == 0:
                d[cur_path[pos]] += conversions * 0.3

            elif pos == L-1 and pos != 0:
                d[cur_path[pos]] += conversions * 0.3

            else:
                d[cur_path[pos]] += int((conversions * 0.4) / (L-2))

    if plot == True:
            d_ = {k: v/sum(d.values()) for k, v in zip(d.keys(), d.values())}
            fig = px.bar(y=d_.keys(), x=d_.values(), title='Position based model', orientation='h')
            fig.update_xaxes(title_text='Доля от всех конверсий')
            fig.update_yaxes(title_text='Группа каналов')
            fig.show()

    if as_frame == True:
        try:
            if keys_col_name is not None and values_col_name is not None:
                return dict_to_frame(d, keys_col_name, values_col_name)
            else:
                raise MissInputData
        except MissInputData:
            print("When as_frame is set to True keys_col_name and values_col_name must be specified")

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

