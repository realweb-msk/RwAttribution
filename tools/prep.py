from collections import Counter
import pandas as pd
from tools.exceptions import MissInputData


def prep_data(df, channel_col, client_id_col, interaction_type_col, with_null_path=True, conv_col=None,
              full_data=True, click_only=False, view_only=False, sort=False, verbose=0, sep='^',
              drop_direct=False, direct_name=None):
    """
    Function that does initial data preprocessing. You can find expected input data format here:
    https://github.com/realweb-msk/RwAttribution#readme

    In case dataset has paths, that did not lead to conversion set parameter with_null_path to False

    :param df: (pandas.DataFrame), dataset with conversion paths
    :param channel_col: (str), name of the column with channel of touchpoint
    :param client_id_col: (str), name of the column with some id (conversionId, clientId, cookie, etc.) of touchpoint
    :param interaction_type_col: (str), name of the column with one of the following interaction_types:
    "Click", "Impression".
    :param with_null_path: (bool, optional, default=True), In case dataset has paths, that did not lead to conversion
    set parameter to False and provide conv_col
    :param conv_col: (str, optional), name of the column with path's conversion flag (int or bool). Must be provided
    if with_null_path is set to False
    :param full_data: (bool, optional, default=True), whether to return full_data, click_data and impression_data
    :param click_only: (bool, optional, default=False), whether to return click_data only
    :param view_only: (bool, optional, default=False), whether to return impression_data only
    :param sort: (bool, optional, default=True)
    :param verbose: (int, optional, default=0), if greater than zero prints progress at runtime
    :param sep: (str, optional, default='^'), character that will separate channels in paths after grouping
    :param drop_direct: (bool, optional, default=False), if True then drops "Direct" from touchpoints with any other
    touchpoint
    :param direct_name: (str, optional, default=None), name of "Direct" touchpoint in your paths,
     must be not None when null_direct=True
    :return:
    """

    if not with_null_path:
        try:
            if conv_col is not None:
                id_s = df.query(f"{conv_col} == 1")[client_id_col].unique()
                df = df[df[client_id_col].isin(id_s)]
            else:
                raise MissInputData
        except MissInputData as e:
            print("When with_null_path=True, data must contain conv_col")
            raise e

    df["channel_new"] = df[channel_col] + sep
    # Собираем цепочки вместе
    if full_data:
        full = (df
                .groupby(client_id_col, as_index=False).agg({'channel_new': 'sum'})
                .rename(columns={'channel_new': 'path'})
                )

    click = (df
             .query(f'{interaction_type_col} == "Click"')
             .groupby(client_id_col, as_index=False).agg({'channel_new': 'sum'})
             .rename(columns={'channel_new': 'path'})
             )

    view = (df
            .query(f'{interaction_type_col} == "Impression"')
            .groupby(client_id_col, as_index=False).agg({'channel_new': 'sum'})
            .rename(columns={'channel_new': 'path'})
            )

    if verbose > 0:
        print("1-st step is done")

    # Преобразует массив в строку через разделитель
    def conc(a, sep=sep):
        res = ''
        for i in a:
            res += i + sep

        return res[:-1]

    # Отсортируем каналы в цепочке по алфавиту и сгруппируем по цепочкам
    def sort_and_group(df, sort):
        df['path'] = df['path'].apply(lambda x: x[:-1])
        df_gr = (df
                 .groupby('path', as_index=False)
                 .agg({client_id_col: 'nunique'})
                 .rename(columns={client_id_col: 'conversion'})
                 )
        if sort:
            df_gr['path'] = df_gr['path'].apply(lambda x: sorted(x.split(sep)))
            df_gr['path_len'] = df_gr['path'].apply(lambda x: len(x))
            df_gr['path'] = df_gr['path'].apply(lambda x: conc(x))

            return df_gr

        return df_gr

    # Дропает из группированной цепочки прямой траффик если он не единственный
    def dropper(path, sep=sep, direct_name=direct_name):
        replace_strs = [sep + direct_name, direct_name + sep]
        if direct_name in path and len(Counter(path.split(sep))) > 1:
            new_path = str.replace(path, replace_strs[0], '')
            new_path = str.replace(new_path, replace_strs[1], '')

            return new_path

        return path

    if full_data:
        full_gr = sort_and_group(full, sort)
        click_gr = sort_and_group(click, sort)
        view_gr = sort_and_group(view, sort)

        if drop_direct:
            full_gr['path'] = full_gr['path'].apply(lambda x: dropper(x))
            click_gr['path'] = click_gr['path'].apply(lambda x: dropper(x))
            view_gr['path'] = view_gr['path'].apply(lambda x: dropper(x))

            full_gr = full_gr.groupby('path', as_index=False).agg({'conversion': 'sum'})
            click_gr = click_gr.groupby('path', as_index=False).agg({'conversion': 'sum'})
            view_gr = view_gr.groupby('path', as_index=False).agg({'conversion': 'sum'})

        return full_gr, click_gr, view_gr

    if click_only:
        click_gr = sort_and_group(click, sort)
        if drop_direct:
            click_gr['path'] = click_gr['path'].apply(lambda x: dropper(x))
            click_gr = click_gr.groupby('path', as_index=False).agg({'conversion': 'sum'})
        return click_gr

    if view_only:
        view_gr = sort_and_group(view, sort)
        if drop_direct:
            view_gr['path'] = view_gr['path'].apply(lambda x: dropper(x))
            view_gr = view_gr.groupby('path', as_index=False).agg({'conversion': 'sum'})
        return view_gr


def dict_to_frame(dictionary, keys_col_name, values_col_name):
    """
    Transforms simple dict (without nested structure) into pandas.DataFrame
    :param dictionary: dictionary
    :param keys_col_name: name of first column in DataFrame
    :param values_col_name: name of second columns in DataFrame

    :return: pandas.DataFrame where first col in keys of dictionary and second col is values of dictionary
    """

    df = pd.DataFrame()
    df[keys_col_name] = dictionary.keys()
    df[values_col_name] = dictionary.values()

    return df


def compute_FIC(df, int_type_col, col_to_group, order_col, id_col, divided=False):
    """
    Computes frequency impact coefficient (FIC)
    FIC is defined as: (number of unique paths where channel appeared)
    / (number of occurences  of a channel in dataset)

    :param df: (pandas.DataFrame) DataFrame with raw touchpoint interactions
    :param int_type_col: (str), name of the column with one of the following interaction_types:
    "Click", "Impression".
    :param col_to_group: (str), name of column to aggregate data
    :param order_col: (order_col), name of column to order by
    :param id_col: (id_col), name of column with some ID. For further details check out
    https://github.com/realweb-msk/RwAttribution#readme
    :param divided: (bool, optional default=False)
    """

    if not divided:
        # clicks
        c = df[df[int_type_col] == "Click"]
        c = (c
             .groupby(col_to_group, as_index=False)
             .agg({order_col: 'count', id_col: 'nunique'})
             .rename(columns={order_col: 'total_occ', id_col: 'uniq_path'})
             )
        c['FIC'] = c['uniq_path'] / c['total_occ']

        # impressions
        i = df[df[int_type_col] == "Impression"]
        i = (i
             .groupby(col_to_group, as_index=False)
             .agg({order_col: 'count', id_col: 'nunique'})
             .rename(columns={order_col: 'total_occ', id_col: 'uniq_path'})
             )
        i['FIC'] = i['uniq_path'] / i['total_occ']

        return c[[col_to_group, 'FIC']], i[[col_to_group, 'FIC']]

    _df = (df.groupby(col_to_group, as_index=False)
           .agg({order_col: 'count', id_col: 'nunique'})
           .rename(columns={order_col: 'total_occ', id_col: 'uniq_path'})
           )

    _df['FIC'] = _df['uniq_path'] / _df['total_occ']

    return _df[[col_to_group, 'FIC']]
