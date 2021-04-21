import pandas as pd
from tools.exceptions import MissInputData

def prep_data(df, channel_col, client_id_col, interaction_type_col, with_null_path=True, conv_col=None,
              full_data=True, click_only=False, view_only=False, sort=False, verbose=0, sep='^'):

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
    :param verbose:
    :param sep:
    :return:
    """

    if with_null_path == False:
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
                .groupby(client_id_col, as_index = False).agg({'channel_new': 'sum'})
                .rename(columns = {'channel_new': 'path'})
               )


    click = (df
             .query(f'{interaction_type_col} == "Click"')
             .groupby(client_id_col, as_index = False).agg({'channel_new': 'sum'})
             .rename(columns = {'channel_new': 'path'})
            )

    view = (df
            .query(f'{interaction_type_col} == "Impression"')
            .groupby(client_id_col, as_index = False).agg({'channel_new': 'sum'})
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
                 .groupby('path', as_index = False)
                 .agg({client_id_col : 'nunique'})
                 .rename(columns={client_id_col : 'conversion'})
                 )
        if sort:
            df_gr['path'] = df_gr['path'].apply(lambda x: sorted(x.split(sep)))
            df_gr['path_len'] = df_gr['path'].apply(lambda x: len(x))
            df_gr['path'] = df_gr['path'].apply(lambda x: conc(x))

            return df_gr

        return df_gr

    if full_data:
        full_gr = sort_and_group(full, sort)
        click_gr = sort_and_group(click, sort)
        view_gr = sort_and_group(view, sort)

        return full_gr, click_gr, view_gr

    if click_only:
        click_gr = sort_and_group(click, sort)

        return click_gr

    if view_only:
        view_gr = sort_and_group(view, sort)

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


# Функция рассчета FIC
def compute_FIC(df, int_type_col, col_to_group, order_col, id_col):

    """
    Computes frequency impact coefficient (FIC)
    FIC is defined as: (number of unique paths where channel appeared)
    / (number of occurences  of a channel in dataset)

    Inputs:
    - df (pandas.DataFrame)
    - int_type_col (str)
    - cnt_col (str)
    """

    # clicks
    c = df[df[int_type_col] == "Click"]
    c = (c
         .groupby(col_to_group, as_index = False)
         .agg({order_col : 'count', id_col : 'nunique'})
         .rename(columns = {order_col : 'total_occ', id_col: 'uniq_path'})
        )
    c['FIC'] = c['uniq_path'] / c['total_occ']


    # impressions
    i = df[df[int_type_col] == "Impression"]
    i = (i
         .groupby(col_to_group, as_index = False)
         .agg({order_col : 'count', id_col : 'nunique'})
         .rename(columns = {order_col : 'total_occ', id_col: 'uniq_path'})
        )
    i['FIC'] = i['uniq_path'] / i['total_occ']

    return (c[[col_to_group, 'FIC']], i[[col_to_group, 'FIC']])