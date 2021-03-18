import pandas as pd
import numpy as np


def prep_data(df, channel_col, client_id_col, interaction_type_col, full_data = True, click_only = False,
              view_only = False, sort = True, verbose=0):

    """
    Function that does initial data preprocessing

    Returns: tuple
    """

    df["channel_new"] = df[channel_col] + '^'
    # Собираем цепчки вместе
    if full_data:
        full = (df
                .groupby(client_id_col, as_index = False).agg({'channel_new' : 'sum'})
                .rename(columns = {'channel_new' : 'path'})
               )


    click = (df
             .query(f'{interaction_type_col} == "Click"')
             .groupby(client_id_col, as_index = False).agg({'channel_new' : 'sum'})
             .rename(columns = {'channel_new' : 'path'})
            )

    view = (df
            .query(f'{interaction_type_col} == "Impression"')
            .groupby(client_id_col, as_index = False).agg({'channel_new' : 'sum'})
            .rename(columns = {'channel_new' : 'path'})
           )

    if verbose > 0:
        print("1-st step is done")

    # Преобразует массив в строку через разделитель
    def conc(a, sep = '^'):
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
            df_gr['path'] = df_gr['path'].apply(lambda x: sorted(x.split('^')))
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

