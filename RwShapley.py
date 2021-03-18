from shapley import RwShap
import pandas as pd


# Учитываем частоту как FIC
# TODO: Rewrite for general case of dataset
def freq(df, int_type_col, col_to_group, order_col, id_col, click_weight = 1, view_weight = 3):

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


# Combine Shapley and FIC
def shap_and_freq(sh_clicks, sh_views, df_for_freq, int_type_col, channel_col, col_to_group):

    data = df_for_freq


    freq_c, freq_i = freq(data, int_type_col, col_to_group)

    click = sh_clicks.merge(freq_c, left_on = channel_col, right_on = channel_col)
    view = sh_views.merge(freq_i, left_on = channel_col, right_on = channel_col)

    total = (click
             .merge(view, how = 'outer', left_on = channel_col, right_on = channel_col)
             .fillna(0)
            )

    total['total_weight'] = total['weight_x'] * total['FIC_x'] + total['weight_y'] * total['FIC_y']

    total['total_weight'] = total['total_weight'] / total['total_weight'].sum()

    return total[[channel_col, 'total_weight']]
