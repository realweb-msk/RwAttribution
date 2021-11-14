import plotly.graph_objs as go
from plotly.io import write_html


def compare(attributions, channel_col, weight_col, names=None, orientation=None, width=None, height=None, path=None):
    """
    Plots results of different attribution models for comparison

    :param attributions: (iterable) - Iterable with pandas.DataFrames with attribution results
    :param channel_col: (str) - Name of column (must be the same in all attribution results dataframes) with channels
    :param weight_col: (str) - Name of column (must be the same in all attribution results dataframes) with channel
    weights
    :param names: (iterable, optional default=None) - Iterable with names of attributions. If None, plots will be
    untitled
    :param orientation: (str, optional default=None) - Orientation to plot data: "h" or "v"
    :param width: (int, optional default=None) - With of plot in pixels. If None will be set to default value 900
    :param height: (int, optional default=None) - Height of plot in pixels. If None will be set to default value 900
    :param path: (str, optional default=None) - Path to export plot as html. If None plot won't be exported

    """

    names = [i for i in range(len(attributions))] if names is None else names
    orientation = 'h' if orientation is None else orientation

    figures = [
        go.Bar(y=attr.sort_values(by=channel_col)[channel_col],
               name=f'{name} model',
               x=attr.sort_values(by=channel_col)[weight_col],
               orientation=orientation
               )
        for attr, name in zip(attributions, names)
    ]
    fig = go.Figure(figures)

    width = 900 if width is None else width
    height = 900 if height is None else height

    fig.update_layout(autosize=False, width=width, height=height, title='Model comparison')

    fig.show()

    if path is not None:
        write_html(fig, path)

    return
