import plotly.graph_objs as go
from plotly.io import write_html
from tools.exceptions import MissInputData


def compare(attributions, channel_col, weight_col, names=None, orientation=None, width=None, height=None,
            to_html=False, path=None):

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

    if to_html:
        if path is not None:
            write_html(fig, path)
        else:
            print('Please pass path where to save htm file')
            raise MissInputData
    return

