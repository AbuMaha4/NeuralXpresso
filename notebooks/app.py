import dash
from dash import dcc
from dash.dependencies import Input, Output, State
from dash import html

import pandas as pd
import plotly.express as px
import neuralxpresso2 as nx
import plots as plots


app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H1("NeuralXpresso", className='header')
    ]),
    dcc.Input(id='input', type='text', value=''),
    html.Button('Submit', id='submit-button', n_clicks=0),
    html.Div(id='graphs-container')
])


@app.callback(
    Output('graphs-container', 'children'),
    Input('submit-button', 'n_clicks'),
    State('input', 'value')
)


def update_graph(n_clicks, value):
    if n_clicks == 0:
        return []
    
    nxp = nx.NeuralXpressoSession(yt_link = value)
    result = nxp.run_analysis()
    df_character = result['character_overview']
    df_video = result['video_overview']

    figures = []

    # Call the get_overview_normalized function and create a dcc.Graph component
    overview_fig = plots.get_overview_normalized(df_video)
    overview_graph = dcc.Graph(figure=overview_fig)

    for ID in df_character.person_ID:
        if (df_character.loc[df_character['person_ID'] == ID].appearances.values[0] > (0.2 * df_character.appearances.sum())):
            fig = plots.get_character_overview(df_video, ID, result)
            figures.append(fig)

    # Create dcc.Graph components for each character overview figure
    character_graphs = [dcc.Graph(figure=fig) for fig in figures]

    graphs = [overview_graph] + character_graphs


    return graphs


if __name__ == '__main__':
    app.run_server(debug=True)