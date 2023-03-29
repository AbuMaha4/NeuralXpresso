import dash
from dash import dcc
from dash.dependencies import Input, Output, State
from dash import html
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import pandas as pd
import numpy as np



import pandas as pd
import plotly.express as px
import neuralxpresso2 as nx
import plots as plots




x = np.random.sample(100)
y = np.random.sample(100)
z = np.random.choice(a = ['a','b','c'], size = 100)


df1 = pd.DataFrame({'x': x, 'y':y, 'z':z}, index = range(100))

fig1 = px.scatter(df1, x= x, y = y, color = z)

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "20%",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}


sidebar = html.Div(
    [
        html.H2("Filters"),
        html.Hr(),
        html.P(
            "A simple sidebar layout with filters", className="lead"
        ),
        dbc.Nav(
            [
                dcc.Dropdown(id='one'),
                html.Br(),
                dcc.Dropdown(id='two'),
                html.Br(),
                dcc.Dropdown(id='three')

            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)












### ----------------------CREATING LAYOUT OF THE APP-------------###
app = dash.Dash(external_stylesheets=[dbc.themes.LUX])


app.layout = html.Div(children=[
    dbc.Row([
        dbc.Col(),
        dbc.Col(html.H1('NeuralXpresso'), width=9, style={'margin-left': '7px', 'margin-top': '7px'})
    ]),
    dbc.Row([
        dbc.Col(),
        dbc.Col([
            dbc.InputGroup([
                dbc.Input(id='input', type='text', value='', placeholder='Enter a YouTube link or upload a video file'),
                dbc.Button("Submit", id='submit-button', n_clicks=0),
            ], style={'margin-top': '10px', 'width': '50%', 'margin-left': '5px'})
        ], width=9)
    ]),
    dbc.Row([
        dbc.Col(),
        dbc.Col([
            html.Div(id='output-div')
    ], width=9, style={'margin-top': '10px', 'margin-left': '5px'})
]),
    dbc.Row([
        dbc.Col(sidebar),
        dbc.Col(dcc.Graph(id='graph1', figure=fig1), width=9, style={'margin-left': '30px', 'margin-top': '7px', 'margin-right': '15px'})
    ]),
])


@app.callback(
    Output('output-div', 'children'),
    Input('submit-button', 'n_clicks'),
    State('input', 'value')
)


def run_model(n_clicks, value):
    if n_clicks == 0:
        return []
    nxp = nx.NeuralXpressoSession(yt_link = value)
    result = nxp.run_analysis()
    return result

    

def update_video_stats(n_clicks, input_link):
    if n_clicks == 0:
        return "Please enter a video link and click Submit"
    
    analysis_results = run_model(input_link)
    return analysis_results



if __name__ == '__main__':
        app.run_server(debug=True)


