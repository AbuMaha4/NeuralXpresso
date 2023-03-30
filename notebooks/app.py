import dash
from dash import dcc
from dash.dependencies import Input, Output, State
from dash import html
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import pandas as pd
import numpy as np
import cv2
import plotly.graph_objs as go



import pandas as pd
import plotly.express as px
import neuralxpresso as nx
import plots as plots





SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "20%",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}


logo = html.Div(
    [
        html.Img(
            src="assets/logo.png",
            style={"height": "60px"}
        )
    ],
    style={
        "position": "absolute",
        "top": "20px",
        "left": "20px",
        "padding-top": "10px",
        "padding-left": "10px"
    }
)


sidebar = html.Div(
    [
        html.H2("Filters"),
        html.Hr(),
        html.P("A simple sidebar layout with filters", className="lead"),
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
    logo,
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
    ]),

    dbc.Row([
        dbc.Col(),
        dbc.Col([
            html.Div([
                html.Div(id='output-stats'),
                dcc.Graph(id='output-thumbnail'),
                html.Button("Start Analysis", id='analysis-button', disabled=True),
                html.Div(id='analysis-output')
            ])
        ], width=9, style={'margin-top': '10px', 'margin-left': '5px'}),
        #for overallview text
                dbc.Row([
                    dbc.Col(),
                    dbc.Col(html.H2('Overall emotion of the entire video for all characters'), width=9, style={'margin-left': '40px', 'margin-top': '7px'})
                ]),

                ])

    ])


# Define the update_video_stats callback
@app.callback(
    [Output('output-stats', 'children'),
     Output('output-thumbnail', 'figure'),
     Output('analysis-button', 'disabled'),
     Output('analysis-output', 'children')],
    [Input('submit-button', 'n_clicks'),
     Input('analysis-button', 'n_clicks')],
    State('input', 'value')
)

def update_video_stats(submit_clicks, analysis_clicks, input_value):
    try:
        if not input_value:
            return [], go.Figure(), True, []

        ctx = dash.callback_context
        triggered_by = ctx.triggered[0]['prop_id']

        if 'submit-button' in triggered_by:
            Video_processor = nx.VideoProcessor(input_value)
            stats = Video_processor.get_video_info()

            # Extract relevant statistics
            thumbnail = stats['thumbnail']
            title = stats['title']
            duration = stats['total_frame_count']
            views = stats['views']
            resolution = stats['available_resolutions']

            # Create table with video statistics
            table_header = [html.Tr([html.Th("Info_Type"), html.Th("Info")])]
            table_body = [html.Tr([html.Td("Title"), html.Td(title)]),
                          html.Tr([html.Td("Views"), html.Td(views)]),
                          html.Tr([html.Td("Duration"), html.Td(duration)]),
                          html.Tr([html.Td("Resolution"), html.Td(resolution)])]
            stats_table = html.Table(table_header + table_body,
                                     style={'border': '1px solid black', 'border-collapse': 'collapse',
                                            'width': '100%'})

            # Display thumbnail image using Plotly Express
            if thumbnail is None:
                fig = None
            else:
                fig = px.imshow(cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)).update_xaxes(showticklabels=False). \
                    update_yaxes(showticklabels=False)

            return stats_table, fig, False, []

        elif 'analysis-button' in triggered_by:
            nxp = nx.NeuralXpressoSession(yt_link=input_value)
            result = nxp.run_analysis(main_character_threshold=0.25, skip_frames=12)
            data = result['new_export']['main_character_data']

            figures = []



            # Call the get_overview_normalized function and create a dcc.Graph component
            overview_fig = plots.get_overall_overview(result['new_export']['overview_mean'])

            overview_graph = dcc.Graph(figure=overview_fig)

            for ID in data:
                fig1 = plots.get_character_overview(data[ID], ID, result)
                figures.append(fig1)
                fig2 = plots.get_strongest_emotions_plot(data[ID])
                figures.append(fig2)


            # Create dcc.Graph components for each character overview figure
            character_graphs = [dcc.Graph(figure=fig) for fig in figures]

            graphs = [overview_graph] + character_graphs

            return dash.no_update, dash.no_update, dash.no_update, graphs
        else:
            return [], go.Figure(), True, []


    except Exception as e:
        return html.Div([f'Error: {e}']), go.Figure(), True, []



if __name__ == '__main__':
        app.run_server(debug=True)


