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
import base64



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


algorithm_info = {
    'MTCNN': 'MTCNN is a deep learning-based face detection algorithm that uses a cascading technique to detect faces in an image. It is highly accurate and can detect faces of varying sizes and orientations.',
    'Haar Cascade': 'Haar Cascade is a machine learning-based face detection algorithm that uses a set of Haar-like features to detect faces in an image. It is fast and can detect faces in real-time, but it may have lower accuracy compared to deep learning-based approaches.',
    'Face Recognition': 'The Face-Recognition package is a deep learning-based face recognition algorithm that uses a neural network to encode faces into a 128-dimensional vector space. It can recognize faces with high accuracy, but it requires the faces to be aligned and well-lit, and may be computationally expensive for large datasets. Right now, this is the default model.'
}



sidebar = html.Div(
    [
        html.H2("Playground", style={'fontSize': 22}),
        html.Hr(),
        html.P("Coming soon", className="lead", style={'fontSize': 19}),
        dbc.Nav(
            [   html.Label('Chose Video Resolution', style={'fontSize': 14}),
                dcc.Dropdown(
                    id='DP one',
                    options=[
                        {'label': '1040p', 'value': 'opt1'},
                        {'label': '720p', 'value': 'opt2'},
                        {'label': '360p', 'value': 'opt3'},
                        {'label': '144p', 'value': 'opt4'}
                    ],
                    value='opt1',
                    clearable=False
                ),
                html.Br(),
                html.Label('Choose Face Detector',style={'fontSize': 14}),
                dcc.Dropdown(
                    id='DP two',
                    options=[
                        {'label': 'Face Recognition', 'value': 'Face Recognition'},
                        {'label': 'MTCNN', 'value': 'MTCNN'},
                        {'label': 'Haar Cascade', 'value': 'Haar Cascade'}
                    ],
                    value='None',
                    clearable=False
                ),
                html.Br(),
                html.Label('Face Detection Sensitivity',style={'fontSize': 14}),
                dcc.Slider(
                    id='two',
                    min=0,
                    max=100,
                    step=10,
                    value=60,
                    marks={i: str(i) for i in range(0, 100, 20)}
                ),
                html.Br(),
                html.Label('Character Recognition Sensitivity',style={'fontSize': 14}),

                dcc.Slider(
                    id='three',
                    min=0,
                    max=100,
                    step=10,
                    value=10,
                    marks={i: str(i) for i in range(0, 100, 20)}
                ),
                html.Br(),
                html.Label('Skip Frames',style={'fontSize': 14}),

                dcc.Slider(
                    id='three',
                    min=0,
                    max=100,
                    step=10,
                    value=20,
                    marks={i: str(i) for i in range(0, 100, 20)}
                ),
                html.Br(),
                html.Div(id='algorithm-info', style={'padding': '10px', 'border': '1px solid #ccc'})

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
    ]),

    dbc.Row([
        dbc.Col(),
        dbc.Col([
            html.Div([
                html.Div(id='output-stats'),
                dcc.Graph(id='output-thumbnail', style={'display': 'block'}),
                html.Button("Start Analysis", id='analysis-button', disabled=True),
                html.Div(id='analysis-output')
            ])
        ], width=9, style={'margin-top': '10px', 'margin-left': '5px'})
    ]),

])


# Define the update_video_stats callback
@app.callback(
    [Output('output-stats', 'children'),
     Output('output-thumbnail', 'figure'),
     Output('analysis-button', 'disabled'),
     Output('analysis-output', 'children')],
    [Input('submit-button', 'n_clicks'),
     Input('analysis-button', 'n_clicks')],
    State('input', 'value'))

def update_video_stats(submit_clicks, analysis_clicks,input_value):
    try:
        if not input_value:
            fig = go.Figure(layout=go.Layout(xaxis=dict(showgrid = False), yaxis=dict(showgrid=False)))
            fig.update_yaxes(visible=False, showticklabels=False)
            fig.update_xaxes(visible=False, showticklabels=False)
            fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            return [], fig, True, []

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
            #table_header = [html.Tr([html.Th("Info_Type"), html.Th("Info")])]
            table_body = [html.Tr([html.Td("Title"), html.Td(title)]),
                          html.Tr([html.Td("Views"), html.Td(views)]),
                          html.Tr([html.Td("Duration [Frames]"), html.Td(duration)]),
                          html.Tr([html.Td("Resolution"), html.Td(resolution)])]
            stats_table = html.Table(table_body,
                                     style={'border': '1px solid black', 'border-collapse': 'collapse',
                                            'width': '100%'})
            graph_style = {'display': 'block'}
            # Display thumbnail image using Plotly Express
            if thumbnail is None:
                fig = go.Figure(layout=go.Layout(xaxis=dict(showgrid = False), yaxis=dict(showgrid=False)))
                fig.update_yaxes(visible=False, showticklabels=False)
                fig.update_xaxes(visible=False, showticklabels=False)
                fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                return stats_table, fig, False, []
            else: 
                fig = px.imshow(cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)).update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
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


@app.callback(
    Output('algorithm-info', 'children'),
    Input('DP two', 'value'),
    prevent_initial_call=True  
)

def update_algorithm_info(algorithm):
    ctx = dash.callback_context  # Add this line to access callback context
    if ctx.triggered:  # Check if the callback was triggered by an event
        return html.P(algorithm_info.get(algorithm, ''))
    return ''


if __name__ == '__main__':
        app.run_server(debug=True)


