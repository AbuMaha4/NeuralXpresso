import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plots_functions as pf
import neuralxpresso as nx


app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Input(id='input', type='text', value=''),
    dcc.Graph(id='graph1'),
    dcc.Graph(id='graph2'),
    dcc.Graph(id='graph3')
    
])



@app.callback(
    [Output('graph1', 'figure'),Output('graph2', 'figure'), Output('graph3','figure')],
    [Input('input', 'value')]
)


def update_graph(value):
    # Convert the user input to a numpy array
    #df_plotting = pd.read_csv("/Users/mohan/neuefische/NeuralXpresso/data/plotting.csv")
    nxp = nx.NeuralXpressoSession(yt_link = value)
    df_plotting = nxp.run_analysis()

    fig1 = px.bar(df_plotting, x="frame", y='probability', color = 'emotion',
             barmode="stack", title="Distribution of Emotions across Frames (every 10th frame)")
    fig1.update_layout(xaxis_title="Frame", yaxis_title="Emotion Distribution",     title={
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    })
    
    df = df_plotting
    df['moving_avg'] = df['probability'].rolling(window=10, center=True).mean()
    fig2 = px.area(df, x="frame", y="moving_avg", color="emotion")
    #fig2 = pf.ThreeDimensionalGraph(df_plotting)

    fig3 = pf.hist_max_prob_emo(df_plotting)
    

    return [fig1,fig2,fig3]


if __name__ == '__main__':
    app.run_server(debug=True)