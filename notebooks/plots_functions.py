import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

df_plotting = pd.read_csv("/Users/mohan/neuefische/NeuralXpresso/data/plotting.csv")


#defining function
def UpdateCBPalette(emotion_colors):
    """
    Updates a color palette with the specified colors for each emotion.

    Parameters:
    - emotion_colors (dict): a dictionary mapping emotions to color codes (e.g. {'Neutral': '#CA3435', 'Happy': '#00ff00'})
    - default_palette (list, optional): a list of color codes to use as the default palette. 
    If not specified, the default is ['#CA3435', '#ff7f00', '#9370DB', '#40E0D0', '#1f77b4', '#9467bd', '#7f7f7f'].

    Returns:
    - a modified color palette with the specified colors for each emotion.
    """
    # Define the default color palette
    default_palette = ['#CA3435', '#ff7f00', '#9370DB', '#40E0D0', '#1f77b4', '#9467bd', '#7f7f7f']

    # Create a copy of the default palette to modify
    new_palette = default_palette.copy()

    # Update the color of each emotion in the new palette
    for emotion, color in emotion_colors.items():
        if emotion in ['Neutral', 'Happy', 'Sad', 'Angry', 'Surprise', 'Fear', 'Disgust']:
            index = ['Neutral', 'Happy', 'Sad', 'Angry', 'Surprise', 'Fear', 'Disgust'].index(emotion)
            new_palette[index] = color

    return new_palette

#function for three_dimensional graph
def ThreeDimensionalGraph(df_plotting):
    '''
    takes input as data frame from the emotional model.
    '''
    # Group the DataFrame by emotion and frame, and count the number of occurrences
    df_counts = df_plotting.groupby(['emotion', 'frame'])['probability'].sum().reset_index()

    # Define the color palette
    #cb_palette = ['#40E0D0', '#ff7f00', '#9370DB', '#CA3435', '#1f77b4', '#9467bd', '#7f7f7f']

    # Define a dictionary with the desired color for each emotion
    emotion_colors = {'Neutral': '#7f7f7f', 'Happy': '#40E0D0', 'Sad': '#1f77b4', 'Angry': '#CA3435',
                  'Surprise': '#9467bd', 'Fear': '#9370DB', 'Disgust': '#ff7f00'}

    # Update the color palette
    cb_palette = UpdateCBPalette(emotion_colors)

    # Create a trace for each emotion
    emotions = df_counts['emotion'].unique()
    traces = []
    for i, emotion in enumerate(emotions):
        trace = go.Scatter3d(
            x=df_counts.loc[df_counts['emotion'] == emotion, 'frame'],
            y=np.full(len(df_counts.loc[df_counts['emotion'] == emotion]), i),
            z=df_counts.loc[df_counts['emotion'] == emotion, 'probability'],
            mode='markers',
            name=emotion,
            marker=dict(size=5, color=cb_palette[i % len(cb_palette)], opacity=0.8)
        )
        traces.append(trace)

    # Create the layout for the plot
    layout = go.Layout(
        title=dict(
            text='<b>Emotions Count over Frames</b>',
            x=0.5,
            y=0.9
        ),
        scene=dict(
            xaxis=dict(title='Frame'),
            yaxis=dict(title='Emotion'),
            zaxis=dict(title='probability')
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    # Create the plot
    fig = go.Figure(data=traces, layout=layout)

    return fig





def hist_max_prob_emo(df_plotting):

    max_prob_rows = df_plotting.groupby('frame')['probability'].idxmax().reset_index()
    max_prob_df = df_plotting.loc[max_prob_rows['probability']]
    feeling_counts = max_prob_df.groupby('emotion')['frame'].nunique()
    max_prob_df

    # Define a dictionary with the desired color for each emotion
    emotion_colors = {'Neutral': '#7f7f7f', 'Happy': '#40E0D0', 'Sad': '#1f77b4', 'Angry': '#CA3435',
                  'Surprise': '#9467bd', 'Fear': '#9370DB', 'Disgust': '#ff7f00'}

    cb_palette = UpdateCBPalette(emotion_colors)

    # Define palette 
    #cb_palette = ['#40E0D0', '#ff7f00', '#9370DB', '#CA3435', '#1f77b4', '#9467bd', '#7f7f7f']
    #CA3435
    # Create the bar plot
    fig = px.bar(max_prob_df, x='frame', y='probability', color='emotion', 
                color_discrete_sequence=cb_palette, hover_data={"text": max_prob_df['emotion']})

    # Update the layout
    fig.update_layout(   
        title={
            'text': '<b>Maximum Probability of Emotion per Frame</b>',
            'font': {'size': 24, 'color': 'black'},
            'x': 0.5,
            'y': 0.9,
            'yanchor': 'middle'
        },
        xaxis_title='Frame',
        yaxis_title='Probability',
        legend_title='Emotion',
        font=dict(family='Arial', size=14),
        margin=dict(l=50, r=50, t=100, b=50),
        plot_bgcolor='white'
    )

    # Update the legend with customizations
    fig.update_traces(
        hovertemplate='<br>'.join([
            'Emotion: %{fullData.name}',
            'Frame: %{x}',
            'Probability: %{y:.2f}'
        ]),
        hoverlabel=dict(bgcolor='white', font_size=14),
        showlegend=True,
        hoverinfo='all'
    )
    return fig


def radar_chart(df_plotting):

    max_emotions = df_plotting.apply(get_max_emotion, axis=1)
    emotions_counts = max_emotions['emotion'].value_counts().reset_index()

    # Define a dictionary with the desired color for each emotion
    emotion_colors = {'Neutral': '#7f7f7f', 'Happy': '#40E0D0', 'Sad': '#1f77b4', 'Angry': '#CA3435',
                  'Surprise': '#9467bd', 'Fear': '#9370DB', 'Disgust': '#ff7f00'}

    cb_palette = UpdateCBPalette(emotion_colors)

    # Define the color palette
    #cb_palette = ['#40E0D0', '#ff7f00', '#9370DB', '#CA3435', '#1f77b4', '#9467bd', '#7f7f7f']
    # create the plot
    fig = go.Figure(data=go.Scatterpolar(
        r=emotions_counts['count'].tolist(),
        theta=emotions_counts['emotion'].tolist(),
        fill='toself',
        name='',
        line_color=cb_palette[0],
        fillcolor=cb_palette[0],
        hovertemplate='%{theta}: %{r}'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(emotions_counts['count'])],
                showticklabels=False,
                showgrid=False,
                ticks=''
            ),
            angularaxis=dict(
                tickfont=dict(size=14),
                rotation=90,
                direction='clockwise'
            )
        ),
        showlegend=False,
        height=400,
        margin=dict(t=80, b=50, l=50, r=50),
        font=dict(size=16)
    )

    fig.update_layout(
        title={
            'text': "<span style='font-size: 24px'>Prevalence Of Feelings During The Video</span>",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        paper_bgcolor='white',
        plot_bgcolor='white',
    )

    return fig

