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
    cb_palette = ['#40E0D0', '#ff7f00', '#9370DB', '#CA3435', '#1f77b4', '#9467bd', '#7f7f7f']

    # Define a dictionary with the desired color for each emotion
    emotion_colors = {'Neutral': '#CA3435', 'Happy': '#00ff00', 'Sad': '#0000ff', 'Angry': '#ff0000',
                    'Surprise': '#ffff00', 'Fear': '#800080', 'Disgust': '#ffa500'}

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

    # Show the plot
    fig.show()

ThreeDimensionalGraph(df_plotting)