import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import cv2


def get_cb_palette(emotions):
    """
    Return a plotly cb_palette mapping emotions to colors.
    
    Arguments:
    emotions -- a list of emotions to map to colors
    
    Returns:
    A plotly cb_palette object
    """
    color_map = {
        'Angry': '#ff0000',
        'Disgust': '#ffff00',
        'Fear': '#ffa500',
        'Sad': '#FC00CC',
        'Neutral': '#00ff00',
        'Happy': '#008080',
        'Surprise': '#0000ff'
    }
    return [color_map[emotion] for emotion in emotions]


def overview_plot(df_video):
    '''
    intakes df_video
    outputs: Emotions over whole video, independent of character. Just the time-series
  
    '''
    emotions = df_video.emotion.unique().tolist()
    cb_palette = get_cb_palette(emotions)

    grouped_df = df_video.groupby(['frame', 'emotion'])['probability'].sum().unstack()
    normalized_df = grouped_df.div(grouped_df.sum(axis=1), axis=0)
    normalized_df = normalized_df.rolling(window=10).mean().dropna()
    normalized_df.index = normalized_df.index.astype(str)

    fig = px.area(normalized_df, x=normalized_df.index, y=normalized_df.columns,
                color_discrete_sequence=cb_palette)
    
 # Update the layout
    fig.update_layout(   
        title={
            'text': 'Emotion Probability per frame',
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

def get_aggregated_emotion_counts(df_video, df_character):
  
    '''
    This outputs the df which we need for the general radar plot
    '''

    # Initialize an empty DataFrame with all emotions as index
    aggregated_counts = pd.DataFrame(columns=['index', 'emotion'])
    aggregated_counts['index'] = ['Angry', 'Disgust', 'Fear', 'Sad', 'Neutral', 'Happy', 'Surprise']
    aggregated_counts['emotion'] = 0

    for ID in df_character.person_ID:
        if (df_character.loc[df_character['person_ID'] == ID].appearances.values[0] > (0.2 * df_character.appearances.sum())):
            # Iterate over all unique person IDs and sum their emotion counts
            df_single_person = get_df_single_person(df_video, ID=ID)
            df_radar = get_df_radar(df_single_person)
            aggregated_counts['emotion'] += df_radar['emotion']

    return aggregated_counts

def get_radar_plot_overview(df_aggregated_emotion_counts):
    
    '''
    this plots the general Radar plot
    '''
    
    emotions = ['Angry', 'Disgust', 'Fear', 'Sad', 'Neutral', 'Happy', 'Surprise']
    cb_palette = get_cb_palette(emotions)


    fig = go.Figure(data=go.Scatterpolar(
        r=df_aggregated_emotion_counts['emotion'],
        theta=df_aggregated_emotion_counts['index'],
        fill='toself',
        name='Emotion Counts',
        line_color=cb_palette[0],
        hovertemplate='%{theta}: %{r}',
        showlegend=False
    ))
    fig.update_xaxes(title_text="Person")

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(df_aggregated_emotion_counts['emotion'])],
                showticklabels=False,
                showgrid=True,
                ticks=''
            ),
            angularaxis=dict(
                tickfont=dict(size=14),
                rotation=90,
                direction='clockwise'
            )
        ),
        showlegend=False,
        height=300,
        margin=dict(t=80, b=50, l=50, r=50),
        font=dict(size=16),
        title={
            'text': "Overall Prevalence Of Feelings During The Video",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        paper_bgcolor='white',
        plot_bgcolor='white'
    )

    return fig



def get_df_single_person(df_video, ID):
    '''
    This returns a dataframe for a single person
    '''

    df_single_person = df_video[df_video['person_ID'] == ID].copy()


    all_emotions = ['Angry', 'Disgust', 'Fear', 'Sad', 'Neutral', 'Happy', 'Surprise']

    for e in all_emotions:
        frames_with_data = df_single_person[df_single_person['emotion'] == e]['frame'].values
        first_frame = frames_with_data[0]
        last_frame = frames_with_data[-1]

        all_frames = np.arange(first_frame, last_frame+1)

        missing_frames = set(all_frames) - set(frames_with_data)

        df_missing = pd.DataFrame()
        df_missing['frame'] = list(missing_frames)
        df_missing['person_ID'] = ID
        df_missing['emotion'] = e
        df_missing['probability'] = 0

        df_single_person = pd.concat([df_single_person, df_missing], axis=0)

    df_single_person.sort_values(by=['emotion', 'frame'], inplace=True)

    df_single_person.loc[:, 'moving_avg'] = df_single_person['probability'].rolling(window=5, center=True).mean().fillna(method='ffill').fillna(method='bfill')

    return df_single_person

def get_df_radar(df_single_person):

    '''
    This gets the emotion counts for one character
    '''
    # Apply the function to each row of the DataFrame
    df_single_person = df_single_person[df_single_person.probability > 0]
    max_prob_rows = df_single_person.groupby('frame')['probability'].idxmax().reset_index()
    max_prob_df = df_single_person.loc[max_prob_rows['probability']]
    df_radar = max_prob_df['emotion'].value_counts().reset_index()

    # List of all possible emotions
    all_emotions = ['Angry', 'Disgust', 'Fear', 'Sad', 'Neutral', 'Happy', 'Surprise']

    # Reindex the DataFrame with all possible emotions and fill missing values with 0
    df_radar = df_radar.set_index('index').reindex(all_emotions, fill_value=0).reset_index()

    df_radar.columns = ['index', 'emotion']

    return df_radar




def get_emotion_landscape(df_single_person):

    '''
    This is the plot for the time series for a single character
    '''
    # Create the plot 
    emotions = df_single_person.emotion.unique().tolist()
    cb_palette = get_cb_palette(emotions)

    fig = px.area(df_single_person, x="frame", y="probability", color="emotion",
                color_discrete_sequence=cb_palette, hover_data={"text": df_single_person['emotion']})

    # Update the layout
    fig.update_layout(   
        title={
            'text': 'Emotion Probability per frame',
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





def get_radar_plot(df_radar):


    '''
    This is the radar plot for a single character because df_radar was generated using df_single_character
    '''
    emotions = ['Angry', 'Disgust', 'Fear', 'Sad', 'Neutral', 'Happy', 'Surprise']
    cb_palette = get_cb_palette(emotions)


    fig = go.Figure(data=go.Scatterpolar(
        r=df_radar['emotion'],
        theta=df_radar['index'],
        fill='toself',
        name='Emotion Counts',
        line_color=cb_palette[0],
        hovertemplate='%{theta}: %{r}',
        showlegend=False
    ))
    fig.update_xaxes(title_text="Person")

    fig.update_layout(
        polar=dict(
            
            radialaxis=dict(
                visible=True,
                range=[0, max(df_radar['emotion'])],
                showticklabels=False,
                showgrid=True,
                ticks=''
            ),
            angularaxis=dict(
                tickfont=dict(size=14),
                rotation=90,
                direction='clockwise'
            )
        ),
        showlegend=False,
        height=700,
        margin=dict(t=80, b=50, l=50, r=50),
        font=dict(size=16),
        title={
            'text': "Prevalence Of Feelings During The Video",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        paper_bgcolor='white',
        plot_bgcolor='white'
    )

    return fig


def get_strongest_emotions_plot(df):


    df_max_rows = df.groupby('frame')['probability'].idxmax().reset_index()
    df_max_probs = df.loc[df_max_rows['probability']]

    emotions = df_max_probs.emotion.unique().tolist()
    cb_palette = get_cb_palette(emotions)

    fig = px.bar(df_max_probs, x='frame', y='probability', color='emotion', 
                color_discrete_sequence=cb_palette, hover_data={"text": df_max_probs['emotion']})

    # Update the layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                showticklabels=False,
                
            ),
            angularaxis=dict(
                tickfont=dict(size=14),
                rotation=90,
                direction='clockwise'
            )
        ),
        showlegend=True,
        legend=dict(
        x=1.1,
        y=0.5,
        xanchor='left',
        yanchor='middle',
        title_font=dict(size=16),
        orientation='v',
        traceorder='reversed'
        ),
        margin=dict(t=80, b=50, l=50, r=50),
        font=dict(size=16)
    )

    annotations = [
        dict(
            x=0.13,
            y=1.17,
            xref='paper',
            yref='paper',
            text='Strongest Emotion Per Frame',
            showarrow=False,
            font=dict(size=18)
        ),
    ]
    fig.update_layout(annotations=annotations)

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




def get_character_overview(df_video, ID, result):

    '''
    This stacks together:
        1. Portrait
        2. Timeseries for 1 character
        3. Radar_plot for 1 character

        Iteration over characters happens in app.py
    '''

    
    df_radar = get_df_radar(df_video)


    fig_radar = get_radar_plot(df_radar)
    fig_area = get_emotion_landscape(df_video)
    print(df_video)
    # Create a 2x2 grid of subplots
    fig = make_subplots(rows=2, cols=2, specs=[[{'type': 'image'}, {'type': 'polar'}],[{'type': 'xy', 'colspan':2},None]], horizontal_spacing=0.1, vertical_spacing=0.15)


    # Add trace1, trace2, and the image to their respective subplots
    image_array = result['portraits'][ID]
    fig.add_trace(px.imshow(image_array, ).data[0], row=1, col=1)
    fig.add_trace(fig_radar.data[0], row=1, col=2)

    for i in range(7):
        fig.add_trace(fig_area.data[i], row=2, col=1)

        
    fig.update_layout(
            polar=dict(
                domain=dict(y=[0.5, 1]),
                radialaxis=dict(
                    showticklabels=False,  
                ),
                angularaxis=dict(
                    tickfont=dict(size=14),
                    rotation=90,
                    direction='clockwise'
                )
            ),

            showlegend=True,
            legend=dict(
            x=1.1,
            y=0,
            xanchor='left',
            yanchor='bottom',
            title_font=dict(size=12),
            orientation='v',
            traceorder='reversed'
            ),

            margin=dict(t=80, b=50, l=50, r=50),
            font=dict(size=12),

            # Add the title above the image
            annotations=[
                dict(
                    text=f'Character: {ID}',
                    xref='x domain',
                    yref='y domain',
                    x=0.5,
                    y=1.2,
                    showarrow=False,
                    font=dict(size=16),
                )
            ],
        )


    fig.update_xaxes(showticklabels=False, zeroline=False, visible=False, row=1, col=1)
    fig.update_yaxes(showticklabels=False, zeroline=False, visible=False, row=1, col=1)

    fig.update_xaxes(title_text="Frame Count", title_font=dict(size=12), title_standoff=8, row=2, col=1)
    fig.update_yaxes(title_text="Emotion Probability", title_font=dict(size=12), title_standoff=8, tickmode='linear', dtick=0.2, row=2, col=1)

    fig.update_xaxes(title_text="Emotion Counts", row=1, col=3)
    fig.update_yaxes(title_text="Prevalence", row=1, col=3)



    annotations = [
        dict(
            x=0.13,
            y=1.17,
            xref='paper',
            yref='paper',
            text=f'Emotion-Landscape over Frames for Character {ID}',
            showarrow=False,
            font=dict(size=18)
        ),
    ]
    fig.update_layout(annotations=annotations)

    return fig


def get_overall_overview(df_video):


    '''
        This stacks together the general time series and the general radar plot
    '''

  #  df_aggregated_emotion_counts = get_aggregated_emotion_counts(df_video, df_character)
    
    df_radar = get_df_radar(df_video)


    fig_radar = get_radar_plot(df_radar)
    
    
    fig_area = get_emotion_landscape(df_video)
    
    
    fig = make_subplots(rows=1, cols=2, column_widths=[0.7, 0.3], specs=[[{'type': 'xy'}, {'type': 'polar'}]], horizontal_spacing=0.1, vertical_spacing=0.2)

    # Add each trace object in fig_normalized.data to the subplot one by one


    for i in range(7):
        fig.add_trace(fig_area.data[i], row=1, col=1)

    fig.add_trace(fig_radar.data[0], row=1, col=2)


    fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    showticklabels=False,
                    
                ),
                angularaxis=dict(
                    tickfont=dict(size=14),
                    rotation=90,
                    direction='clockwise'
                )
            ),
            showlegend=True,
            legend=dict(
            x=1.1,
            y=0.5,
            xanchor='left',
            yanchor='middle',
            title_font=dict(size=16),
            orientation='v',
            traceorder='reversed'
            ),
            margin=dict(t=80, b=50, l=50, r=50),
            font=dict(size=16)
        )


    # Update axis titles
    fig.update_xaxes(title_text="Frame Count", row=1, col=1)
    fig.update_yaxes(title_text="Emotion Probability", row=1, col=1)
    fig.update_xaxes(title_text="Emotion Counts", row=1, col=2)
    fig.update_yaxes(title_text="Prevalence", row=1, col=2)

    # Add subtitles for each subplot
    annotations = [
        dict(
            x=0.13,
            y=1.17,
            xref='paper',
            yref='paper',
            text='Emotion-Landscape over Frames',
            showarrow=False,
            font=dict(size=18)
        ),
    ]
    fig.update_layout(annotations=annotations)

    # Show the figure
    return fig