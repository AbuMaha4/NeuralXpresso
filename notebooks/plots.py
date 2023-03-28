import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import cv2


def get_pallette():
    return ['#CA3435', '#ff7f00', '#9370DB', '#40E0D0', '#1f77b4', '#9467bd', '#7f7f7f']



def get_overview_normalized(df_video):
    df_video['normalized_probability'] = df_video.groupby('frame')['probability'].apply(lambda x: x / x.sum())

    fig = px.bar(df_video, x="frame", y='normalized_probability', color='emotion',
              barmode="stack", title="Distribution of Emotions across Frames")
    fig.update_layout(xaxis_title="Frame", yaxis_title="Emotion Distribution",
                   title={
                       'y': 0.9,
                       'x': 0.5,
                       'xanchor': 'center',
                       'yanchor': 'top'
                   })

    return fig


def get_df_single_person(df_video, ID):
    df_single_person = df_video.loc[df_video['person_ID'] == ID].copy()
    df_single_person.loc[:, 'moving_avg'] = df_single_person['probability'].rolling(window=10, center=True).mean()
    return df_single_person

def get_df_radar(df_single_person):
    # Apply the function to each row of the DataFrame
    max_prob_rows = df_single_person.groupby('frame')['probability'].idxmax().reset_index()
    max_prob_df = df_single_person.loc[max_prob_rows['probability']]
    df_radar = max_prob_df['emotion'].value_counts().reset_index()

    # List of all possible emotions
    all_emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    # Reindex the DataFrame with all possible emotions and fill missing values with 0
    complete_df = df_radar.set_index('index').reindex(all_emotions, fill_value=0).reset_index()

    complete_df.columns = ['index', 'emotion']

    return complete_df



def get_emotion_landscape(df_single_person):
    # Create the plot 
    cb_palette = get_pallette()
    fig = px.area(df_single_person, x="frame", y="moving_avg", color="emotion",
                color_discrete_sequence=cb_palette, hover_data={"text": df_single_person['emotion']})

    # Update the layout 
    fig.update_layout(
        title={
            'text': 'Probability of emotion per frame',
            'font': {'size': 24, 'color': 'black'}
            #'x': 0.5, #center header
            #'y': 0.95 #center header
        },
        xaxis_title='Frames',
        yaxis_title='Probability',
        legend_title='Emotion',
        font=dict(family='Arial', size=14),
        margin=dict(l=50, r=50, t=100, b=50),
        plot_bgcolor='white',
        xaxis=dict(dtick=1),  #set the x-axis step 
        yaxis=dict(
        title='Probability',
        title_standoff=3
    )
    )
    

    # Update the legend with customizations
    fig.update_traces(
        hovertemplate='<br>'.join([
            'Emotion: %{fullData.name}',
            'Frame: %{x:.2f} ',
            'Probability: %{y:.2f}'
        ]),
        hoverlabel=dict(bgcolor='white', font_size=14),
        line=dict(width=2),
        showlegend=True,
        stackgroup='one',
        hoverinfo='all',

    )

    return fig


def get_radar_plot(df_radar):
    cb_palette = get_pallette()


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
        height=300,
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


def get_character_overview(df_video, ID, result):

    df_single_person= get_df_single_person(df_video, ID)
    df_radar = get_df_radar(df_single_person)



    fig_radar = get_radar_plot(df_radar)
    image_array = result['portraits'][ID]
    fig_area = get_emotion_landscape(df_single_person)

    # Create a 3x1 grid of subplots with custom column widths
    fig = make_subplots(rows=1, cols=3, column_widths=[0.2, 0.6, 0.2], specs=[[{'type': 'image'}, {'type': 'xy'}, {'type': 'polar'}]], horizontal_spacing=0.1, vertical_spacing=0.2)


    # Add trace1, trace2, and the image to their respective subplots
    fig.add_trace(px.imshow(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB), ).data[0], row=1, col=1)


    for i in range(7):
        fig.add_trace(fig_area.data[i], row=1, col=2)

        

    fig.add_trace(fig_radar.data[0], row=1, col=3)


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
            title_font=dict(size=16)
            ),
            margin=dict(t=80, b=50, l=50, r=50),
            font=dict(size=16)
        )


# Update layout
    #fig.update_layout(title='Character Overview')

    fig.update_xaxes(title_text="Frame", row=1, col=2)
    fig.update_yaxes(title_text="Probability", row=1, col=2)

    fig.update_xaxes(title_text="Emotion Counts", row=1, col=3)
    fig.update_yaxes(title_text="Prevalence", row=1, col=3)

    # Add subtitles for each subplot
    annotations = [
        dict(
            x=0.02,
            y=1.17,
            xref='paper',
            yref='paper',
            text=f'Character: {ID}',
            showarrow=False,
            font=dict(size=18)
        ),
        dict(
            x=0.50,
            y=1.17,
            xref='paper',
            yref='paper',
            text='Emotion-Landscape over Frames',
            showarrow=False,
            font=dict(size=18)
        ),
        dict(
            x=1.0,
            y=1.17,
            xref='paper',
            yref='paper',
            text='Emotion Prevalence',
            showarrow=False,
            font=dict(size=18)
        ),
    ]
    fig.update_layout(annotations=annotations)

    # Show the figure
    return fig