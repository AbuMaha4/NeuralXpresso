![pulsar_star_wp](nx_header.png)

## Project Overview
### <span style="color: #21618C;"> **Objective:** </span>
**NeuralXpresso** is designed to revolutionize the way we analyze video content. By harnessing the power of advanced machine learning techniques, NeuralXpresso can detect faces, recognize characters, and interpret a wide range of emotions in video data, offering unparalleled insights and analytics.

### <span style="color: #21618C;"> **Context:**  </span> 
Analyzing video is no small feat. Videos are dynamic and constantly changing, with varying lighting conditions, angles, and resolutions. These challenges make it difficult to achieve accurate and consistent results. **NeuralXpresso** addresses these complexities head-on, providing a robust solution that adapts to different video environments and accurately identifies emotional cues that often go unnoticed.

### <span style="color:#21618C;"> **Significance:**  </span> 
The potential applications of **NeuralXpresso** are vast. In **security**, it can enhance surveillance systems by accurately identifying individuals and their emotional states. In **media analysis**, it can provide deeper insights into viewer reactions and character dynamics. For **user experience research**, it can track emotional engagement with content, helping to create more immersive and effective experiences. Accurate video analysis not only pushes the boundaries of current research but also opens up new possibilities for practical applications.

### <span style="color:#21618C;"> **Goal:**  </span> 
**NeuralXpresso's** ultimate aim is to be your go-to tool for video content analysis. By accurately identifying key characters and interpreting emotions with high precision, it provides a detailed and nuanced understanding of video content. This level of insight can drive better decision-making, more targeted research, and innovative applications across various fields.


## Team Members

- Team Member 1: [Ben Köhler](https://github.com/Ben6586)
- Team Member 2: [Larissa Roth](https://github.com/LarissaRoth)
- Team Member 3: [Mohan Mukunda](https://github.com/Mohan263)
- Team Member 4: [Steve Kelly](https://github.com/Skelly85)
- Team Member 5: [Maha Abu-Khousa](https://github.com/AbuMaha4)



## Python Scripts

This project consists of several scripts that serve different purposes:

1. **app.py**: The app.py file sets up the user interface for NeuralXpresso using the Dash framework, providing controls for video input, resolution, and face detection settings. It defines callbacks to process video input, update video statistics, and run face and emotion detection analyses. The results are visualized and displayed using Plotly, creating an interactive video analysis tool.

2. **neuralxpresso.py**: The neuralxpresso.py file contains the core logic for processing and analyzing video content in the NeuralXpresso application. It defines the NeuralXpressoSession class, which handles downloading YouTube videos, processing frames, detecting faces, and recognizing emotions. The file also includes auxiliary classes such as VideoProcessor, FaceDetector, EmotionDetector, and PersonIdentifier to manage various aspects of the video analysis pipeline, including face detection, emotion detection, and character identification.

3. **plots.py**: The plots.py file provides functions for creating various visualizations used in the NeuralXpresso application. It defines functions to generate different types of plots, including:

- Overview Plot: Visualizes the overall emotion probabilities across all frames in the video.
- Radar Plots: Creates radar charts to show the distribution of detected emotions for characters.
- Emotion Landscape: Produces time series plots to display the probability of each emotion over time for individual characters.
- Strongest Emotions Plot: Generates bar charts highlighting the strongest emotions detected in each frame. These functions use Plotly to create interactive and visually appealing plots that help users understand the results of the video analysis.

4. **ModelEvaluation.ipynb**: This notebook evaluates the final models on a test dataset, providing detailed metrics and visualizations to assess their performance.

## Installation and Setup

To set up the project locally, follow these steps:

1. Clone the repository:
    ```sh
    git clone git@github.com:AbuMaha4/NeuralXpresso.git
    ```
2. Navigate to the project directory:
    ```sh
    cd NeuralXpresso
    ```
3. Create Virtual environment 
    ```sh
    conda create --name neuralxpresso python=3.8
    conda activate neuralxpresso
    ```
4. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```
4. Run the app.py file in order ro start the dashboard 
    ```sh
    python app.py 
    ```

    And click on the Server Link that is shown in the terminal. It shouls look like this: http://127.0.0.1:8050/


If you want to run neuralxpresso locally, you can use the tests.ipynb notebook, paste your link and follow the instructions there. 


### Demonstration 

<video width="640" height="480" controls>
  <source src="Output_video.mp4" type="video/mp4">
</video>

## Dataset

The dataset used for NeuralXpresso was obtained from various sources, including YouTube for video content and publicly available datasets for face and emotion recognition. Download links for the datasets are provided where publicly available.

## Attribute Information

The dataset contains the following attributes:

1. **Video URL**: The URL of the YouTube video.
2. **Frame Number**: The specific frame number in the video.
3. **Face Bounding Box**: Coordinates of the detected face in the frame.
4. **Emotion Probabilities**: Probabilities for each detected emotion (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral).
5. **Character ID**: Identifier for the recognized character.

The dataset contains a total of xxx examples, with xxx positive examples and xxx negative examples.

## EDA/Cleaning

In the EDA and data cleaning phase, we focused on:

- Removing duplicate frames.
- Normalizing face bounding box coordinates.
- Handling missing values in emotion probabilities.
- Ensuring consistent labeling of character IDs.

## Model Choices

We used several pretrained Models models for face detection and emotion recognition, including:

- **MTCNN**: For robust face detection.

    - Source: We used the MTCNN implementation from the facenet-pytorch library.
    - Purpose: MTCNN is utilized for robust face detection. It excels in detecting faces with varying orientations and scales.
    - Training Details: MTCNN is a deep learning-based face detection algorithm that uses a cascading technique to detect faces in images. It was trained on large datasets of facial images, such as the CelebA dataset, to learn to detect faces accurately.

- **Haar Cascade**: For real-time face detection.

    - Source: The Haar Cascade classifiers are available in the OpenCV library.
    - Purpose: Haar Cascade is used for real-time face detection. It is a classical machine learning approach that provides quick detection, making it suitable for real-time applications.
    - Training Details: Haar Cascade classifiers are trained using a set of positive images (images with faces) and negative images (images without faces). The training process involves applying Haar-like features to the images to train the classifier to detect faces.

- **Face Recognition**: For character identification.
    - Source: We used the face_recognition library, which is built on top of dlib's state-of-the-art face recognition model.
    - Purpose: The Face Recognition model is used for identifying and verifying characters in the video by encoding faces into a 128-dimensional vector space.
    - Training Details: The model was trained using deep convolutional neural networks on a large dataset of facial images, such as the Labeled Faces in the Wild (LFW) dataset. The neural network learns to map facial images to a 128-dimensional embedding space, where the distance between embeddings represents the similarity between faces.

- **Emotion Detection CNN**: 

    - Purpose: This model is designed to detect and classify emotions from facial expressions in video frames.
    - Training Details: The CNN was trained on a dataset of facial images labeled with different emotions, such as the FER-2013 (Facial Expression Recognition) dataset. The training process involved preprocessing the images (e.g., resizing, grayscale conversion), data augmentation to enhance the dataset, and training the CNN to recognize and classify different emotional expressions accurately.

Links to the Models will be provided. 

## Results

The NeuralXpresso project demonstrated impressive accuracy in emotion detection, with many emotions being correctly identified just by judging with the human eye. Although we lacked a labeled video to test the model precisely, the qualitative results were promising. We plan to incorporate labeled data for precise evaluation in the future. However, we encountered limitations such as dependency on pre-trained models, which may not generalize well to all video conditions, and computational resource demands, which can affect real-time analysis performance. Despite these challenges, NeuralXpresso shows great potential for robust video content analysis.



## Final Remarks

NeuralXpresso provides a comprehensive tool for video analysis, combining state-of-the-art techniques for face detection and emotion recognition. We believe it can significantly enhance applications in various fields, providing deeper insights and more accurate analyses.


## Preesentation 

<video width="640" height="480" controls>
  <source src="Neuralexpresso.mp4" type="video/mp4">
</video>