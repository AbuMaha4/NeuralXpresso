import cv2
from facenet_pytorch import MTCNN
import face_recognition

from pytube import YouTube
import cv2
import numpy as np
import pandas as pd
from keras.models import load_model
import urllib.request

class NeuralXpressoSession:
    def __init__(
            self,
            yt_link
    ):
       self.video_processor = VideoProcessor(yt_link)
         
    def run_analysis(self,
                     skip_frames = 20,
                     #batch_size = 10000,
                     video_output = False,
                     face_detector_type = 'face_recognition',
                     emotion_detector_offset = 1.4,
                     face_recognition_offset = 1,
                     face_recognition_sensitivity = 0.6,
                     main_character_threshold = 0.1,
                     emotion_interpolation_frames_threshold = 2                     
                     ):
        
        # non-permanent solution, currently simply all frames = batch-size
        batch_size = None

        self.main_character_threshold = main_character_threshold
        self.emotion_interpolation_frames_threshold = emotion_interpolation_frames_threshold
        self.initialize_tools(skip_frames, batch_size, video_output, face_detector_type, emotion_detector_offset, face_recognition_offset, face_recognition_sensitivity)
        result_dict = self.process_video()
        
        return result_dict
        
    def initialize_tools(self, skip_frames, batch_size, video_output, face_detector_type, emotion_detector_offset, face_recognition_offset, face_recognition_sensitivity):
        
        self.video_processor.update_video_info_during_tool_initialization(skip_frames, batch_size, video_output)

        self.face_detector = FaceDetector(detection_type=face_detector_type)
        self.emotion_detector = EmotionDetector(box_offset = emotion_detector_offset,
                                                       frame_width = self.video_processor.width, 
                                                       frame_height = self.video_processor.height)
        self.person_identifier = PersonIdentifier(threshold=face_recognition_sensitivity, 
                                                         box_offset = face_recognition_offset,
                                                         frame_width = self.video_processor.width, 
                                                         frame_height = self.video_processor.height)

    def process_video(self):
        frame_nr = 0

        if self.video_processor.video_output == False:
            for batch in self.video_processor.get_batches():
                for frame in batch:
                    frame_nr +=1
                    face_boxes = self.face_detector.detect_faces(frame)
                    
                    if len(face_boxes) == 0:
                        self.video_processor.frames_without_faces += 1
                        continue
                    
                    for box in face_boxes: 
                        prob = self.emotion_detector.detect_emotion(frame, box)
                        character_id = self.person_identifier.identify_person(frame, box)

                        self.video_processor.video_emotions_by_frame.append(prob)
                        self.video_processor.video_info_by_frame.append((frame_nr, character_id))

        else:
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')    
            writer =  cv2.VideoWriter('output_video_new.mp4', fourcc, 10, (self.video_processor.width, self.video_processor.height)) 
            
            for batch in self.video_processor.get_batches():
                for frame in batch:
                    frame_nr +=1
                    face_boxes = self.face_detector.detect_faces(frame)
                    
                    if len(face_boxes) == 0:
                        self.video_processor.frames_without_faces += 1
                        writer.write(frame)
                        continue
                    
                    for box in face_boxes: 
                        prob = self.emotion_detector.detect_emotion(frame, box)
                        character_id = self.person_identifier.identify_person(frame, box)

                        self.video_processor.video_emotions_by_frame.append(prob)
                        self.video_processor.video_info_by_frame.append((frame_nr, character_id))                 
                    
                        frame = self.draw_text_on_frame(frame, box, prob, character_id)

                    writer.write(frame)
            
            writer.release() 

        self.video_processor.total_frames_processed_actual = frame_nr

        self.process_results()

        video_overview = self.get_video_overview()
        plottable_video_overview = pd.melt(video_overview, id_vars=['frame', 'person_ID'], value_vars=EmotionDetector.get_emotion_categories(), var_name='emotion', value_name='probability')
        person_overview = self.get_person_overview(video_overview)
        portraits = self.get_portraits()

        return {'video_overview': plottable_video_overview, 
                'character_overview': person_overview,
                'portraits': portraits, 
                'new_export': self.results
                }

        return self.results
    
    def draw_text_on_frame(self, frame, box, prob, character_id):
        max_emotion, max_prob = np.argmax(prob), np.max(prob)
        emotion_text = EmotionDetector.get_emotion_categories()[max_emotion]

        y_min, y_max, x_min, x_max = box

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, f"Prob: {max_prob:.1%}", (x_min, y_max + 60), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
        cv2.putText(frame, f"{emotion_text}", (x_min, y_max + 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
        cv2.putText(frame, f"ID: {character_id}", (x_min, y_min -20), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)

        return frame
        
    
    def get_video_overview(self):
        """
        Generate an overview DataFrame of the video, showing the emotions detected for each frame and the corresponding
        person ID.

        Returns:
        df (pd.DataFrame): DataFrame showing the emotions detected for each frame and the corresponding person ID.
        """

        df_emotions = pd.DataFrame(self.video_processor.video_emotions_by_frame, columns=EmotionDetector.get_emotion_categories())
        df_frame_info = pd.DataFrame(self.video_processor.video_info_by_frame, columns=['frame', 'person_ID'])
        df = pd.concat([df_emotions, df_frame_info], axis=1)
        
        return df
    
    def get_person_overview(self, video_overview_df):
        """Generate an aggregated overview of each person's emotions and appearances in the video.
        
        Args:
        video_overview_df (pandas.DataFrame): DataFrame containing emotions, frames, and person ID for each frame
        
        Returns:
        pandas.DataFrame: DataFrame containing aggregated information of each person's emotions and appearances
        """

        agg_funcs = {'frame': 'count', 'Angry': 'mean', 'Disgust': 'mean', 'Fear': 'mean',
                    'Happy': 'mean', 'Sad': 'mean', 'Surprise': 'mean', 'Neutral': 'mean'}
        df_agg = video_overview_df.groupby('person_ID').agg(agg_funcs).reset_index()
        df_agg = df_agg.rename(columns={'frame': 'appearances'})

        return df_agg
    
    def get_portraits(self):
        portraits_dict = {}
        for person in self.person_identifier.persons:
            portraits_dict[person.id] = person.representative_portrait
        return portraits_dict
    
    def process_results(self):

        self.results = {}
        self.results['main_character_data'] = {}

        self.extract_main_characters()
        df = self.get_video_overview()

        self.results['overview_mean'] = self.get_timeline_df(df, ID = None)
        for main_char in self.main_characters[:,0]:
            self.results['main_character_data'][main_char] = self.get_timeline_df(df, ID = [main_char])

    def get_timeline_df(self, df, ID = None):

        if not ID:
            reference_character_list = list(self.main_characters[:,0])
        else:
            reference_character_list = ID

        df_main = df[df['person_ID'].isin(reference_character_list)]
        # Group by frame and average emotions
        df_grouped = df_main.groupby('frame')[['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']].mean().reset_index()
        total_frames = pd.DataFrame({'frame': range(1, self.video_processor.total_frames_processed_actual+1)})
        # Merge the new DataFrame with the original DataFrame
        merged_df = pd.merge(total_frames, df_grouped, on='frame', how='left')

        interpolated_df = self.interpolate_emotion_in_empty_frames(merged_df)
        interpolated_df= interpolated_df.fillna(0)
        depivoted_df = pd.melt(interpolated_df, id_vars=['frame'], value_vars=EmotionDetector.get_emotion_categories(), var_name='emotion', value_name='probability')

        return depivoted_df

    def extract_main_characters(self):
        appearances_framewise = np.array(self.video_processor.video_info_by_frame)
        total_appearances = len(appearances_framewise)
        main_character_framecount_threshold = total_appearances*self.main_character_threshold
        person_frames = np.unique(appearances_framewise[:, 1], return_counts=True)
        
        main_characters_mask = person_frames[1] * main_character_framecount_threshold > total_appearances
        main_characters = person_frames[0][main_characters_mask]
        main_characters_appearances = person_frames[1][main_characters_mask]

        main_characters_array = np.column_stack((main_characters, main_characters_appearances))

        self.main_characters = main_characters_array

    def interpolate_emotion_in_empty_frames(self, df):

        # Compute the intermediate values using a Series, 'Neutral' chosen at random
        calc = pd.Series(np.where(~df['Neutral'].isna(), df['frame'], np.nan))

        # Compute the gap between consecutive values in the Series
        gap = calc.bfill() - calc.ffill()

        # Create a boolean mask based on the gap and emotion values, 'Neutral' chosen at random
        mask = (gap <= self.emotion_interpolation_frames_threshold+1) & df['Neutral'].isna()

        for emotion in EmotionDetector.get_emotion_categories():
            # Use the mask for interpolation
            df['Interpolated'] = df[emotion].interpolate(method='linear', limit_direction='forward', inplace=False, mask=mask)

            df[emotion]=np.where(mask==True, df['Interpolated'], df[emotion])

            df = df.drop(['Interpolated'], axis=1)

        return df


class VideoProcessor:
    def __init__(self, yt_link):
        self.yt_link = yt_link
        self.frames_without_faces = 0
        self.video_emotions_by_frame = []
        self.video_info_by_frame = []

        self.load_video_from_youtube()


    def load_video_from_youtube(self):

        # Download the YouTube video and get the highest resolution stream
        yt_video = YouTube(self.yt_link)
        stream = yt_video.streams.get_highest_resolution()

        # Open the video stream using OpenCV
        self.video = cv2.VideoCapture(stream.url)

        # Get available and used resolution - Debugging
        self.yt_available_resolutions = [streams.resolution for streams in yt_video.streams.filter(type="video", progressive=True)]
        #self.used_resolution = stream.resolution

        self.yt_title = yt_video.title
        self.yt_description = VideoProcessor.get_first_line(yt_video.description)
        self.yt_views = yt_video.views
        self.yt_rating = yt_video.rating
        self.yt_thumbnail_url = yt_video.thumbnail_url
        self.yt_keywords = yt_video.keywords

        # Get the number of frames in the video
        self.total_video_frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

        # Get the frame rate of the video
        self.video_fps = int(self.video.get(cv2.CAP_PROP_FPS))

        # Get the height and width of the video frames
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))

    def get_video_info(self):
        result = {}
        for attr_name in dir(self):
            if attr_name.startswith('yt_'):
                new_attr_name = attr_name[3:] # remove the 'yt_' prefix
                result[new_attr_name] = getattr(self, attr_name)
        
        result['total_frame_count'] = self.total_video_frame_count
        result['thumbnail'] = VideoProcessor.get_image_array_from_URL(self.yt_thumbnail_url)
        
        
        return result

    def update_video_info_during_tool_initialization(self, skip_frames, batch_size, video_output): 
        self.video_output = video_output
        self.skip_frames = skip_frames
        # Calculate the total number of frames to process after skipping frames
        self.total_frames_processed_predicted = len(range(0, self.total_video_frame_count, self.skip_frames))
        
        #non-permanent solution
        self.batch_size = int(self.total_frames_processed_predicted * 1.1)
        # Calculate the number of batches required to process all the frames
        self.num_batches_predicted = int(np.ceil(self.total_frames_processed_predicted / self.batch_size))   

    def get_batches(self):
        # Initialize an empty numpy array to hold the frames
        frames = np.empty((self.batch_size, self.height, self.width, 3), np.dtype('uint8'))

        self.original_frames_read = 0

        # Read the frames in batches and fill up the numpy array
        for batch_start in range(0, self.total_video_frame_count, self.batch_size * self.skip_frames):
            batch_end = min(batch_start + (self.batch_size * self.skip_frames), self.total_video_frame_count)
            batch_index = 0


            for i in range(batch_start, batch_end):
                ret, frame = self.video.read()
                self.original_frames_read +=1
                if not ret:
                    break

                if i % self.skip_frames == 0:
                    frames[batch_index] = frame
                    batch_index += 1

            # Resize the numpy array to fit the actual number of frames in the batch
            if batch_index < self.batch_size:
                frames = frames[:batch_index]

            # Yield the current batch of frames
            yield frames

        # Release the video stream
        self.video.release()

    @staticmethod
    def get_first_line(s):
        if not s:
            return None
        if '\n' in s:
            return s.split('\n', 1)[0]
        else:
            return s[:150].split(' ', 1)[0]
    
    @staticmethod
    def get_image_array_from_URL(url):
        with urllib.request.urlopen(url) as url_response:
            img_data = url_response.read()
            img_array = np.asarray(bytearray(img_data), dtype=np.uint8)
            img = None
            if url.endswith('.jpg') or url.endswith('.jpeg'):
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            elif url.endswith('.png'):
                img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
                if img.shape[2] == 4: # check if image has an alpha channel
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            
        return img



class FaceDetector:
    def __init__(self, detection_type="face_recognition"):
        #self.frame_width = frame_width
        #self.frame_height = frame_height
        self.detection_type = detection_type

        if self.detection_type == "MTCNN":
            # Initialize the MTCNN face detector
            self.face_detector = MTCNN(keep_all=True, post_process=False, margin=20)
        elif self.detection_type == "haarcascade":
            # Load the Haar Cascade face detector
            self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        elif self.detection_type == "face_recognition":
            # Package face_detector doesn't require object initialization
            self.face_detector = None
        else: 
            raise ValueError("Choose one of the implemented models, ya cunt!")


    def detect_faces(self, frame):
        if self.detection_type == 'haarcascade':
            frame_preprocessed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            boxes = self.face_detector.detectMultiScale(frame_preprocessed, scaleFactor=1.3, minNeighbors=3)
            
        elif self.detection_type == 'MTCNN':
            # No preprocessing needed
            boxes, _ = self.face_detector.detect(frame)
            
        elif self.detection_type == 'face_recognition':
            frame_preprocessed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Package face_detector directly return boxes without object initialization
            boxes = face_recognition.face_locations(frame_preprocessed, number_of_times_to_upsample=1)
        
        if (boxes is None) or (len(boxes) == 0):
            return [] 

        # Convert output boxes to normalized format
        norm_boxes = self.get_norm_boxes(boxes)

        valid_boxes = []
        for box in norm_boxes: 
            box_rearranged = np.array([box[0],box[3], box[1], box[2]])
            face_landmark = face_recognition.face_landmarks(frame, [box_rearranged])[0]
            if FaceDetector.valid_landmarks(face_landmark):
                # check for squared
                # box = self.augment_box(box, self.frame_width, self.frame_height, self.offset)
                valid_boxes.append(box) 

        return valid_boxes

    def get_norm_boxes(self, boxes):
        """
        Normalize the bounding box coordinates from MTCNN to numpy indexing format.
        Output format: np.array(y_min, y_max, x_min, x_max)
        """
        normalized_box = []

        if self.detection_type == 'haarcascade':
            for box in boxes:
                x, y, w, h = box
                normalized_box.append([y, y+h, x, x+w])
            return np.array(normalized_box)    

        elif self.detection_type == "MTCNN":
            for box in boxes:
                x_min, y_min, x_max, y_max = box.astype(int)
                normalized_box.append([y_min, y_max, x_min, x_max])
            return np.array(normalized_box)

        elif self.detection_type =='face_recognition':
            for box in boxes:
                y_min, x_max, y_max, x_min = box
                normalized_box.append([y_min, y_max, x_min, x_max])
            return np.array(normalized_box)    
    
    @staticmethod
    def valid_landmarks(face_landmark):
        left_eye = face_landmark['left_eye']
        right_eye = face_landmark['right_eye']
        mouth = face_landmark['top_lip'] + face_landmark['bottom_lip']

        left_eye_center = np.mean(left_eye, axis=0)
        right_eye_center = np.mean(right_eye, axis=0)
        mouth_center = np.mean(mouth, axis=0)

        eye_distance = np.linalg.norm(right_eye_center - left_eye_center)
        mouth_to_eye_distance = np.linalg.norm(mouth_center - left_eye_center)

        return mouth_to_eye_distance > eye_distance * 0.8

    @staticmethod
    def augment_box(box, frame_width, frame_height, offset=1):
        y_min, y_max, x_min, x_max = box
        box_height = y_max - y_min
        box_width = x_max - x_min

        # Check if result will be too large to fit into frame, return original box if so
        max_side = max(box_height, box_width)
        if max_side * offset > min(frame_width, frame_height):
            return box

        # Calculate middle point of rectangle, redefine corner points from there by multiplying offset
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        
        y_min = center_y - max_side * offset / 2
        y_max = center_y + max_side * offset / 2
        x_min = center_x - max_side * offset / 2
        x_max = center_x + max_side * offset / 2

        # Move box back into frame if scaled box lays outside some side
        # As check for too large box already was done in step 1, we can use elif
        if x_min < 0 or y_min < 0 or x_max > frame_width or y_max > frame_height:
            x_offset, y_offset = 0, 0
            if x_min < 0:
                x_offset = -x_min
            elif x_max > frame_width:
                x_offset = frame_width - x_max
            if y_min < 0:
                y_offset = -y_min
            elif y_max > frame_height:
                y_offset = frame_height - y_max
            y_min += y_offset
            y_max += y_offset
            x_min += x_offset
            x_max += x_offset

        return [int(y_min), int(y_max), int(x_min), int(x_max)]            

class EmotionDetector:
    def __init__(self, box_offset, frame_width, frame_height):
        self.emotion_detector = load_model("models/emotion_model.hdf5", compile=False)
        self.emotion_categories = self.get_emotion_categories()
        self.box_offset = box_offset
        self.frame_width = frame_width
        self.frame_height = frame_height
    
    def detect_emotion(self, frame, box):

        gray_cropped_face = self.crop_face(box, frame)

        prob = self.emotion_detector.predict(gray_cropped_face)[0]  # check for underscore
        return prob
    
    def crop_face(self, box, frame):
        box_augmented = FaceDetector.augment_box(box, self.frame_width, self.frame_height, self.box_offset)
        face = frame[box_augmented[0]:box_augmented[1], box_augmented[2]:box_augmented[3]]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (64, 64))
        face = face.astype('float32')/ 255
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)

        return face
    
    @staticmethod
    def get_emotion_categories():
        return ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
class PersonIdentifier:
    def __init__(self, box_offset, frame_width, frame_height, threshold = 0.6):
        self.threshold = threshold
        self.persons = []
        self.box_offset = box_offset
        self.frame_width = frame_width
        self.frame_height = frame_height

    def identify_person(self, frame, box):
        
        box_augmented = FaceDetector.augment_box(box, self.frame_width, self.frame_height, self.box_offset)
        curr_portrait = frame[box_augmented[0]:box_augmented[1], box_augmented[2]:box_augmented[3]]
        
        box_rearranged = np.array([box[0],box[3], box[1], box[2]])
        curr_face_encoding = face_recognition.face_encodings(frame, [box_rearranged])[0]
        min_distance = float("inf")
        matched_person = None

        if not self.persons:
            new_person_id = len(self.persons) + 1
            new_person = Person(new_person_id, curr_face_encoding, curr_portrait)
            self.persons.append(new_person)
            return new_person.id

        for person in self.persons:
            distance = face_recognition.face_distance([person.get_reference_face_encoding()], curr_face_encoding)[0]
            if distance < self.threshold:
                if distance < min_distance:
                    min_distance = distance
                    matched_person = person

        if matched_person is None:
            new_person_id = len(self.persons) + 1
            new_person = Person(new_person_id, curr_face_encoding, curr_portrait)
            self.persons.append(new_person)
            matched_person = new_person

        matched_person.add_appearance()

        return matched_person.id

class Person:
    def __init__(self, person_id, face_encoding, portrait, portrait_edge_length = 256):
        self.id = person_id
        self.appearances = 1
        self.reference_face_encoding = face_encoding
        self.representative_portrait = cv2.resize(cv2.cvtColor(portrait, cv2.COLOR_BGR2RGB), (portrait_edge_length, portrait_edge_length))

    def add_appearance(self):
        self.appearances += 1
    
    def get_reference_face_encoding(self):
        return self.reference_face_encoding
    
    