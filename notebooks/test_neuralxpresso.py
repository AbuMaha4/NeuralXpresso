import NeuralXpresso as nx
import unittest

class TestStrings(unittest.TestCase):
    def test_upper(self):
        self.assertEqual("spam".upper(), "SPAM")

    def test_loading_video_from_youtube(self):
        my_video_processor = nx.VideoProcessor()
        path = 'https://www.youtube.com/watch?v=vtT78TfDfXU'
        my_video_processor.load_video_from_youtube(path)
        self.assertEqual(my_video_processor.total_frames, 1000)
