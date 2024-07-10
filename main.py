from mediapipe.tasks.python.vision import HandLandmarkerResult
from numpy import ndarray

from frameProcessor import FrameProcessor
from new_api import run as start_model
from drawing_utils import *

#################################
# Type? (video or live_stream):
option = 'live_stream'

# video file path?
video_path = 'video.mov'

# enable velocities?
calculate_velocities = True
#################################

if option == 'live_stream':
    video_path = None


def main():
    processor = FrameProcessor()
    start_model(video_path, processor)
    print('Closing Program')


print('Starting...')
main()
