from frame_processor import FrameProcessor
from cv_controller import run as start_cv_window

#################################
# Type? (video or live_stream):
option = 'live_stream'

# video file path?
video_path = 'Videos/F1 - H-006.mp4'

# enable velocities?
calculate_velocities = True
#################################

if option == 'live_stream':
    video_path = None


def main():
    with FrameProcessor() as processor:
        start_cv_window(processor, video_path)
        print('Closing Program')


print('Starting...')
main()
