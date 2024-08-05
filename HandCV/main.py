from frameProcessor import FrameProcessor
from window_handler import run as start_model

#################################
# Type? (video or live_stream):
option = 'video'

# video file path?
video_path = 'Videos/F1 - H-006.mp4'
#################################

if option == 'live_stream':
    video_path = None


def main():
    with FrameProcessor() as processor:
        start_model(processor, video_path)
        print('Closing Program')


print('Starting...')
main()
