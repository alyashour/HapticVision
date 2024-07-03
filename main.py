from live_stream import start_live_stream
from video import detect_video
import cv2

option = 'video'


def main():
    cv2.destroyAllWindows()

    if option == 'live_stream':
        print('Starting Live Stream')
        start_live_stream()
    elif option == 'video':
        print('Analyzing Video...')
        detect_video()
    else:
        print('Option is invalid.')

    print('Closing Program')


print('Starting...')
main()
