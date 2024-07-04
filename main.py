from solution import start
from model import detect_video
import cv2

option = 'video'


def main():
    cv2.destroyAllWindows()

    if option == 'live_stream':
        print('Starting Live Stream')
        start()
    elif option == 'video':
        print('Analyzing Video...')
        start('video.mov')
    else:
        print('Option is invalid.')

    print('Closing Program')


print('Starting...')
main()
