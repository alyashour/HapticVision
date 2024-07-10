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


# DO NOT TOUCH ANYTHING BELOW #
if option == 'live_stream':
    video_path = None


def main():
    def annotations(frame, results, previous_results):
        if results:
            draw_translucent_3d_plane(frame, results)
        if previous_results:
            draw_velocity_arrows(frame, results, previous_results)

    start_model(video_path, annotations)
    print('Closing Program')


print('Starting...')
main()
