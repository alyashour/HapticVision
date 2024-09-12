import time

import cv2
import mediapipe as mp
import PySimpleGUI as sg

from GUI.main import CVMode
from GUI.time_formatter import format_duration
from HandCV.model_result import ModelResult
from HandCV.frame_processor import FrameProcessor

# todo: function too long, refactor
def run(processor: FrameProcessor,
        mode: CVMode,
        video_path: str = None,
        camera_index: int = 0,
        display_video: bool = False
        ):
    # todo: write doc below
    """

    :param processor:
    :param mode:
    :param video_path:
    :param camera_index:
    :param display_video:
    :return:
    """
    # make sure there's a video path when in video mode
    if mode == CVMode.VIDEO and video_path is None:
        raise Exception("No video path")

    # if the mode is set to VIDEO, set the capture to the video stream, else set it to the camera index
    if mode == CVMode.VIDEO:
        print('Entering Video Mode')
    else:
        print('Entering Camera (live) Mode')

    cap = cv2.VideoCapture(video_path if mode == CVMode.VIDEO else camera_index)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # todo: refactor this into the GUI directory
    layout = [
        [sg.Text('Processing Video...')],
        [sg.ProgressBar(total_frames, orientation='h', size=(20, 20), key='-PROGRESS-')],
        [sg.Text('Elapsed Time: '), sg.Text('', key='-ELAPSED-')],
        [sg.Text('Estimated Time Remaining: '), sg.Text("", key='-ESTIMATED-')],
        [sg.Cancel('Cancel'), sg.Push(), sg.Text('', key='-PROGRESS_LABEL-')]
    ]
    window = sg.Window('Progress', layout, finalize=True)

    with (mp.solutions.hands.Hands(
            model_complexity=1,
            min_detection_confidence=0.2,
            min_tracking_confidence=0.5) as hands):

        # analytics for the UI
        start_time = time.time()
        frame_count = 0

        # start reading the video
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                # If loading a video, use 'break' instead of 'continue'
                # as an unsuccessful read means the end of a video but could simply be lag in a stream
                if video_path:
                    print("End of Video")
                    break
                else:
                    print("Ignoring empty camera frame.")
                    continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            foreign_result = hands.process(image)

            # prep the image for writing
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # draw the hand annotations on the image.
            process_is_success = bool(foreign_result.multi_hand_landmarks)
            if process_is_success:
                # convert from the foreign result to the ModelResult class
                result = ModelResult.get_from_raw_output(foreign_result)
                processor.process_frame(image, result)

            # optionally show the video as its being analyzed to the user
            if display_video:
                # Flip the image horizontally for a selfie-view display.
                cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
                cv2.imshow('Main', image)

            # update UI vars
            frame_count += 1
            elapsed_time = time.time() - start_time
            progress = frame_count / total_frames * 100

            # update progress UI labels
            window['-PROGRESS-'].update(progress)
            window['-PROGRESS_LABEL-'].update(f'{frame_count}/{total_frames}')

            # calculate remaining time
            if frame_count > 0:
                avg_time_per_frame = elapsed_time / frame_count
                remaining_frames = total_frames - frame_count
                estimated_remaining_time = avg_time_per_frame * remaining_frames
            else:
                estimated_remaining_time = 0

            # update time display
            window['-ELAPSED-'].update(format_duration(elapsed_time))
            window['-ESTIMATED-'].update(format_duration(estimated_remaining_time))
            # break on 'q'
            if cv2.waitKey(1) == ord('q'):
                break

            # check if the user pressed cancel
            event, _ = window.read(timeout=0)
            if event == 'Cancel' or event == sg.WIN_CLOSED:
                break

    cap.release()
    window.close()