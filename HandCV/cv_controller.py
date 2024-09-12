import concurrent.futures
import time

import PySimpleGUI as sg
import cv2
import mediapipe as mp

from FrameProcessor.frame_processor import FrameProcessor
from HandCV.cv_mode import CVMode
from GUI.time_formatter import format_duration, clock
from HandCV.model_result import ModelResult


# this is the main CV loop

##### PARALLEL
# todo: add UI here
def load_video_until_size(video_path, memory_limit_gb, start_frame = 0):
    """
    Loads a video from storage into memory up to a certain size. Returns the final frame number and the reference to the place in memory.
    :param video_path:
    :param memory_limit_gb:
    :param start_frame:
    :return: end_frame: int, frames: []
    """
    end_of_video_reached = False

    # convert memory limit to bytes
    memory_limit_bytes = memory_limit_gb * (1024 ** 3)

    # open the video file
    cap = cv2.VideoCapture(video_path)

    # set the frame to start reading from
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # confirm
    print(f'Starting to load video into memory from frame: {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}')

    if not cap.isOpened():
        print('Error: Could not open video file')

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    channels = 3 # for RGB

    # calc the size of 1 frame in memory
    bytes_per_frame = frame_width * frame_height * channels
    max_frames = memory_limit_bytes / bytes_per_frame

    # accumulate frames
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('End of video reached (or error)')
            end_of_video_reached = True
            break

        frames.append(frame) # load into memory
        frame_count += 1

        if frame_count >= max_frames:
            break

    cap.release()
    final_frame = frame_count + start_frame

    return end_of_video_reached, final_frame, frames

def process_frame(frame) -> (bool, ModelResult):
    with (mp.solutions.hands.Hands(
            model_complexity=1,
            min_detection_confidence=0.2,
            min_tracking_confidence=0.5,
            static_image_mode=True) as hands):

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        foreign_result = hands.process(frame)

        # draw the hand annotations on the image.
        process_is_success = bool(foreign_result.multi_hand_landmarks)
        if process_is_success:
            # convert from the foreign result to the ModelResult class
            result = ModelResult.get_from_raw_output(foreign_result)
            return True, result

        return False, None

def process_frames_in_parallel(frames) -> list:
    results = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # submit all frames for processing
        future_to_frame = {executor.submit(process_frame, frame): i for i, frame in enumerate(frames)}

        for future in concurrent.futures.as_completed(future_to_frame):
            frame_number = future_to_frame[future]
            try:
                success, result = future.result()

                if success:
                    results[frame_number] = result
            except Exception as exc:
                print(f'Exception occurred while parallel processing frame: {exc}')

    return [results[key] for key in sorted(results.keys())]

def process_video(video_path, memory_limit_gb):
    """
    Takes in a video path and a chunking memory limit in gigabytes. Returns a list indexed by the frames giving the results of the CV analysis.
    :param video_path:
    :param memory_limit_gb:
    :return:
    """
    print(f'Entering CV loop - {clock()}')

    end_of_video_reached = False
    final_frame = 0
    results = []

    # until we've reached the end of the video
    while not end_of_video_reached:
        # load part of the video into memory
        end_of_video_reached, final_frame, frames = load_video_until_size(video_path, memory_limit_gb, start_frame=final_frame)

        # process that part of the video and save that to memory
        results.append(process_frames_in_parallel(frames))

    print(f'Exiting CV loop - {clock()}')

    return results
######

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

    cap = cv2.VideoCapture(video_path if mode == CVMode.VIDEO else camera_index)

    # if the mode is set to VIDEO, set the capture to the video stream, else set it to the camera index
    if mode == CVMode.VIDEO:
        print('Entering Video Mode')
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
    else:
        print('Entering Camera (live) Mode')

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
                processor.process_frame(frame_count, image, result)

            # optionally show the video as its being analyzed to the user
            if display_video:
                # Flip the image horizontally for a selfie-view display.
                cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
                cv2.imshow('Main', image)

            if mode == CVMode.VIDEO:
                # update UI vars
                frame_count += 1
                elapsed_time = time.time() - start_time

                # update progress UI labels
                window['-PROGRESS-'].update(frame_count)
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