import os

from FrameProcessor.frame_processor import FrameProcessor
from HandCV.cv_controller import run as start_cv_window
from HandCV.cv_mode import CVMode
from HandCV.exceptions import InvalidInputError


def ensure_inputs(mode: CVMode, filename, output_filename, output_directory) -> bool:
    if not isinstance(mode, CVMode):
        return False
    # todo: ensure mode is of type CVMode
    # todo: ensure filename exists
    if not os.path.exists(filename):
        return False
    # todo: ensure output_directory is valid and doesn't contain a file with the output file name
    return True

def run_cv(mode, display_live, filename, output_directory):
    print('Starting CV analysis')

    # if inputs are valid
    if ensure_inputs(mode, filename, output_directory, output_directory):
        print('Inputs validated, starting processor.')
        with FrameProcessor(
            output_path=output_directory
        ) as processor:
            start_cv_window(processor, mode, filename, display_video=display_live)
    else:
        raise InvalidInputError('Invalid inputs')

def analyze_data(data):
    print('Analyzing data')