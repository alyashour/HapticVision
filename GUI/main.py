import logging
import warnings
from pathlib import Path
from typing import Tuple

import PySimpleGUI as sg

from GUI.controller import *
from HandCV.cv_controller import run, process_video as cv_process_video
from HandCV.cv_mode import CVMode, from_str as cvmode_from_str
from HandCV.dms import *
from HandCV.frame_processor import fp_process_data

# disable module level warning
warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    message='SymbolDatabase\.GetPrototype\(\) is deprecated\. Please use message_factory\.GetMessageClass\(\) instead\. SymbolDatabase\.GetPrototype\(\) will be removed soon\.'
)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

# Suppress all logging messages at the warning level and below
logging.getLogger().setLevel(logging.ERROR)

previous_input_data_path = 'previous_input.json'
DEFAULT_DIRECTORY = Path.home() / 'Desktop'

def read_previous_inputs_from_file() -> Tuple:
    try:
        filename, output_directory = load_json(previous_input_data_path)
        return True, filename, output_directory
    except FileNotFoundError:
        print('Previous inputs data - file not found.')
        return False, False, False


def run_main_menu():
    # load previous inputs if they exist
    success, previous_filename, previous_output_directory = read_previous_inputs_from_file()

    cv_tab_layout = [[sg.Text('Mode:'), sg.Combo([CVMode.VIDEO, CVMode.LIVE_STREAM], readonly=True, default_value='Video', key='-MODE-'),
                      sg.Push(), sg.Checkbox('Display live?', key='-DISPLAY_LIVE-')],
                     [sg.Text('Filename:'), sg.Push(),
                      sg.InputText(key='-FILENAME-', default_text=previous_filename if success else 'Please Select Video'),
                      sg.FileBrowse(file_types=(('MP4', '*.mp4'), ('MOV', '*.mov')), initial_folder=DEFAULT_DIRECTORY)],
                     [sg.Text('Output Directory:'),
                      sg.InputText(key='-OUTPUT_DIRECTORY-', default_text=previous_output_directory if success else 'Please Select Output Directory'),
                      sg.FolderBrowse(initial_folder=previous_output_directory if success else DEFAULT_DIRECTORY)],
                     [sg.Submit('Run'), sg.CloseButton('Close')]]

    # todo: add label at the top to ensure there is data available
    analysis_tab_layout = [[sg.Checkbox('Do Smoothing?', key='-DO_SMOOTHING-')],
                           [sg.Text('Smoothing Window Width: '), sg.InputText(key='-SMOOTHING_WINDOW_WIDTH-')],
                           [sg.Text('X Label'), sg.InputText(key='-X_LABEL-')],
                           [sg.Text('Y Label'), sg.InputText(key='-Y_LABEL-')],
                           [sg.Submit('Analyze'), sg.CloseButton('Cancel')]]

    layout = [[sg.TabGroup([[sg.Tab('CV', cv_tab_layout), sg.Tab('Analysis', analysis_tab_layout)]])]]

    # Create the Window
    window = sg.Window('CRIPT LAB - HAPTICS CV', layout)

    event, values = window.read()
    window.close()
    return event, values


def main():
    event, values = run_main_menu()

    try:
        if event == 'Run':
            # pull inputs from values
            mode = cvmode_from_str(values['-MODE-'].__str__())  # have to do this to make sure enum data type is correct
            display_live: bool = values['-DISPLAY_LIVE-']
            memory_limit_gb = 8  # todo: put this in the UI
            filename: str = values['-FILENAME-']
            output_directory: str = values['-OUTPUT_DIRECTORY-']
            save_json((filename, output_directory), previous_input_data_path)

            # execute command
            # todo: parallel
            if False:
                cv_results = cv_process_video(filename, memory_limit_gb)
                fp_results = fp_process_data(cv_results)
            # regular
            else:
                with FrameProcessor(output_directory) as fp:
                    run(fp, mode, filename, display_video=display_live)
        if event == 'Analyze':
            analyze_data(data=values)
    except InvalidInputError as e:
        sg.popup(f'Oops! Something happened.\nError: {str(e)}', title='Exception Occurred')
        # simply rerun the menu
        main()

    print('Closing application')


if __name__ == '__main__':
    main()
