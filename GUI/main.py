from typing import Tuple

import PySimpleGUI as sg
from Controller import *
import warnings
from DMS.dms import *

# disable module level warning
warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    message='SymbolDatabase\.GetPrototype\(\) is deprecated\. Please use message_factory\.GetMessageClass\(\) instead\. SymbolDatabase\.GetPrototype\(\) will be removed soon\.'
)

previous_input_data_path = 'previous_input.json'

# Get the parent directory
SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
PARENT_DIRECTORY = os.path.dirname(SCRIPT_DIRECTORY)
# Define the default paths for the asset and output directories
DEFAULT_ASSETS_DIR = os.path.join(PARENT_DIRECTORY, 'Assets', 'Videos')
DEFAULT_OUTPUT_DIR = os.path.join(PARENT_DIRECTORY, 'Output')

def read_previous_inputs_from_file() -> Tuple:
    try:
        filename, output_directory = load_data(previous_input_data_path)
        return True, filename, output_directory
    except FileNotFoundError:
        print('Previous inputs data - file not found.')
        return False, False, False

def run_main_menu():
    # load previous inputs if they exist
    success, previous_filename, previous_output_directory = read_previous_inputs_from_file()

    cv_tab_layout = [[sg.Text('Mode:'), sg.Combo([CVMode.VIDEO, CVMode.LIVE_STREAM], readonly=True, default_value='Video', key='-MODE-'),
                      sg.Push(), sg.Checkbox('Display live?', key='-DISPLAY_LIVE-')],
                     [sg.Text('Filename:'), sg.Push(), sg.InputText(key='-FILENAME-', default_text= previous_filename if success else 'Please Select Video'),
                      sg.FileBrowse(file_types=(('MP4', '*.mp4'), ('MOV', '*.mov')), initial_folder=DEFAULT_ASSETS_DIR)],
                     [sg.Text('Output Directory:'), sg.InputText(key='-OUTPUT_DIRECTORY-', default_text= previous_output_directory if success else DEFAULT_OUTPUT_DIR),
                      sg.FolderBrowse(initial_folder=DEFAULT_OUTPUT_DIR)],
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
            mode = cvmode_from_str(values['-MODE-'].__str__()) # have to do this to make sure enum data type is correct
            display_live: bool = values['-DISPLAY_LIVE-']
            filename: str = values['-FILENAME-']
            output_directory: str = values['-OUTPUT_DIRECTORY-']
            save_data((filename, output_directory), previous_input_data_path)

            # execute command
            success = run_cv(mode, display_live, filename, output_directory)
        if event == 'Analyze':
            analyze_data(data=values)
    except InvalidInputError as e:
        sg.popup(f'Oops! Something happened.\nError: {str(e)}', title='Exception Occurred')
        # simply rerun the menu
        main()

    print('Closing application')


if __name__ == '__main__':
    main()
