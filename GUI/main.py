import PySimpleGUI as sg

cv_tab_layout = [ [sg.Text('Mode: '), sg.Combo(['Video', 'Live Stream'], readonly=True, default_value='Video')],
                  [sg.Text('Filename: '), sg.InputText(key='filename'), sg.FileBrowse('Select', file_types=(('MP4', '*.mp4'), ('MOV', '*.mov')))],
                  [sg.Text('Output Directory'), sg.InputText(key='directory_name'), sg.FolderBrowse('select directory')],
                  [sg.Button('Run')]]

analysis_tab_layout = [ [sg.Checkbox('Do Smoothing?')],
                        [sg.Text('Smoothing Window Width: '), sg.InputText(key='smoothing_window')],
                        [sg.Text('X Label'), sg.InputText(key='x_label')],
                        [sg.Text('Y Label'), sg.InputText(key='y_label')],
                        [sg.Submit()]]

layout = [[sg.TabGroup([[sg.Tab("CV", cv_tab_layout), sg.Tab("Analysis", analysis_tab_layout)]])]]

# Create the Window
window = sg.Window('CRIPT LAB - HAPTICS CV', layout)

# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel':	# if user closes window or clicks cancel
        break
    print('You entered ', values[0])

window.close()
