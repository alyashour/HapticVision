import PySimpleGUI as sg

# NOT COMPLETE
# todo: finish
def ProgressView(max):
    layout = [
                [sg.Text('Processing Video...')],
                [sg.ProgressBar(max, orientation='h', size=(20, 20), key='-PROGRESS-')],
                [sg.Text('Elapsed Time: '), sg.Text('', key='-ELAPSED-')],
                [sg.Text('Estimated Time Remaining: '), sg.Text("", key='-ESTIMATED-')],
                [sg.Cancel('Cancel'), sg.Push(), sg.Text('', key='-PROGRESS_LABEL-')] # progress label looks like 45/500
            ]
    window = sg.Window('Progress', layout, finalize=True)