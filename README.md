# Hand Gesture Analysis for Haptic Awareness

## Overview
This project leverages computer vision techniques using Mediapipe and OpenCV to analyze videos of hands manipulating objects. The system detects and classifies various gestures such as tapping, holding, rotating, feeling, and probing. This research is being conducted at the University of Western Ontario and aims to advance the understanding of haptic abilities in medical sciences.

## Motivation
Haptic abilities are crucial in medical sciences, particularly in training and assessment of manual skills. By accurately detecting and analyzing hand gestures, we can gain deeper insights into haptic abilities and learn to answer questions like what are signs of high/low haptic abiltiies. This software will aid in that research by quickly parsing footage of the HAT (haptic abilities test) allowing researchers to gain more insights from the same trials.

## Features
- **Gesture Detection**: Identify and classify various hand gestures such as tapping, holding, rotating, feeling, and probing.
- **Video Analysis**: Process and analyze video frames to detect hand movements and interactions with objects.
- **Real-time Processing**: Utilize Mediapipe and OpenCV for real-time gesture recognition and feedback.
- **Data Logging**: Record and store gesture data for further analysis and research.

## Status
This project is currently a work in progress. Key features and functionalities are still being developed and tested.

## Usage
1. Ensure you have a video file of hand manipulations to analyze or a webcam feed.
2. Input configurations directly into the main script using any text/code editor.
3. Acquire dependencies using `pip install -r requirements.txt`
4. Run `main.py`.
5. The script will process the video/live-stream.

* Executables coming soon.

## Implementation Details
The project is built using Python, leveraging Mediapipe for hand detection and OpenCV for video processing. Key components include:
- **Hand Detection**: Mediapipe's hand tracking solution is used to detect and track hand landmarks in each video frame.
- **Gesture Classification**: A custom classifier is implemented to recognize specific gestures based on the hand landmark data.
- **Video Processing**: OpenCV is used to handle video input, frame extraction, and display of processed frames.
