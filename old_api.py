import cv2
import mediapipe as mp
from time import time_ns


def start(video_path: str = None):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    capture = cv2.VideoCapture(video_path if video_path else 0)
    hands = mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        static_image_mode=False
    )

    while capture.isOpened():
        success, frame_bgr = capture.read()
        frame_start_time = time_ns()
        if not success:
            if video_path:
                break
            else:
                continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # process image
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # draw landmarks
                mp_drawing.draw_landmarks(
                    frame_bgr,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        # calculate framerate
        processing_time = (time_ns() - frame_start_time) * 1e-9
        fps = int(1 / processing_time)

        # print the framerate to the screen
        cv2.putText(frame_bgr, str(fps), (200, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('MediaPipe Hands', frame_bgr)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    print('Releasing Capture')
    capture.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == '__main__':
    start()
