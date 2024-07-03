import cv2
import mediapipe as mp


def start_live_stream():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    capture = cv2.VideoCapture(0)
    hands = mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=False
    )

    while capture.isOpened():
        success, frame_bgr = capture.read()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if not success:
            continue

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

        cv2.imshow('MediaPipe Hands', frame_bgr)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    print('Releasing Capture')
    capture.release()
    cv2.destroyAllWindows()
    cv2.waitKey()


if __name__ == '__main__':
    start_live_stream()
