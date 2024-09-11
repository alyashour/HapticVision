from dataclasses import dataclass


@dataclass
class Landmark:
    x: float
    y: float
    z: float


@dataclass
class NormalizedLandmark:
    x: float
    y: float
    z: float


class HandednessResult:
    def __init__(self, classification):
        label = classification.label
        score = classification.score

        assert label == 'Left' or label == 'Right'

        self.label: str = label
        self.score: float = score


# todo: consider going from matching lists to 1 list of Hand dataclasses
@dataclass
class ModelResult:
    """
        Contains all the results that are given by the model

        Attributes
        ----------
        multi_hand_landmarks: [[NormalizedLandmark]]
            image coords with (0, 0) in the top right.
        multi_hand_world_landmarks: [[NormalizedLandmark]]
            Real-world 3D coordinates in meters with the origin at the hand's approximate geometric center.
        multi_handedness: [HandednessResult]
            List of handedness results {label, confidence}
        """
    multi_hand_landmarks: [[NormalizedLandmark]]
    multi_hand_world_landmarks: [[NormalizedLandmark]]
    multi_handedness: [HandednessResult]

    @staticmethod
    def get_from_raw_output(result):
        # all of these are aligned such that [hand1, hand2, ...]
        return ModelResult(
            multi_hand_landmarks=[_get_hand_landmarks(hand_landmarks) for hand_landmarks in result.multi_hand_landmarks],
            multi_hand_world_landmarks=[_get_normalized_hand_landmarks(normalized_hand_landmarks) for normalized_hand_landmarks in result.multi_hand_world_landmarks],
            multi_handedness=[HandednessResult(hand_classifications.classification[0]) for hand_classifications in result.multi_handedness]
        )


def _get_hand_landmarks(hand_landmarks) -> [Landmark]:
    new_landmark_list = []
    for landmark in hand_landmarks.landmark:
        new_landmark_list.append(Landmark(landmark.x, landmark.y, landmark.z))
    return new_landmark_list


def _get_normalized_hand_landmarks(normalized_hand_landmarks) -> [NormalizedLandmark]:
    new_landmark_list = []
    for normalized_landmark in normalized_hand_landmarks.landmark:
        new_landmark_list.append(NormalizedLandmark(normalized_landmark.x, normalized_landmark.y, normalized_landmark.z))
    return new_landmark_list
