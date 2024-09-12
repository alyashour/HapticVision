from enum import Enum

class CVMode(Enum):
    VIDEO = 0
    LIVE_STREAM = 1

    def __str__(self):
        if self == CVMode.VIDEO:
            return 'Video'
        if self == CVMode.LIVE_STREAM:
            return 'Live Stream'

def from_str(s: str) -> CVMode:
    if s == 'Video':
        return CVMode.VIDEO
    else:
        return CVMode.LIVE_STREAM
