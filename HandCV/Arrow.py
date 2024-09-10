import numpy as np
from numpy import ndarray


class Arrow:
    def __init__(self, start: ndarray, end: ndarray):
        self.start = start
        self.end = end

    def get_np_array(self):
        return self.end - self.start

    def get_norm(self):
        vector = self.get_np_array()
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / np.linalg.norm(vector)