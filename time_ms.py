import sys
from time import time_ns


def time_ms() -> int:
    return round(time_ns() * 1e-6)


sys.modules[__name__] = time_ms
