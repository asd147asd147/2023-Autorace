from enum import Enum

class ECP_SIGN_STATE(Enum):
    CONSTRUCTION = 0
    FORBID = 1
    LEFT = 2
    PARKING = 3
    RIGHT = 4
    STOP = 5
    TUNNEL = 6
    INTERSECTION = 7
    NONE = 8