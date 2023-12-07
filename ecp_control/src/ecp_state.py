from enum import Enum

class ECP_STATE(Enum):
    NORMAL = 0
    INTERSECTION = 1
    INTERSECTION_LEFT = 2
    INTERSECTION_RIGHT = 3
    OBSTACLE = 4
    PARKING = 5
    LEVELCROSS = 6
    TUNNEL = 7