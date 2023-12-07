#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

# Authors: Leon Jung, Gilbert, Ashe Kim, Special Thanks : Roger Sacchelli

import rospy
from std_msgs.msg import UInt8, Float64

from ecp_state import ECP_STATE
from ecp_sign_state import ECP_SIGN_STATE

class Controller():
    def __init__(self):
        self.sub_sign = rospy.Subscriber('/detect/traffic_sign', UInt8, self.changeSignState, queue_size = 1)

        self.pub_moving_state = rospy.Publisher('/control/moving/state', UInt8, queue_size = 1)

        self.sign_state = ECP_SIGN_STATE.NONE.value
        self.ecp_state = ECP_STATE.NORMAL.value
        self.is_intersection = False
        self.is_parking = False

    def changeSignState(self, msg):
        self.sign_state = ECP_SIGN_STATE(msg.data).name
        rospy.loginfo("Detect %s Sign", self.sign_state)

        msg_moving_state = UInt8()

        if self.sign_state == ECP_SIGN_STATE.NONE.value:
            self.ecp_state = ECP_STATE.NORMAL.value

        elif self.sign_state == ECP_SIGN_STATE.CONSTRUCTION.value:
            self.ecp_state = ECP_STATE.OBSTACLE
            self.is_intersection = False
        
        elif self.sign_state == ECP_SIGN_STATE.FORBID.value:
            self.ecp_state = ECP_STATE.NORMAL.value

        elif self.sign_state == ECP_SIGN_STATE.INTERSECTION.value:
            self.ecp_state = ECP_STATE.INTERSECTION
            self.is_intersection = True

        elif self.sign_state == ECP_SIGN_STATE.LEFT.value:
            if self.is_intersection:
                self.ecp_state = ECP_STATE.INTERSECTION_LEFT.value
            elif self.is_parking:
                pass

        elif self.sign_state == ECP_SIGN_STATE.RIGHT.value:
            self.ecp_state = ECP_STATE.INTERSECTION_RIGHT.value

        elif self.sign_state == ECP_SIGN_STATE.PARKING.value:
            self.ecp_state = ECP_STATE.PARKING.value
            self.is_parking = True

        elif self.sign_state == ECP_SIGN_STATE.STOP.value:
            self.ecp_state = ECP_STATE.LEVELCROSS.value
            self.is_parking = False
            

        elif self.sign_state == ECP_SIGN_STATE.TUNNEL.value:
            self.ecp_state = ECP_STATE.TUNNEL.value

        msg_moving_state.data = self.ecp_state
        self.pub_moving_state.publish(msg_moving_state)
            



    def main(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('contorl')
    node = Controller()
    node.main()