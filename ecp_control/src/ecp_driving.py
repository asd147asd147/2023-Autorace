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
import time
import numpy as np

from std_msgs.msg import UInt8, Float64, Bool
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

from enum import Enum
import math
import tf
from tf import transformations
from ecp_state import ECP_STATE
from ecp_sign_state import ECP_SIGN_STATE

from ecp_preproc.msg import DetectLaneInfo

class ControlDriving():
    def __init__(self):
        #moving internal valiables
        self.theta = 0.0
        self.current_theta = 0.0
        self.last_current_theta = 0.0
        self.lastError = 0.0

        self.sub_moving_state = rospy.Subscriber('/control/moving/state', UInt8, self.get_moving_state, queue_size = 1)
        self.sub_odom = rospy.Subscriber('/odom', Odometry, self.cbOdom, queue_size=1)

        self.sub_lane = rospy.Subscriber('/detect/lane', DetectLaneInfo, self.get_lane_info, queue_size = 1)
        self.sub_scan_dis = rospy.Subscriber('/scan', LaserScan, self.get_scan_dis, queue_size=1)
        self.sub_parking_state = rospy.Subscriber('/parking/state', UInt8, self.cbGetParkingState, queue_size = 1)
        self.sub_detect_levelcrossbar = rospy.Subscriber('/detect/levelcrossbar', Bool, self.cbGetLevelCrossBar, queue_size=1)
        self.sub_detect_traffic_signal = rospy.Subscriber('/detect/traffic_signal', UInt8, self.cbGetTrafficSign, queue_size=1)
        
        self.pub_check_parking_ready = rospy.Publisher('/check/parking_ready', UInt8, queue_size = 1)
        self.pub_check_level_cross = rospy.Publisher('/check/levelcross', UInt8, queue_size=1)
        self.pub_detect_lane_side = rospy.Publisher('/detect/lane_side', UInt8, queue_size = 1)
        self.pub_check_traffic = rospy.Publisher('/check/traffic', UInt8, queue_size=1)

        self.pub_check_tunnel = rospy.Publisher('/check/tunnel', UInt8, queue_size=1)

        self.pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size = 1)

        #moving type enum
        self.TypeOfMoving = Enum('TypeOfMoving', 'idle left right forward backward')
        self.TypeOfState = Enum('TypeOfState', 'idle start stop finish')
        
        self.current_pos_x = 0.0
        self.current_pos_y = 0.0
        
        self.desired_center = 160
        self.lane_pos = 0
        
        self.obstacle_trigger = False

        self.parking_state = 0 #0: None, 1:detect parking area, 2: detect end line

        self.scan_dis = [0.0] * 360
        self.lastError = 0
        self.parking_done = False
        self.parking = False

        self.level_cross_state = False
        self.level_cross_stop = False

        self.start_ready = True
        self.traffic = 0 #0: stop, 1: GO

        #DEBUG
        # self.parking_state = 2
        #DEBUG

        #moving params
        self.moving_type = ECP_STATE.NORMAL.value
        rospy.on_shutdown(self.fnShutDown)
        time.sleep(5)
        loop_rate = rospy.Rate(100) # 10hz
        while not rospy.is_shutdown():
            # rospy.loginfo('moving type %d', self.moving_type)
            if self.moving_type == ECP_STATE.NORMAL.value:
                if self.start_ready:
                    msg_check_traffic = UInt8()
                    msg_check_traffic = 1
                    self.pub_check_traffic.publish(msg_check_traffic)
                    if self.traffic == 0:
                        self.set_cmd_vel([0,0,0], [0,0,0])
                    else:
                        self.fnNormalDriving()
                        self.start_ready = False
                else:
                    self.fnNormalDriving()
            elif self.moving_type == ECP_STATE.INTERSECTION.value:
                self.fnNormalDriving()
            elif self.moving_type == ECP_STATE.INTERSECTION_LEFT.value:
                msg_lane_side = UInt8()
                msg_lane_side.data = 1
                self.pub_detect_lane_side.publish(msg_lane_side)
                if self.lane_pos != 1:
                    self.set_cmd_vel([0.08,0,0], [0,0,0.2])
                else:
                    self.fnNormalDriving()
            elif self.moving_type == ECP_STATE.INTERSECTION_RIGHT.value:
                msg_lane_side = UInt8()
                msg_lane_side.data = 2
                self.pub_detect_lane_side.publish(msg_lane_side)
                if self.lane_pos != 2:
                    self.set_cmd_vel([0.08,0,0], [0,0,-0.2])
                else:
                    self.fnNormalDriving()
            elif self.moving_type == ECP_STATE.OBSTACLE.value:
                self.fnConstructionMission()
            elif self.moving_type == ECP_STATE.PARKING.value:
                msg_lane_side = UInt8()
                msg_parking_ready = UInt8()
                if self.parking_state == 0:
                    msg_lane_side.data = 1
                    self.pub_detect_lane_side.publish(msg_lane_side)
                    msg_parking_ready.data = 1
                    self.pub_check_parking_ready.publish(msg_parking_ready)
                    if self.lane_pos != 1:
                        self.set_cmd_vel([0.08,0,0], [0,0,0.7])
                    else:
                        self.fnNormalDriving()
                elif self.parking_state == 1:
                    msg_lane_side.data = 0
                    self.pub_detect_lane_side.publish(msg_lane_side)

                    msg_parking_ready.data = 2
                    self.pub_check_parking_ready.publish(msg_parking_ready)
                    
                    self.fnNormalDriving()
                elif self.parking_state == 2:
                    if not self.parking:
                        self.set_cmd_vel([0,0,0], [0,0,0])
                        self.parking = True
                    else:
                        if not self.parking_done:
                            

                            lidr_resolution = len(self.scan_dis)
                            degree45 = lidr_resolution//8

                            left_front = degree45
                            left_back = degree45*3

                            right_front = degree45*7
                            right_back = degree45*5

                            left_scan_count = self.fn_cal_scan_count(self.scan_dis[left_front:left_back], 0.3)
                            right_scan_count = self.fn_cal_scan_count(self.scan_dis[right_back:right_front], 0.3)
                            print(self.scan_dis[right_back:right_front])
                            rospy.logwarn("left: %d, right: %d", left_scan_count, right_scan_count)

                            if left_scan_count < right_scan_count:
                                # self.fnRotateDriving(90, True)
                                self.fnRotateDriving(0.4, True)
                                self.fnStraightDriving(0.8,True)
                                self.fnStopDriving(0.1)
                                # rospy.logwarn("Parking Done")
                                self.fnStraightDriving(0.8,False)
                                self.fnRotateDriving(0.8, True)
                                # self.fnRotateDriving(90, True)

                                self.parking_done = True
                            else:
                                # self.fnRotateDriving(90, False)

                                self.fnRotateDriving(0.4, False)
                                self.fnStraightDriving(0.8,True)
                                self.fnStopDriving(0.1)
                                # rospy.logwarn("Parking Done")
                                self.fnStraightDriving(0.8,False)
                                self.fnRotateDriving(0.8, False)
                                # self.fnRotateDriving(90, False)
                                self.parking_done = True
                        else:
                            msg_lane_side.data = 1
                            self.pub_detect_lane_side.publish(msg_lane_side)

                            if self.lane_pos != 1:
                                self.set_cmd_vel([0.08,0,0], [0,0,0.2])
                            else:
                                self.fnNormalDriving()
                            
                            msg_parking_ready.data = 0
                            self.pub_check_parking_ready.publish(msg_parking_ready)
            elif self.moving_type == ECP_STATE.LEVELCROSS.value:
                msg_check_level_cross = UInt8()
                msg_check_level_cross.data = 1
                self.pub_check_level_cross.publish(msg_check_level_cross)
                # print(self.level_cross_state)
                if not self.level_cross_state:
                    self.fnNormalDriving()
                else:
                    self.set_cmd_vel([0.0,0,0], [0,0,0])
                    self.level_cross_stop = True
                    # rospy.logwarn("Detect Level Cross")
                
            elif self.moving_type == ECP_STATE.TUNNEL.value:
                #TODO
                msg_check_tunnel = UInt8()
                msg_check_tunnel.data = 1
                self.pub_check_tunnel.publish(msg_check_tunnel)
            loop_rate.sleep()


    def cbGetTrafficSign(self, msg):
        self.traffic = msg.data

    def cbGetLevelCrossBar(self, msg):
        self.level_cross_state = msg.data

    def cbGetParkingState(self, msg):
        self.parking_state = msg.data

    def fn_cal_angle(self, theta):
        if theta > 2*math.pi:
            theta -= 2*math.pi
        elif theta > math.pi:
            theta = 2*math.pi - theta
        return theta

    def fn_cal_odom_angle(self, theta1, theta2):
        theta = abs(theta2 - theta1)
        return self.fn_cal_angle(theta)
    
    def fn_cal_odom_dis(self, pos1, pos2):
        return math.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2)

    def fnStopDriving(self, t):
        self.check_time_pre = time.time()
        while time.time() - self.check_time_pre < t:
            self.set_cmd_vel([0,0,0], [0,0,0])
        
    def fnStraightDriving(self, t, forward):
        self.check_time_pre = time.time()
        if forward:
            while time.time() - self.check_time_pre < t:
                # rospy.loginfo(time.time() - self.check_time_pre)
                self.set_cmd_vel([2.2,0,0], [0,0,0])
            self.set_cmd_vel([0,0,0], [0,0,0])
        else:
            while time.time() - self.check_time_pre < t:
                # rospy.loginfo(time.time() - self.check_time_pre)
                self.set_cmd_vel([-2.2,0,0], [0,0,0])
            self.set_cmd_vel([0,0,0], [0,0,0])

    def fnRotateDriving(self, t, ccw):
        self.check_time_pre = time.time()

        if ccw:
            while time.time() - self.check_time_pre < t:
                self.set_cmd_vel([0,0,0], [0,0,1.8])
            self.set_cmd_vel([0,0,0], [0,0,0])
        else:
            while time.time() - self.check_time_pre < t:
                self.set_cmd_vel([0,0,0], [0,0,-1.8])
            self.set_cmd_vel([0,0,0], [0,0,0])

    # def fnRotateDriving(self, degree, ccw):
    #     self.odom_theta_start = self.current_theta
    #     if ccw:
    #         while self.fn_cal_odom_angle(self.fn_cal_angle(self.odom_theta_start+degree*math.pi/180), self.current_theta)*180/math.pi > 2:
    #             rospy.loginfo(self.fn_cal_odom_angle(self.odom_theta_start+degree*math.pi/180, self.current_theta)*180/math.pi)
    #             self.set_cmd_vel([0,0,0], [0,0,0.6])
    #         self.set_cmd_vel([0,0,0], [0,0,0])
    #     else:
    #         while self.fn_cal_odom_angle(self.fn_cal_angle(self.odom_theta_start-degree*math.pi/180), self.current_theta)*180/math.pi > 2:
    #             rospy.loginfo(self.fn_cal_odom_angle(self.fn_cal_angle(self.odom_theta_start-degree*math.pi/180), self.current_theta)*180/math.pi)
    #             self.set_cmd_vel([0,0,0], [0,0,-0.6])
    #         self.set_cmd_vel([0,0,0], [0,0,0])

    def fnShutDown(self):
        rospy.loginfo("Shutting down. cmd_vel will be 0")

        twist = Twist()
        twist.linear.x = 0
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = 0
        self.pub_cmd_vel.publish(twist) 

    def fnNormalDriving(self):
        center = self.desired_center
        error = center - 160

        # Kp = 0.035
        # Kd = 0.007

        Kp = 0.011
        Kd = 0.01

        angular_z = Kp * error + Kd * (error - self.lastError)
        self.lastError = error

        angular_z = -max(angular_z, -3.0) if angular_z < 0 else -min(angular_z, 3.0)

        self.set_cmd_vel([0.8,0,0], [0,0,angular_z])


    def fnConstructionMission(self):
        forward_obstacle_count = self.fn_cal_scan_count(self.scan_dis[330:359]+self.scan_dis[0: 30], 0.50)
        if 12 < forward_obstacle_count < 28:
            rospy.loginfo('Forward Obstacle Detecting')
            if self.lane_pos == 1: #left lane
                while self.lane_pos == 1:
                    self.set_cmd_vel([0.1,0,0], [0,0,-0.9])
            elif self.lane_pos == 2: #right lane
                while self.lane_pos == 2:
                    self.set_cmd_vel([0.1,0,0], [0,0,-0.9])
        else:
            self.fnNormalDriving()

    # def fnConstructionMission(self):
    #     forward_obstacle_count = self.fn_cal_scan_count(self.scan_dis[-15 :]+self.scan_dis[0: 15], 0.30)
    #     rospy.loginfo('Forward Obstacle Detecting : %d', forward_obstacle_count)
    #     if 12 < forward_obstacle_count < 30 :
    #         # elif self.lane_pos == 2: #right lane
    #             # while self.lane_pos == 2:
    #         if self.obstacle_trigger == False:
    #             self.obstacle_trigger=True
    #             prev_time = time.time()
    #             while time.time() - prev_time < 0.9:
    #                 self.set_cmd_vel([0.2,0,0], [0,0,1.9])
    #             # if self.lane_pos == 1: #left lane
    #                 # while self.lane_pos == 1:
    #             prev_time = time.time()
    #             while time.time() - prev_time < 1.4:
    #                 self.set_cmd_vel([0.2,0,0], [0,0,-1.9])
    #         else:
    #             self.obstacle_trigger=False
    #             prev_time = time.time()
    #             while time.time() - prev_time < 0.9:
    #                 self.set_cmd_vel([0.2,0,0], [0,0,-1.9])
    #             # if self.lane_pos == 1: #left lane
    #                 # while self.lane_pos == 1:
    #             prev_time = time.time()
    #             while time.time() - prev_time < 1.4:
    #                 self.set_cmd_vel([0.2,0,0], [0,0,1.9])
    #     else:
    #         self.fnNormalDriving()


    def set_cmd_vel(self, linear, angular):
        twist = Twist()
        twist.linear.x = linear[0]
        twist.linear.y = linear[1]
        twist.linear.z = linear[2]
        twist.angular.x = angular[0]
        twist.angular.y = angular[1]
        twist.angular.z = angular[2]
        self.pub_cmd_vel.publish(twist) 


    def fn_cal_scan_count(self, list_scan_dis, max_dis):
        list_scan_dis = np.array(list_scan_dis)
        return list_scan_dis[list_scan_dis < max_dis].size
        # scan_count = 0
        # for scan_dis in list_scan_dis:
        #     if 0 < scan_dis < max_dis:
        #         scan_count += 1
        # return scan_count

    def get_lane_info(self, msg):
        # rospy.loginfo(msg)
        self.desired_center = msg.desired_center
        self.lane_pos = msg.lane_pos

    def get_moving_state(self, msg):
        self.moving_type = msg.data

    def get_scan_dis(self, msg):
        self.scan_dis = msg.ranges

        # self.scan_dis = msg.intensities
        
        
    def cbOdom(self, odom_msg):
        quaternion = (odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w)
        self.current_theta = self.euler_from_quaternion(quaternion)

        if (self.current_theta - self.last_current_theta) < -math.pi:
            self.current_theta = 2. * math.pi + self.current_theta
            self.last_current_theta = math.pi
        elif (self.current_theta - self.last_current_theta) > math.pi:
            self.current_theta = -2. * math.pi + self.current_theta
            self.last_current_theta = -math.pi
        else:
            self.last_current_theta = self.current_theta

        self.current_pos_x = odom_msg.pose.pose.position.x
        self.current_pos_y = odom_msg.pose.pose.position.y

    def euler_from_quaternion(self, quaternion):
        theta = tf.transformations.euler_from_quaternion(quaternion)[2]
        return theta

    def main(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('control_driving')
    node = ControlDriving()
    node.main()
