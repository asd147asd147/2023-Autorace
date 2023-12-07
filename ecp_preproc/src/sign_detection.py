#!/usr/bin/env python3
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

# Authors: Leon Jung, [AuTURBO] Ki Hoon Kim (https://github.com/auturbo), Gilbert

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import os
from rospkg import RosPack
from skimage import feature
from PIL import Image

from std_msgs.msg import UInt8
from ecp_preproc.msg import BoundingBox, BoundingBoxes
from imutils import paths
import imutils

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from skimage import feature
import cv2
import numpy as np
import os

from ecp_sign_state import ECP_SIGN_STATE
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))



package = RosPack()
package_path = package.get_path('ecp_preproc')

class SignDetection():
    def __init__(self):
        self.image_topic = rospy.get_param('~image_topic', '/camera/rgb/image_raw')
        self.publish_image = rospy.get_param('~publish_image')
        self.weights_path = os.path.join(package_path, 'models', 'ext_HOG.pkl')
        self.model = self.HOG_KNN()

        self.prev_traffic_sign = ECP_SIGN_STATE.NONE
        self.traffic_sign_count = 0

        rospy.loginfo("HOG+KNN model loaded")

        self.bridge = CvBridge()

        # Define subscribers
        self.image_sub = rospy.Subscriber(self.image_topic, CompressedImage, self.imageCb, queue_size = 1, buff_size = 2**24)

        # Define publishers
        self.pub_traffic = rospy.Publisher('/detect/traffic', UInt8, queue_size=1)
        self.pub_viz_ = rospy.Publisher('/HOG_KNN/detections_image_topic/compressed', CompressedImage, queue_size=1)
        rospy.loginfo("Launched node for object detection")

        # Spin
        rospy.spin()

    def fnCalcMSE(self, arr1, arr2):
        squared_diff = (arr1 - arr2) ** 2
        sum = np.sum(squared_diff)
        num_all = arr1.shape[0] * arr1.shape[1] #cv_image_input and 2 should have same shape
        err = sum / num_all
        return err

    def imageCb(self, data):
        # Convert the image to OpenCV
        np_arr = np.frombuffer(data.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Configure input
        detection_result = self.signDetecting(cv_image, data)

        if detection_result is not None:
            current_traffic_sign = ECP_SIGN_STATE[detection_result.Class.upper()].value
            rospy.logwarn("Detect Sign : %s", detection_result.Class.upper())
            if self.prev_traffic_sign == current_traffic_sign:
                self.traffic_sign_count += 1
            else:
                self.prev_traffic_sign = current_traffic_sign
                self.traffic_sign_count = 0

            if self.traffic_sign_count > 30:
                self.pub_traffic.publish(current_traffic_sign)

        # Visualize detection results
        if (self.publish_image):
            self.visualizeAndPublish(detection_result, cv_image)
        return True
    

    def signDetecting(self, img, data):
        HSV_RED_LOWER = np.array([0, 100, 100])
        HSV_RED_UPPER = np.array([10, 255, 255])

        HSV_RED_LOWER1 = np.array([160, 100, 100])
        HSV_RED_UPPER1 = np.array([179, 255, 255])

        HSV_YELLOW_LOWER = np.array([10, 80, 120])
        HSV_YELLOW_UPPER = np.array([40, 255, 255])

        HSV_BLUE_LOWER = np.array([80, 160, 65])
        HSV_BLUE_UPPER = np.array([140, 255, 255])

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        redBinary = cv2.inRange(hsv, HSV_RED_LOWER, HSV_RED_UPPER)
        redBinary1 = cv2.inRange(hsv, HSV_RED_LOWER1, HSV_RED_UPPER1)
        redBinary = cv2.bitwise_or(redBinary, redBinary1)

        yellowBinary = cv2.inRange(hsv, HSV_YELLOW_LOWER, HSV_YELLOW_UPPER)
        blueBinary = cv2.inRange(hsv, HSV_BLUE_LOWER, HSV_BLUE_UPPER)
        binary = cv2.bitwise_or( blueBinary, (redBinary))

        _, contours, hierachy = cv2.findContours(binary, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            binary = cv2.drawContours(binary, [cnt], -1, (255,255,255), -1)

        _, goodContours, hierachy = cv2.findContours(binary, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_and(binary, gray)

        for cnt in goodContours:
            area = cv2.contourArea(cnt)
            if area > 800.0 :
                x, y, w, h = cv2.boundingRect(cnt)
                rate = w / h
                if rate > 0.8 and rate < 1.2 :
                    cv2.rectangle(img, (x, y), (x+w, y+h), (200, 152, 50), 2)
                    inputImage = img[y:y+h, x:x+w]
                    inputImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
                    logo = cv2.resize(inputImage, (128,128))
                    (H, hogImage) = feature.hog(logo, orientations=8, pixels_per_cell=(12, 12), \
					    cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2", visualize=True)
                    pred = self.model.predict(H.reshape(1, -1))[0]

                    detection_msg = BoundingBox()
                    detection_msg.xmin = x
                    detection_msg.xmax = x+w
                    detection_msg.ymin = y
                    detection_msg.ymax = y+h
                    detection_msg.probability = 1.
                    detection_msg.Class = pred.title()
                    return detection_msg
        return None


    def visualizeAndPublish(self, output, imgIn):
        # Copy image and visualize
        imgOut = imgIn.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.6
        thickness = 2
        if output is not None:
            label = output.Class
            x_p1 = output.xmin
            y_p1 = output.ymin
            x_p3 = output.xmax
            y_p3 = output.ymax
            
            # Create rectangle
            cv2.rectangle(imgOut, (int(x_p1), int(y_p1)), (int(x_p3), int(y_p3)), (200, 152, 50),thickness)
            text = ('{:s}').format(label)
            cv2.putText(imgOut, text, (int(x_p1), int(y_p1+20)), font, fontScale, (255,255,255), thickness ,cv2.LINE_AA)
        # Publish visualization image
        self.pub_viz_.publish(self.bridge.cv2_to_compressed_imgmsg(imgOut, "jpg"))

    def HOG_KNN(self):
        data = []
        labels = []

        # loop over the image paths in the training set
        for imagePath in paths.list_images('../traffic_image/'):
        # extract the make of the car
            make = imagePath.split("/")[-1].split('.')[-2].split('_')[0]
            rospy.logwarn(make)

            # load the image, convert it to grayscale, and detect edges
            image = cv2.imread(imagePath)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            gray = cv2.resize(gray, (400, 400))

            kernel = np.ones((3, 3), np.uint8)
            erosion = cv2.erode(gray, kernel, iterations=1)

            logo = cv2.resize(erosion, (128,128))
            # cv2.imshow("logo", logo)
            # cv2.waitKey(0)

            # extract Histogram of Oriented Gradients from the logo
            H = feature.hog(logo, orientations=8, pixels_per_cell=(12, 12),
                cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2")

            # update the data and labels
            data.append(H)
            labels.append(make)

            for r in range(0,360, 5):
                M = cv2.getRotationMatrix2D((64, 64), r, 1.0)
                rotate = cv2.warpAffine(logo, M, (128, 128))
                H = feature.hog(rotate, orientations=8, pixels_per_cell=(12, 12),
                cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2")

                data.append(H)
                labels.append(make)
                

        # "train" the nearest neighbors classifier
        model = KNeighborsClassifier(n_neighbors=1)
        # model = SVC(gamma = 'auto')

        model.fit(data, labels)

        return model



    def main(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('sign_detection')
    node = SignDetection()
    node.main()
