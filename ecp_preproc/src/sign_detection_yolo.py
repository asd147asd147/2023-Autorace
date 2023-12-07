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
from skimage.transform import resize

from ecp_preproc.msg import BoundingBox, BoundingBoxes

# Deep learning imports
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from models import *
from utils.utils import *

package = RosPack()
package_path = package.get_path('ecp_preproc')

class SignDetection():
    def __init__(self):
        weights_name = rospy.get_param('~weights_name', 'yolov3.weights')
        self.weights_path = os.path.join(package_path, 'models', weights_name)
        rospy.loginfo("Found weights, loading %s", self.weights_path)

        if not os.path.isfile(self.weights_path):
            raise IOError(('{:s} not found.').format(self.weights_path))
        
        self.image_topic = rospy.get_param('~image_topic', '/camera/rgb/image_raw')
        self.confidence_th = rospy.get_param('~confidence', 0.7)
        self.nms_th = rospy.get_param('~nms_th', 0.3)

        config_name = rospy.get_param('~config_name', 'yolov3.cfg')
        self.config_path = os.path.join(package_path, 'config', config_name)
        classes_name = rospy.get_param('~classes_name', 'coco.names')
        self.classes_path = os.path.join(package_path, 'classes', classes_name)
        self.gpu_id = rospy.get_param('~gpu_id', 0)
        self.network_img_size = rospy.get_param('~img_size', 416)
        self.publish_image = rospy.get_param('~publish_image')

        self.h = 0
        self.w = 0
        
        # Load net
        self.model = Darknet(self.config_path, img_size=self.network_img_size)
        self.model.load_weights(self.weights_path)
        if torch.cuda.is_available():
            self.model.cuda()

        self.model.eval() # Set in evaluation mode
        rospy.loginfo("Deep neural network loaded")

        self.bridge = CvBridge()

        # Load classes
        self.classes = load_classes(self.classes_path) # Extracts class labels from file
        self.classes_colors = {}
        
        # Define subscribers
        self.image_sub = rospy.Subscriber(self.image_topic, CompressedImage, self.imageCb, queue_size = 1, buff_size = 2**24)

        # Define publishers
        self.pub_ = rospy.Publisher('/yolo/detected_objects_topic', BoundingBoxes, queue_size=1)
        self.pub_viz_ = rospy.Publisher('/yolo/detections_image_topic/compressed', CompressedImage, queue_size=1)
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
        self.cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Initialize detection results
        detection_results = BoundingBoxes()
        detection_results.header = data.header
        detection_results.image_header = data.header

        # Configure input
        input_img = self.imagePreProcessing(self.cv_image)

        if torch.cuda.is_available():
            input_img = Variable(input_img.type(torch.cuda.FloatTensor))
        else:
            input_img = Variable(input_img.type(torch.FloatTensor))
        
        # Get detections from network
        with torch.no_grad():
            detections = self.model(input_img)
            detections = non_max_suppression(detections, 80, self.confidence_th, self.nms_th)
        
        # Parse detections
        if detections[0] is not None:
            for detection in detections[0]:
                # Get xmin, ymin, xmax, ymax, confidence and class
                xmin, ymin, xmax, ymax, _, conf, det_class = detection
                pad_x = max(self.h - self.w, 0) * (self.network_img_size/max(self.h, self.w))
                pad_y = max(self.w - self.h, 0) * (self.network_img_size/max(self.h, self.w))
                unpad_h = self.network_img_size-pad_y
                unpad_w = self.network_img_size-pad_x
                xmin_unpad = ((xmin-pad_x//2)/unpad_w)*self.w
                xmax_unpad = ((xmax-xmin)/unpad_w)*self.w + xmin_unpad
                ymin_unpad = ((ymin-pad_y//2)/unpad_h)*self.h
                ymax_unpad = ((ymax-ymin)/unpad_h)*self.h + ymin_unpad

                # Populate darknet message
                detection_msg = BoundingBox()
                detection_msg.xmin = xmin_unpad
                detection_msg.xmax = xmax_unpad
                detection_msg.ymin = ymin_unpad
                detection_msg.ymax = ymax_unpad
                detection_msg.probability = conf
                detection_msg.Class = self.classes[int(det_class)]

                # Append in overall detection message
                detection_results.bounding_boxes.append(detection_msg)
	
        # Publish detection results
        self.pub_.publish(detection_results)

        # Visualize detection results
        if (self.publish_image):
            self.visualizeAndPublish(detection_results, self.cv_image)
        return True
    

    def imagePreProcessing(self, img):
        # Extract image and shape
        img = np.copy(img)
        img = img.astype(float)
        height, width, channels = img.shape
        
        if (height != self.h) or (width != self.w):
            self.h = height
            self.w = width
            
            # Determine image to be used
            self.padded_image = np.zeros((max(self.h,self.w), max(self.h,self.w), channels)).astype(float)
            
        # Add padding
        if (self.w > self.h):
            self.padded_image[(self.w-self.h)//2 : self.h + (self.w-self.h)//2, :, :] = img
        else:
            self.padded_image[:, (self.h-self.w)//2 : self.w + (self.h-self.w)//2, :] = img
        
        # Resize and normalize
        input_img = resize(self.padded_image, (self.network_img_size, self.network_img_size, 3))/255.

        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))

        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()
        input_img = input_img[None]

        return input_img


    def visualizeAndPublish(self, output, imgIn):
        # Copy image and visualize
        imgOut = imgIn.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.6
        thickness = 2
        for index in range(len(output.bounding_boxes)):
            label = output.bounding_boxes[index].Class
            x_p1 = output.bounding_boxes[index].xmin
            y_p1 = output.bounding_boxes[index].ymin
            x_p3 = output.bounding_boxes[index].xmax
            y_p3 = output.bounding_boxes[index].ymax
            confidence = output.bounding_boxes[index].probability

            # Find class color
            if label in self.classes_colors.keys():
                color = self.classes_colors[label]
            else:
                # Generate a new color if first time seen this label
                color = np.random.randint(0,198,3)
                self.classes_colors[label] = color
            
            # Create rectangle
            cv2.rectangle(imgOut, (int(x_p1), int(y_p1)), (int(x_p3), int(y_p3)), (int(color[0]),int(color[1]),int(color[2])),thickness)
            text = ('{:s}: {:.3f}').format(label,confidence)
            cv2.putText(imgOut, text, (int(x_p1), int(y_p1+20)), font, fontScale, (255,255,255), thickness ,cv2.LINE_AA)
        # Publish visualization image
        self.pub_viz_.publish(self.bridge.cv2_to_compressed_imgmsg(imgOut, "jpg"))


    def main(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('sign_detection')
    node = SignDetection()
    node.main()
