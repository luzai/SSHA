from lz import *

from deep_pose import datasets, hopenet, utils
import sys, os, argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image

xrange = range


class PoseDetector():
    def __init__(self):
        model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
        spath = work_path + 'hopenet_robust_alpha1.pkl'
        if osp.exists(spath):
            print('load', spath)
            saved_state_dict = torch.load(spath)
            model.load_state_dict(saved_state_dict)

        self.transformations = transforms.Compose([transforms.Scale(224),
                                                   transforms.CenterCrop(224), transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                        std=[0.229, 0.224, 0.225])])

        model.cuda()
        self.model = model
        idx_tensor = [idx for idx in xrange(66)]
        idx_tensor = torch.FloatTensor(idx_tensor).cuda()
        self.idx_tensor = idx_tensor

    def det(self, frame, bbox ,  conf=None,  draw_on=None, **kwargs):
        frame = frame.copy()
        [x_min, y_min, x_max, y_max] = bbox[:4]
        bbox_width = abs(x_max - x_min)
        bbox_height = abs(y_max - y_min)
        x_min -= 3 * bbox_width / 4
        x_max += 3 * bbox_width / 4
        y_min -= 3 * bbox_height / 4
        y_max += bbox_height / 4
        # todo or
        # x_min -= 50
        # x_max += 50
        # y_min -= 50
        # y_max += 30

        x_min = int(x_min)
        y_min = int(y_min)
        x_max = int(x_max)
        y_max = int(y_max)

        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(frame.shape[1], x_max)
        y_max = min(frame.shape[0], y_max)

        cv2_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2_frame[y_min:y_max, x_min:x_max]
        img = Image.fromarray(img)
        img = self.transformations(img)
        img_shape = img.size()
        img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
        img = Variable(img).cuda()

        yaw, pitch, roll = self.model(img)
        yaw_predicted = F.softmax(yaw)
        pitch_predicted = F.softmax(pitch)
        roll_predicted = F.softmax(roll)
        # Get continuous predictions in degrees.
        yaw_predicted = torch.sum(yaw_predicted.data[0] * self.idx_tensor) * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted.data[0] * self.idx_tensor) * 3 - 99
        roll_predicted = torch.sum(roll_predicted.data[0] * self.idx_tensor) * 3 - 99

        yaw, pitch, roll = yaw_predicted, pitch_predicted, roll_predicted
        pitch = pitch * np.pi / 180
        yaw = -(yaw * np.pi / 180)
        roll = roll * np.pi / 180
        if draw_on is None:
            utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted,
                            tdx=(x_min + x_max) / 2,
                            tdy=(y_min + y_max) / 2,
                            size=bbox_height / 2)
        else:
            utils.draw_axis(draw_on, yaw_predicted, pitch_predicted, roll_predicted,
                            tdx=(x_min + x_max) / 2,
                            tdy=(y_min + y_max) / 2,
                            size=bbox_height / 2)
        return yaw, pitch, roll


if __name__ == '__main__':
    imgp = work_path + 'face.yy/gallery/11_0.png'
    img = cvb.read_img(imgp)
    det = PoseDetector()
    det.det(img, 0, 0, img.shape[1], img.shape[0])
    cvb.show_img(img)
