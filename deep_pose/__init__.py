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
        spath = home_path + '.torch/models/hopenet_robust_alpha1.pkl'
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
    
    def det(self, frame, nose, draw_on, **kwargs):
        img = frame.copy()
        # cvb.show_img(frame, wait_time=0)
        # cv2_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
        yaw, pitch, roll = yaw.item(), pitch.item(), roll.item()
        yaw = -yaw
        # pitch_rad = pitch * np.pi / 180
        # yaw_rad = (yaw * np.pi / 180)
        # roll_rad = roll * np.pi / 180
        utils.draw_axis(draw_on, yaw_predicted, pitch_predicted, roll_predicted,
                        tdx=nose[0],
                        tdy=nose[1],
                        size=frame.shape[0])
        return yaw, pitch, roll


if __name__ == '__main__':
    imgp = work_path + 'face.yy/gallery/11_0.png'
    img = cvb.read_img(imgp)
    print(img.shape)
    det = PoseDetector()
    yaw, pitch, roll = det.det(img, [51, 51], img)
    print(yaw, pitch, roll)
    # cvb.show_img(img)
    # plt_imshow(img, 'bgr')
    # plt.show()
