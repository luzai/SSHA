import cv2
from PIL import Image
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe, Value, Array
import torch
from insightface.config import get_config
from insightface.mtcnn import MTCNN
from insightface.Learner import face_learner
from insightface.utils import load_facebank, draw_box_name, prepare_facebank
import lz
from torchvision import transforms
from insightface.model import l2_norm

import pims, cvbase as cvb
from lz import *


class FeaExtractor():
    def __init__(self, day='yy'):
        conf = get_config(False)
        yy, yy2 = msgpack_load(work_path + 'yy.yy2.fea.pk')
        self.yy = yy
        self.yy2 = yy2
        imgs1, imgs2 = msgpack_load(work_path + 'yy.yy2.pk')
        self.yy_imgs = imgs1 if day == 'yy' else imgs2
        self.yy_feas = yy if day == 'yy' else yy2
        self.yy_feas_norms = {k: np.sqrt((fea ** 2).sum()) for k, fea in self.yy_feas.items()}
        learner = face_learner(conf, True)
        # learner.threshold = 0.72

        if conf.device.type == 'cpu':
            learner.load_state(conf, 'cpu_final.pth', True, True)
        else:
            learner.load_state(conf,
                               # '2018-10-24-12-02_accuracy:0.876_step:304456_final.pth'
                               'ir_se50.pth'
                               , True, True)
        learner.model.eval()
        print('learner loaded')
        self.learner = learner
        self.conf = conf
        self.day = day

    def extract_fea(self, res):
        res3 = {}
        for path, img in res.items():
            res3[path] = self.extract_fea_from_img(img)
        return res3

    def extract_fea_from_img(self, img, return_norm=False):  # img show be bgr
        learner = self.learner
        conf = self.conf
        img = img.copy()[..., ::-1].reshape(112, 112, 3)
        img = Image.fromarray(img)
        mirror = transforms.functional.hflip(img)
        with torch.no_grad():
            fea, prenorm_fea = learner.model(x=conf.test_transform(img).cuda().unsqueeze(0), need_prenorm=True)
            fea_norm = np.sqrt((to_numpy(prenorm_fea) ** 2).sum())
            print('--> face norm', fea_norm)
            fea_mirror = learner.model(x=conf.test_transform(mirror).cuda().unsqueeze(0))
            fea = l2_norm(fea + fea_mirror).cpu().numpy().reshape(512)
        if not return_norm:
            return fea
        else:
            return fea, fea_norm

    def compare(self, img, return_norm=False):
        fea, norm = self.extract_fea_from_img(img, return_norm=True)
        # imgs1, imgs2 = msgpack_load(work_path + 'yy.yy2.pk')
        if self.day == 'yy':
            sim = cal_sim([fea], list(self.yy.values()))
        else:
            sim = cal_sim([fea], list(self.yy2.values()))
        if not return_norm:
            return sim
        else:
            return sim, norm


if __name__ == '__main__':
    lz.init_dev(lz.get_dev())
    extractor = FeaExtractor()
    # imgs1, imgs2 = msgpack_load(work_path + 'yy.yy2.pk')
    # # plt_imshow(list(imgs1.values())[0])
    # # plt.show()
    # extractor.compare(list(imgs1.values())[0])
    # fea1 = extractor.extract_fea(imgs1)
    # fea2 = extractor.extract_fea(imgs2)
    # # lz.msgpack_dump([fea1, fea2], work_path + 'yy.yy2.fea.pk')

    imgs = pims.ImageSequence('/data2/xinglu/work/face.yy2/gallery/*.png')
    res = {}
    for ind, (img, p) in enumerate(zip(imgs, imgs._filepaths)):
        _, norm = extractor.extract_fea_from_img(img, return_norm=True)
        res[p] = (norm)
        # if ind>100:
        #     break
    lz.msgpack_dump(res, work_path + 't.pk')
