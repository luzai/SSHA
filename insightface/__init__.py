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

import cvbase as cvb
from lz import *
from recognition.embedding import Embedding
import sklearn


class FeaExtractor():
    def __init__(self, **kwargs):
        conf = get_config(False)
        self.yy_imgs = kwargs.get('yy_imgs')
        self.mx = kwargs.get('mx', False)

        self.yy_feas = {}
        self.yy_feas_norms ={}
        if not self.mx:
            learner = face_learner(conf, True)
            learner.load_state(conf,
                               # '2018-10-24-12-02_accuracy:0.876_step:304456_final.pth'
                               'ir_se50.pth'
                               , True, True)
            learner.model.eval()
            print('learner loaded')
            self.learner = learner
        
        else:
            model_path = root_path + '../insightface/Evaluation/IJB/pretrained_models/MS1MV2-ResNet100-Arcface/model'
            assert os.path.exists(os.path.dirname(model_path)), os.path.dirname(model_path)
            gpu_id = 0
            embedding = Embedding(model_path, 0, gpu_id)
            self.embedding = embedding
            print('mx embedding loaded')
        self.conf = conf
        
        for k, img in self.yy_imgs.items():
            if self.mx:
                fea, norm = self.extract_fea_mx(img, return_norm=True)
            else:
                fea, norm = self.extract_fea_th(img, return_norm=True)
            self.yy_feas[k] = fea
            self.yy_feas_norms[k] = norm
            # break
    
    def extract_fea(self, img):
        if self.mx:
            fea, norm = self.extract_fea_mx(img, return_norm=True)
        else:
            fea, norm = self.extract_fea_th(img, return_norm=True)
        return fea, norm
    
    def extract_fea_th(self, img, return_norm=False):  # here img show be bgr
        learner = self.learner
        conf = self.conf
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
    
    def extract_fea_mx(self, img, return_norm=False):
        fea = self.embedding.get(img, normalize=False)
        norm = np.sqrt( (fea ** 2).sum() )
        fea_n = sklearn.preprocessing.normalize(fea.reshape(1,-1)).flatten()
        if not return_norm:
            return fea_n
        else:
            return fea_n, norm
    
    def compare(self, img, return_norm=False):
        fea, norm = self.extract_fea(img)
        sim = cal_sim([fea], list(self.yy_feas.values()))
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
        _, norm = extractor.extract_fea_th(img, return_norm=True)
        res[p] = (norm)
        # if ind>100:
        #     break
    lz.msgpack_dump(res, work_path + 't.pk')
