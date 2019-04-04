from PIL import Image
import argparse
from pathlib import Path
from insightface.config import conf
from insightface.mtcnn import MTCNN
from insightface.Learner import face_learner
from insightface.utils import load_facebank, draw_box_name, prepare_facebank
import lz
from torchvision import transforms
from insightface.model import l2_norm

from lz import *
from recognition.embedding import Embedding
import sklearn


class FeaExtractor():
    def __init__(self, **kwargs):
        self.yy_imgs = kwargs.get('yy_imgs')
        self.mx = kwargs.get('mx', False)
        
        self.yy_feas = {}
        self.yy_feas_norms = {}
        if not self.mx:
            conf.work_path = conf.save_path = conf.model_path = Path('./kmodel/')
            conf.net_depth = 152
            conf.net_mode = 'ir_se'
            learner = face_learner(conf, )
            learner.load_state(
                # fixed_str ='ir_se50.pth',
                fixed_str='ir_se152.pth',
                resume_path='./kmodel/'
            )
            learner.model.eval()
            print('learner loaded')
            self.learner = learner
        else:
            model_path = root_path + '../insightface/logs/model-r100-arcface-ms1m-refine-v2/model'
            assert os.path.exists(os.path.dirname(model_path)), 'you need download model to use mxnet version'
            embedding = Embedding(model_path, 0, kwargs.get('gpuid', 3))
            self.embedding = embedding
            print('mx embedding loaded')
        self.conf = conf
        if self.yy_imgs is not None:
            for k, img in self.yy_imgs.items():
                if self.mx:
                    fea, norm = self.extract_fea_mx(img, return_norm=True)
                else:
                    fea, norm = self.extract_fea_th(img, return_norm=True)
                self.yy_feas[k] = fea
                self.yy_feas_norms[k] = norm
    
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
            fea, prenorm_fea = learner.model(x=conf.test_transform(img).cuda().unsqueeze(0), return_norm=True)
            fea_norm = np.sqrt((to_numpy(prenorm_fea) ** 2).sum())
            fea_mirror = learner.model(x=conf.test_transform(mirror).cuda().unsqueeze(0))
            fea = l2_norm(fea + fea_mirror).cpu().numpy().reshape(512)
        if not return_norm:
            return fea
        else:
            return fea, fea_norm
    
    def extract_fea_mx(self, img, return_norm=False):  # img bgr
        fea = self.embedding.get(img, normalize=False)
        norm = np.sqrt((fea ** 2).sum())
        fea_n = sklearn.preprocessing.normalize(fea.reshape(1, -1)).flatten()
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
