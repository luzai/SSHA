import datetime
import torch, cv2
# sys.path.append('.')
from ssha_detector import SSHDetector
import lz
from lz import *

lz.init_dev(get_dev())
gpuid = 0
import cvbase as cvb
import itertools
from insightface import FeaExtractor, cal_sim
enroll_dir = f'/home/xinglu/work/youeryuan/20180930 新大一班-林蝶老师-29、30/9.30正、侧、背/'
enroll = msgpack_load(enroll_dir + 'face.pk')
extractor = FeaExtractor(
    yy_imgs=enroll, gpuid=gpuid,
)
print('face feature ex loaded ')

verif_dir = f'/home/xinglu/work/youeryuan/20180930 新大一班-林蝶老师-29、30/20180930 大一班9.30/9.30/'
verif = msgpack_load(verif_dir + 'face/info.pk')

for ind, v in enumerate(verif):
    fea = v['fea']
    sim = cal_sim([fea], list(extractor.yy_feas.values()))
    sim.max
print('ok')
