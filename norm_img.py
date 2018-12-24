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
from insightface import FeaExtractor

extractor = FeaExtractor(
    yy_imgs=None, gpuid=gpuid,
)
print('face feature ex loaded ')

show = False # False
wait_time = 1000 * 10

# src_dir = f'{work_path}/youeryuan/20180930 新大一班-林蝶老师-29、30/20180930 大一班9.30/9.30/face/'
src_dir = f'{work_path}/youeryuan/20180930 新大一班-林蝶老师-29、30/20180930 大一班9.29/9.29/face/'
assert osp.exists(src_dir), src_dir
vs = [glob.iglob(src_dir + f'/*.{suffix}', recursive=False) for suffix in get_img_suffix()]
vseq = itertools.chain(*vs)

# mkdir_p(dst + '/face')
# mkdir_p(dst + '/proc')

for ind_img, imgfp in enumerate(vseq):
    logging.info(f"--- {imgfp} ---")
    frame = cvb.read_img(imgfp)
    fea, norm = extractor.extract_fea(frame)
    if show:
        print('norm ', norm)
        cvb.show_img(frame,'test', wait_time)
    imgfn = osp.basename(imgfp)
    for suffix in get_img_suffix():
        imgfn = imgfn.replace(suffix, '')
    imgfn = imgfn.strip('.')

print('final how many imgs', ind_img)
