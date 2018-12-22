import datetime, lz
from ssha_detector import SSHDetector

scales = [4032, 3024]
detector = SSHDetector('./kmodel/e2e', 0)
try:
    import pims
except:
    print('!! no pims')
from lz import *

init_dev(get_dev(mem_thresh=(.5, .5)))

cv2.namedWindow('test', cv2.WINDOW_NORMAL)
from insightface import FeaExtractor

extractor = FeaExtractor()
# src_dir = '/home/xinglu/work/youeryuan/20180930 新大一班-林蝶老师-29、30/20180930 大一班9.29/9.29正、侧、背/face/'
src_dir = '/home/xinglu/work/youeryuan/20180930 新大一班-林蝶老师-29、30/20180930 大一班9.30/9.30正、侧、背/face/'
# src_dir = '/home/xinglu/work/youeryuan/20180930 新中二班-缪蕾老师班-29、30/中二班9月29日-正侧背/face/'
# src_dir = '/home/xinglu/work/youeryuan/20180930 新中二班-缪蕾老师班-29、30/中二班9月30日-正侧背/face/'
feas = {}
faces = {}
norms = {}
for ind, imgp in enumerate(glob.glob(f'{src_dir}/*.png')):
    print(ind, imgp)
    img = cvb.read_img(imgp)
    fea, norm = extractor.extract_fea_th(img, return_norm=True)  # BGR norm 4.75 # after 43 better
    feas[ind] = fea
    faces[ind] = img
    norms[ind] = norm

lz.msgpack_dump(faces, src_dir + '/face.pk')
lz.msgpack_dump(feas, src_dir + '/fea.pk')
lz.msgpack_dump(norms, src_dir + '/fea.norm.pk')

