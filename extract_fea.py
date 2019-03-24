import datetime, lz
from ssha_detector import SSHDetector
from lz import *
init_dev(3)

scales = [4032, 3024]
# detector = SSHDetector('./kmodel/e2e', 0)
# cv2.namedWindow('test', cv2.WINDOW_NORMAL)
from insightface import FeaExtractor

extractor = FeaExtractor()
# src_dir = '/home/xinglu/work/youeryuan/20180930 新大一班-林蝶老师-29、30/20180930 大一班9.30/9.30正、侧、背/face/'
src_dir = lz.root_path + '/test_image/face'
feas = {}
faces = {}
norms = {}
for ind, imgp in enumerate(glob.glob(f'{src_dir}/*.png')):
    print(ind, imgp)
    img = cvb.read_img(imgp)
    fea, norm = extractor.extract_fea_th(img, return_norm=True)
    feas[ind] = fea
    faces[ind] = img
    norms[ind] = norm

lz.msgpack_dump(faces, src_dir + '/face.pk')
lz.msgpack_dump(feas, src_dir + '/fea.pk')
lz.msgpack_dump(norms, src_dir + '/fea.norm.pk')
