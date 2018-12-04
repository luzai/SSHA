import cv2
import sys
import numpy as np
import datetime
# sys.path.append('.')
from ssha_detector import SSHDetector
import lz

lz.init_dev(3)
scales = [1080*2, 1920*2]
# scales = [3024, 4032]
#  3456, 4608

detector = SSHDetector('./kmodel/e2e', 0)

import pims
from lz import *
import cvbase as cvb

cv2.namedWindow('test', cv2.WINDOW_NORMAL)
# v = pims.Video('face.yy2/video.mov')
# len(v), v.frame_shape

src_dir = '/home/xinglu/work/youeryuan/20180930 新大一班-林蝶老师-29、30/20180930 大一班9.29/9.29正、侧、背/正/'
v = pims.ImageSequence(src_dir+'/*.JPG')
dst = '/home/xinglu/work/youeryuan/20180930 新大一班-林蝶老师-29、30/20180930 大一班9.29/9.29正、侧、背/'


def detect_face(img, ind=None):
    img = img.copy()
    im_shape = img.shape
    # print(im_shape)
    target_size = scales[0]
    max_size = scales[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    if im_size_min > target_size or im_size_max > max_size:
        im_scale = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale)

        # print('resize to', img.shape)
    # for i in xrange(t-1): #warmup
    #   faces = detector.detect(img)
    timea = datetime.datetime.now()
    faces = detector.detect(img, threshold=0.9)
    timeb = datetime.datetime.now()
    if faces.shape[0] != 0:
        for num in range(faces.shape[0]):
            bbox = faces[num, 0:4]
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            kpoint = faces[num, 5:15]
            for knum in range(5):
                cv2.circle(img, (kpoint[2 * knum], kpoint[2 * knum + 1]), 1, [0, 0, 255], 2)
        if ind is None:
            ind = randomword()
        cv2.imwrite(f"{dst}/proc/{ind}.png", img)
        import cvbase as cvb
        cvb.show_img(img, 'test', wait_time=1000 // 80 * 80)

    diff = timeb - timea
    print('detection uses', diff.total_seconds(), 'seconds')
    print('find', faces.shape[0], 'faces')
    if 'im_scale' in locals() and im_scale != 1:
        faces[:, :4] /= im_scale
        faces[:, 5:] /= im_scale
    return faces


res = {}
for ind, frame in enumerate(v):
    frame = cvb.rgb2bgr(frame)
    frame = np.rot90(frame, 3).copy()
    faces = detect_face(frame, ind)
    if faces.shape[0] != 0:
        res[ind] = faces
        logging.info(str(faces.shape))
    for faces_ind in range(len(faces)):
        x, y, x2, y2 = faces[faces_ind, :4]
        x, y, x2, y2 = list(map(int, [x, y, x2, y2]))
        face_img = frame[y:y2, x:x2, :]
        cvb.write_img(face_img, f'{dst}/face/{ind}.png')
        # if ind > 500: break

lz.msgpack_dump(res, dst + 'face.pk')
