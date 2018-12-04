import cv2
import sys
import numpy as np
import datetime
#sys.path.append('.')
from ssha_detector import SSHDetector
import lz
lz.init_dev(3)
scales = [1080, 1920]
# scales = [200, 600]
t = 2
detector = SSHDetector('./kmodel/e2e', 0)

f = 'test_image/test_11.jpg'
if len(sys.argv) > 1:
    f = sys.argv[1]
img = cv2.imread(f)


# cvb.show_img(imgs[0])


def detec_face(img):
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
    faces = detector.detect(img, threshold=0.5)
    timeb = datetime.datetime.now()
    if faces.shape[0] != 0:
        for num in range(faces.shape[0]):
            bbox = faces[num, 0:4]
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            kpoint = faces[num, 5:15]
            for knum in range(5):
                random_colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (25, 9, 96), (39, 142, 40), (223, 210, 11),
                                 (209, 50, 217), (140, 200, 153)]

                cv2.circle(img, (kpoint[2 * knum], kpoint[2 * knum + 1]), 1, random_colors[knum], 2)
        import cvbase as cvb
        cvb.show_img(img, )
        cv2.imwrite("res.jpg", img)
    diff = timeb - timea
    # print('detection uses', diff.total_seconds(), 'seconds')
    # print('find', faces.shape[0], 'faces')

    return faces


def main1():
    faces = detec_face(img)
    print(faces, faces.shape)

    scales = [1200, 1600]
    # scales = [200, 600]
    t = 2
    detector = SSHDetector('./kmodel/e2e', 88)

    f = '../sample-images/t1.jpg'
    f = 'test_image/test_11.jpg'
    if len(sys.argv)>1:
      f = sys.argv[1]
    img = cv2.imread(f)
    im_shape = img.shape
    print(im_shape)
    target_size = scales[0]
    max_size = scales[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    if im_size_min>target_size or im_size_max>max_size:
      im_scale = float(target_size) / float(im_size_min)
      # prevent bigger axis from being more than max_size:
      if np.round(im_scale * im_size_max) > max_size:
          im_scale = float(max_size) / float(im_size_max)
      img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale)
      print('resize to', img.shape)
    # for i in xrange(t-1): #warmup
    #   faces = detector.detect(img)
    timea = datetime.datetime.now()
    faces = detector.detect(img, threshold=0.5)
    timeb = datetime.datetime.now()
    for num in range(faces.shape[0]):
      bbox = faces[num, 0:4]
      cv2.rectangle(img, (bbox[0],bbox[1]),(bbox[2], bbox[3]), (0,255, 0), 2)
      kpoint = faces[num, 5:15]
      for knum in range(5):
          cv2.circle(img, (kpoint[2*knum], kpoint[2*knum+1]), 1, [0,0,255], 2)

    cv2.imwrite("res.jpg", img)
    diff = timeb - timea
    print('detection uses', diff.total_seconds(), 'seconds')
    print('find', faces.shape[0], 'faces')
