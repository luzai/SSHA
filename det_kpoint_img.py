import cv2
import datetime
# sys.path.append('.')
from ssha_detector import SSHDetector
import lz
import pims
from lz import *
import cvbase as cvb
import itertools

lz.init_dev(3)
scales = [4032, 3024]
# scales = [3024, 4032]
#  3456, 4608

detector = SSHDetector('./kmodel/e2e', 0)
cv2.namedWindow('test', cv2.WINDOW_NORMAL)

src_dir = f'{work_path}/youeryuan/20180930 新大一班-林蝶老师-29、30/20180930 大一班9.30/9.30/'
vs = [glob.glob(src_dir + f'/*.{suffix}', recursive=True) for suffix in get_img_suffix()]
v = itertools.chain(vs)
dst = src_dir


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
        cvb.write_img(img, f"{dst}/proc/{ind}.png", )
        cvb.show_img(img, 'test', wait_time=1000 // 80 * 80)
    
    diff = timeb - timea
    print('detection uses', diff.total_seconds(), 'seconds')
    print('find', faces.shape[0], 'faces')
    if 'im_scale' in locals() and im_scale != 1:
        faces[:, :4] /= im_scale
        faces[:, 5:] /= im_scale
    return faces


res = {}
for ind, imgfp in enumerate(v):
    frame = cvb.read_img(imgfp)
    imgfn = osp.basename(imgfp)
    frame = cvb.rgb2bgr(frame)
    # frame = np.rot90(frame, 3).copy()
    faces = detect_face(frame, imgfn)
    if faces.shape[0] != 0:
        res[ind] = faces
        logging.info(str(faces.shape))
    for ind_faces in range(len(faces)):
        bbox, score, kps = faces[ind_faces, :4], faces[ind_faces, 4], faces[ind_faces, 5:].reshape(5, 2)
        warp_face = preprocess(frame, bbox=bbox,
                               landmark=kps
                               )
        # plt_imshow(face_img)
        # plt.show()
        
        cvb.write_img(warp_face, f'{dst}/face/{ind}.{ind_faces}.png')
        # if ind > 500: break

lz.msgpack_dump(res, dst + 'kpts.pk')
