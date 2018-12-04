import lz
import datetime
import pims
from lz import *

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
from ssha_detector import SSHDetector
from skimage import transform as trans

cv2.namedWindow('test', cv2.WINDOW_NORMAL)
lz.init_dev(lz.get_dev())
scales = [4032, 3024]
# scales = [200, 600]
detector = SSHDetector('./kmodel/e2e', 0)


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
        # cv2.imwrite(f"{dst}/proc/{ind}.png", img)
        import cvbase as cvb
        cvb.show_img(img, 'test', wait_time=1000 // 80 * 80)

    diff = timeb - timea
    print('detection uses', diff.total_seconds(), 'seconds')
    print('find', faces.shape[0], 'faces')
    if 'im_scale' in locals() and im_scale != 1:
        faces[:, :4] /= im_scale
        faces[:, 5:] /= im_scale
    return faces


def precess_func(img):
    img = cvb.rgb2bgr(img)
    img = np.rot90(img, 3, axes=(0, 1))
    img = img.copy()
    return img


src_dir = '/home/xinglu/work/youeryuan/20180930 新大一班-林蝶老师-29、30/20180930 大一班9.29/9.29正、侧、背/正/'
imgs = pims.ImageSequence(f'{src_dir}/*.JPG', process_func=precess_func)
imgs2 = pims.ImageSequence(f'{src_dir}/*.JPG', process_func=precess_func)


def get_res(imgs):
    res = {}
    for img, path in zip(imgs, imgs._filepaths):
        img = img.copy()
        # cls = path.split('/')[-1]
        print(path)
        # cvb.show_img(img, wait_time=0)
        faces = detect_face(img)
        assert faces.shape[0] >= 1
        for ind_faces in range(faces.shape[0]):
            bbox, score, kps = faces[ind_faces, :4], faces[ind_faces, 4], faces[ind_faces, 5:].reshape(5, 2)
            area_face = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            area_ttl = img.shape[0] * img.shape[1]
            iou = area_face / area_ttl
            if iou < 0.005:  continue
            warp_face = preprocess(img, bbox=bbox,
                                   landmark=kps
                                   )
            if warp_face.mean() > 200:
                logging.error(f'{warp_face.mean()}' )
                continue
            # cvb.show_img(warp_face, wait_time=1000 // 20)
            assert path not in res
            res[path] = warp_face

            # kp_templ = np.array([
            #     [30.2946, 51.6963],
            #     [65.5318, 51.5014],
            #     [48.0252, 71.7366],
            #     [33.5493, 92.3655],
            #     [62.7299, 92.2041]], dtype=np.float32)
            # if warp_face.shape[1] == 112:
            #     kp_templ[:, 0] += 8.0
            # kp_templ = kp_templ.ravel()
            # for knum in range(5):
            #     cv2.circle(warp_face, (kp_templ[2 * knum], kp_templ[2 * knum + 1]), 1, [0, 0, 255], 2)
            # plt.imshow(warp_face[..., ::-1])
            # plt.show()

        assert path in res
    return res


res = get_res(imgs)
res2 = get_res(imgs2)
# lz.msgpack_dump([res, res2], lz.work_path + 'yy.yy2.pk')

# res = {}
# for ind, frame in enumerate(v):
#     frame = cvb.rgb2bgr(frame)
#     faces = detec_face(frame)
#     if faces.shape[0] != 0:
#         res[ind] = faces
#         logging.info(str(faces.shape))
#     # if ind > 500: break
#
# lz.msgpack_dump(res, 'yy.mov.pk')
