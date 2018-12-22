import datetime
import torch, cv2
# sys.path.append('.')
from ssha_detector import SSHDetector
import lz
from lz import *
import cvbase as cvb
import itertools
from insightface import FeaExtractor

lz.init_dev(1)

extractor = FeaExtractor(
    yy_imgs=None,
)

scales = [4032, 3024]  # scales = [3024, 4032] #  3456, 4608
show = False

detector = SSHDetector('./kmodel/e2e', 0)
if show:
    cv2.namedWindow('test', cv2.WINDOW_NORMAL)

src_dir = f'{work_path}/youeryuan/20180930 新大一班-林蝶老师-29、30/20180930 大一班9.30/9.30/'
vs = [glob.iglob(src_dir + f'/*.{suffix}', recursive=True) for suffix in get_img_suffix()]
v = itertools.chain(*vs)
dst = src_dir


def detect_face(img, imgfn=None, save=False):
    img = img.copy()
    im_shape = img.shape
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

    faces = detector.detect(img, threshold=0.9)  # todo critetion 1
    if faces.shape[0] != 0:
        for num in range(faces.shape[0]):
            score = faces[num, 4]
            assert score >= 0.9
            bbox = faces[num, 0:4]
            label_text = 'det {:.02f}'.format(bbox[4])
            cv2.putText(img, label_text, (int(bbox[0]),
                                          int(bbox[1] - 2)),
                        cv2.FONT_HERSHEY_COMPLEX, 1., (0, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            kpoint = faces[num, 5:15]
            for knum in range(5):
                cv2.circle(img, (kpoint[2 * knum], kpoint[2 * knum + 1]), 1, [0, 0, 255], 2)
        if imgfn is None:
            imgfn = randomword()
        if save:
            cvb.write_img(img, f"{dst}/proc/{imgfn}.png", )
        if show:
            cvb.show_img(img, 'test', wait_time=1000 // 80 * 80)
    lz.timer.since_last_check(f'detection on 1 img, find {faces.shape[0]} faces')
    if 'im_scale' in locals() and im_scale != 1:
        faces[:, :4] /= im_scale
        faces[:, 5:] /= im_scale
    return faces, img


def align_face(frame, imgfn, faces, drawon):
    img = frame = frame.copy()
    warp_faces = []
    for num, landmarks in enumerate(faces[:, 5:]):
        bbox = faces[num, 0:5]
        imgpts, modelpts, rotate_degree, nose = face_orientation(frame, landmarks)  # roll, pitch, yaw

        # pose_angle = pose_det.det(frame_ori, bbox, frame)  # yaw pitch roll

        def get_normalized_pnt(nose, pnt, ):
            nose = np.asarray(nose).reshape(2, )
            pnt = np.asarray(pnt).reshape(2, )
            dir = pnt - nose
            norm = np.sqrt((dir ** 2).sum())
            if norm > 10000:
                print('pose norm is', norm)
                return True
            return False

        ## rule one: pose dir norm
        flag = False
        for imgpt in imgpts:
            flag = get_normalized_pnt(nose, imgpt)
        if flag:
            continue

        cv2.line(drawon, nose, tuple(imgpts[1, 0, :]), (0, 255, 0), 3)  # GREEN
        cv2.line(drawon, nose, tuple(imgpts[0, 0, :]), (255, 0, 0,), 3)  # BLUE
        cv2.line(drawon, nose, tuple(imgpts[2, 0, :]), (0, 0, 255), 3)  # RED
        remapping = [2, 3, 0, 4, 1]

        for index in range(len(landmarks) // 2):
            random_color = random_colors[index]
            cv2.circle(drawon, (landmarks[index * 2], landmarks[index * 2 + 1]), 5, random_color, -1)
            # cv2.circle(frame, tuple(modelpts[remapping[index]].ravel().astype(int)), 2, random_color, -1)

        for j in range(len(rotate_degree)):
            color = [(0, 255, 0), (255, 0, 0), (0, 0, 255)][j]
            cv2.putText(drawon,
                        ('{:05.2f}').format(float(rotate_degree[j])),
                        (10, 30 + 50 * j + 170 * num), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        color, thickness=2, lineType=2)
        if show:
            cvb.show_img(drawon, 'test', wait_time=1000 // 80 * 80)
        cvb.write_img(drawon, f"{dst}/proc/{imgfn}.png", )
        score = faces[num, 4]
        ## rule
        assert score >= 0.9
        bbox = faces[num, 0:4]
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        ## rule
        if min(width, height) < 20: continue
        rotate_degree = np.asarray(rotate_degree, int)
        rotate_degree = np.abs(rotate_degree)
        roll, pitch, yaw = rotate_degree
        ## rule
        if pitch > 70 or yaw > 70: continue
        print('roll pitch yaw ', roll, pitch, yaw)

        kps = faces[num, 5:].reshape(5, 2)
        warp_face = preprocess(frame, bbox=bbox, landmark=kps)
        fea, norm = extractor.extract_fea(warp_face)
        if norm <= 37: continue
        warp_faces.append(warp_face)

    return warp_faces


res = {}
for ind, imgfp in enumerate(v):
    frame = cvb.read_img(imgfp)
    imgfn = osp.basename(imgfp)
    frame = cvb.rgb2bgr(frame)
    # frame = np.rot90(frame, 3).copy()
    faces_infos, drawon = detect_face(frame, imgfn)
    faces_imgs, drawon = align_face(frame, imgfn, faces_infos, drawon)

lz.msgpack_dump(res, dst + 'kpts.pk')
