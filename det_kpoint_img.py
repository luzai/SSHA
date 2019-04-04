import datetime
import torch, cv2, sys
# sys.path.append('.')
from ssha_detector import SSHDetector
import lz
from lz import *
import cvbase as cvb
import itertools
from insightface import FeaExtractor
import face_alignment
from deep_pose import PoseDetector

# lz.init_dev((2,))
lz.init_dev(get_dev())
gpuid = 0

use_fan = True
if use_fan:
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device='cuda:' + str(gpuid))
fa.get_landmarks_from_image(np.random.rand(336, 336, 3))
extractor = FeaExtractor(
    yy_imgs=None, gpuid=gpuid,
)
print('face feature ex loaded ')
scales = [3456, 3456]  # 3456, 4608

pose_det = PoseDetector()

# if imgs are good
# norm_thresh = 42
# min_face = 20
# max_pith = 70
# max_yaw = 70
# det_score_thresh = .99
# pose_norm = 10000

# if we need to be strict
norm_thresh = 54
min_face = 20
max_pith = 45
max_yaw = 45
det_score_thresh = .99
pose_norm = 10000

show = False
show_face = False
wait_time = 1000 * 1
detector = SSHDetector('./kmodel/e2e', 0, ctx_id=gpuid)
logging.info('detector loader succ')

if show:
    cv2.namedWindow('test', cv2.WINDOW_NORMAL)

src_dir = '/home/xinglu/work/youeryuan/20180930 新大一班-林蝶老师-29、30/9.30正、侧、背/'
# src_dir = '/data1/share/youeryuan/20180930 新大一班-林蝶老师-29、30/20180930 大一班9.30/9.30/'
assert osp.exists(src_dir), src_dir
vs = [glob.iglob(src_dir + f'/**/*.{suffix}', recursive=True) for suffix in get_img_suffix()]
vseq = itertools.chain(*vs)
vseq = list(vseq)
vseq = [v for v in vseq if 'proc' not in v and 'face' not in v]
assert vseq, 'chk dir, empty?'
dst = src_dir


def detect_face(img, imgfn=None, save=False):
    frame = img.copy()
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
        frame = cv2.resize(frame, None, None, fx=im_scale, fy=im_scale)
    
    faces = detector.detect(img, threshold=det_score_thresh)  # critetion 1
    if faces.shape[0] != 0:
        for num in range(faces.shape[0]):
            score = faces[num, 4]
            assert score >= 0.9
            bbox = faces[num, 0:5]
            label_text = 'det {:.02f}'.format(bbox[4])
            cv2.putText(img, label_text, (int(bbox[0]),
                                          int(bbox[1] - 2)),
                        cv2.FONT_HERSHEY_COMPLEX, 1., (0, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            if not use_fan:
                kpoint = faces[num, 5:15]
                for knum in range(5):
                    cv2.circle(img, (kpoint[2 * knum], kpoint[2 * knum + 1]), 1, [0, 0, 255], 2)
            else:
                face_crop, bbox_bias = extend_bbox(frame, bbox, .3, .3, .3)
                face_crop = cvb.bgr2rgb(face_crop)
                lmks = fa.get_landmarks_from_image(face_crop)
                if lmks is None:
                    faces[num, 4] = 0.
                    continue
                lmks = lmks[0]
                lmks[:, 0] += bbox_bias[0]
                lmks[:, 1] += bbox_bias[1]
                for lmk in lmks:
                    cv2.circle(img, (lmk[0], lmk[1]), 1, [0, 0, 255], 2)
                # face_crop2, _ = extend_bbox(img, bbox, .2, .2, .2)
                # plt_imshow(face_crop2)
                # plt.show()
                kpoint = to_landmark5(lmks)
                kpoint = kpoint.flatten()
                faces[num, 5:15] = kpoint
        
        if imgfn is None:
            imgfn = randomword()
        if save:
            cvb.write_img(img, f"{dst}/proc/{imgfn}.png", )
        if show:
            cvb.show_img(img, 'test', wait_time=wait_time)
    lz.timer.since_last_check(f'detection on 1 img, find {faces.shape[0]} faces')
    if 'im_scale' in locals() and im_scale != 1:
        faces[:, :4] /= im_scale
        faces[:, 5:] /= im_scale
        img = cv2.resize(img, None, None, fx=1 / im_scale, fy=1 / im_scale)
    
    return faces, img


def get_normalized_pnt(nose, pnt, ):
    nose = np.asarray(nose).reshape(2, )
    pnt = np.asarray(pnt).reshape(2, )
    dir = pnt - nose
    norm = np.sqrt((dir ** 2).sum())
    if norm > pose_norm:
        print('pose norm is', norm)
        return True
    return False


def align_face(frame, imgfn, faces, drawon, save=True, ):
    info = []
    frame = frame.copy()
    warp_faces = []
    for num, landmarks in enumerate(faces[:, 5:]):
        bbox = faces[num, 0:5]
        imgpts, _, rotate_degree, nose = face_orientation(frame, landmarks)  # roll, pitch, yaw
        ## rule one: pose dir norm
        fail_flag = False
        for imgpt in imgpts:
            fail_flag = get_normalized_pnt(nose, imgpt)
        
        if fail_flag:
            img_pose, bbox_bias = extend_bbox(frame, bbox)
            yaw, pitch, roll = pose_det.det(img_pose, nose, drawon)
            print('3d pose by 2d landmark fail, ',
                  'roll, pitch, yaw ', rotate_degree,
                  'roll, pitch, yaw now ', (roll, pitch, yaw))
            rotate_degree = (roll, pitch, yaw)
        else:
            cv2.line(drawon, nose, tuple(imgpts[1, 0, :]), (0, 255, 0), 3)  # GREEN
            cv2.line(drawon, nose, tuple(imgpts[0, 0, :]), (255, 0, 0,), 3)  # BLUE
            cv2.line(drawon, nose, tuple(imgpts[2, 0, :]), (0, 0, 255), 3)  # RED
        
        for index in range(len(landmarks) // 2):
            random_color = random_colors[index]
            cv2.circle(drawon, (landmarks[index * 2], landmarks[index * 2 + 1]), 5, random_color, -1)
        
        # face_crop2, _ = extend_bbox(drawon, bbox, .2, .2, .2)
        # plt_imshow(face_crop2, 'bgr')
        # plt.show()
        # plt_imshow(drawon)
        # plt.show()
        
        for j in range(len(rotate_degree)):
            color = [(0, 255, 0), (255, 0, 0), (0, 0, 255)][j]
            cv2.putText(drawon,
                        ('{:05.2f}').format(float(rotate_degree[j])),
                        (10, 30 + 50 * j + 170 * num), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        color, thickness=2, lineType=2)
        
        score = faces[num, 4]
        ## rule detection score
        if score < det_score_thresh: continue
        bbox = faces[num, 0:4]
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        ## rule face size
        if min(width, height) < min_face: continue
        rotate_degree = np.asarray(rotate_degree, int)
        rotate_degree = np.abs(rotate_degree)
        roll, pitch, yaw = rotate_degree
        ## rule 3
        if pitch > max_pith or yaw > max_yaw: continue
        print('roll pitch yaw ', roll, pitch, yaw)
        
        kps = faces[num, 5:].reshape(5, 2)
        warp_face = preprocess(frame, bbox=bbox, landmark=kps)
        
            # plt_imshow(warp_face)
        # plt.show()
        
        warp_faces.append(warp_face)
        
        info.append({'kpt': kps, 'bbox': bbox, 'rpw': rotate_degree,
                     'imgfn': imgfn
                     })
    if show:
        cvb.show_img(drawon, 'test', wait_time=wait_time)
    cvb.write_img(drawon, f"{dst}/proc/{imgfn}.jpg", )
    return warp_faces, drawon, info


def norm_face(warp_faces, info):
    res_faces = []
    infonew = []
    ind = 0
    for warp_face, info_ in zip(warp_faces, info):
        fea, norm = extractor.extract_fea(warp_face)
        logging.info(f'face norm is {norm}')
        # rule: face norm
        if norm <= norm_thresh:
            continue
        if show_face:
            cvb.show_img(warp_face, 'test', wait_time=wait_time)
        res_faces.append(warp_face)
        info_['norm'] = norm
        info_['fea'] = fea
        info_['ind'] = ind
        ind += 1
        infonew.append(info_)
    return res_faces, infonew


mkdir_p(dst + '/face')
mkdir_p(dst + '/proc')

detect_meter = AverageMeter()
align_meter = AverageMeter()
norm_meter = AverageMeter()
timer.since_last_check('start ')

all_info = []
for ind_img, imgfp in enumerate(vseq):
    logging.info(f"--- {imgfp} ---")
    frame = cvb.read_img(imgfp)
    
    imgfn = osp.basename(imgfp)
    for suffix in get_img_suffix():
        imgfn = imgfn.replace(suffix, '')
    imgfn = imgfn.strip('.')
    # frame = np.rot90(frame, 3).copy()
    
    faces_infos, drawon = detect_face(frame, imgfn)
    detect_meter.update(timer.since_last_check(verbose=False))
    
    faces_imgs, drawon, info = align_face(frame, imgfn, faces_infos, drawon)
    align_meter.update(timer.since_last_check(verbose=False))
    
    faces_imgs, info = norm_face(faces_imgs, info)
    norm_meter.update(timer.since_last_check(verbose=False))
    all_info.extend(info)
    for ind, face in enumerate(faces_imgs):
        lz.mkdir_p(f'{dst}/face/', delete=False)
        cvb.write_img(face, f'{dst}/face/{imgfn}.{ind}.png')
lz.mkdir_p(f'{dst}/face/', delete=False)
lz.msgpack_dump(all_info, f'{dst}/face/info.pk')
print('final how many imgs', ind_img,
      'detect ', detect_meter.avg,
      'align ', align_meter.avg,
      'norm ', norm_meter.avg)
