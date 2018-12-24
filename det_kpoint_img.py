import datetime
import torch, cv2
# sys.path.append('.')
from ssha_detector import SSHDetector
import lz
from lz import *

lz.init_dev((1,))
# lz.init_dev(get_dev())
gpuid = 0
import cvbase as cvb
import itertools
from insightface import FeaExtractor
import face_alignment

use_fan = True
if use_fan:
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device='cuda:' + str(gpuid))

extractor = FeaExtractor(
    yy_imgs=None, gpuid=gpuid,
)
print('face feature ex loaded ')
scales = [3456, 3456]  # 3456, 4608

import hopenet

norm_thresh = 22
min_face = 20
max_pith = 70
max_yaw = 70
det_score_thresh = .9
pose_norm = 10000

norm_thresh = 18
min_face = 20
max_pith = 45
max_yaw = 45
det_score_thresh = .9
pose_norm = 10000

show = False  # False
wait_time = 1000 * 10
detector = SSHDetector('./kmodel/e2e', 0, ctx_id=gpuid)
print('detector loader')

if show:
    cv2.namedWindow('test', cv2.WINDOW_NORMAL)

# src_dir = f'{work_path}/youeryuan/20180930 新大一班-林蝶老师-29、30/20180930 大一班9.30/9.30/'
src_dir ='/home/xinglu/work/youeryuan/20180930 新大一班-林蝶老师-29、30/9.29正、侧、背/'
# src_dir = '/home/xinglu/work/youeryuan/20180930 新大一班-林蝶老师-29、30/9.30正、侧、背/'
# src_dir = f'{work_path}/youeryuan/20180930 新大一班-林蝶老师-29、30/20180930 大一班9.29/9.29/'
assert osp.exists(src_dir), src_dir
vs = [glob.iglob(src_dir + f'/*.{suffix}', recursive=False) for suffix in get_img_suffix()]
vseq = itertools.chain(*vs)
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
    
    faces = detector.detect(img, threshold=0.9)  #   critetion 1
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


def align_face(frame, imgfn, faces, drawon, save=True, ):
    info = []
    frame = frame.copy()
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
            if norm > pose_norm:
                print('pose norm is', norm)
                return True
            return False
        
        ## rule one: pose dir norm
        fail_flag = False
        for imgpt in imgpts:
            fail_flag = get_normalized_pnt(nose, imgpt)
        
        if fail_flag:
            continue
        else:
            cv2.line(drawon, nose, tuple(imgpts[1, 0, :]), (0, 255, 0), 3)  # GREEN
            cv2.line(drawon, nose, tuple(imgpts[0, 0, :]), (255, 0, 0,), 3)  # BLUE
            cv2.line(drawon, nose, tuple(imgpts[2, 0, :]), (0, 0, 255), 3)  # RED
        
        for index in range(len(landmarks) // 2):
            random_color = random_colors[index]
            cv2.circle(drawon, (landmarks[index * 2], landmarks[index * 2 + 1]), 5, random_color, -1)

        face_crop2, _ = extend_bbox(drawon, bbox, .2, .2, .2)
        plt_imshow(face_crop2, 'bgr')
        plt.show()
        # plt_imshow(drawon)
        # plt.show()
        
        for j in range(len(rotate_degree)):
            color = [(0, 255, 0), (255, 0, 0), (0, 0, 255)][j]
            cv2.putText(drawon,
                        ('{:05.2f}').format(float(rotate_degree[j])),
                        (10, 30 + 50 * j + 170 * num), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        color, thickness=2, lineType=2)
        
        score = faces[num, 4]
        ## rule 1
        if score < 0.9:
            continue
        bbox = faces[num, 0:4]
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        ## rule2
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
        if norm <= norm_thresh:
            print('norm skip', norm, )
            continue
        res_faces.append(warp_face)
        info_['norm'] = norm
        info_['fea'] = fea
        info_['ind'] = ind
        ind += 1
        infonew.append(info_)
    return res_faces, infonew


# mkdir_p(dst + '/face')
# mkdir_p(dst + '/proc')

detect_meter = AverageMeter()
align_meter = AverageMeter()
norm_meter = AverageMeter()
timer.since_last_check('start ')

all_info = []
for ind_img, imgfp in enumerate(vseq):
    imgfp = '/home/xinglu/work/youeryuan/20180930 新大一班-林蝶老师-29、30/9.29正、侧、背/正/IMG_9551.JPG' # 9551
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
        cvb.write_img(face, f'{dst}/face/{imgfn}.{ind}.png')
    break
lz.msgpack_dump(all_info, f'{dst}/face/info.pk')
print('final how many imgs', ind_img,
      'detect ', detect_meter.avg,
      'align ', align_meter.avg,
      'norm ', norm_meter.avg)
