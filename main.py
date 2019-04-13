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
logging.info('face feature ex loaded ')
scales = [3456, 3456]  # 3456, 4608

# if imgs are good
NORM_THRESH_GOOD = 42
# if we need to be strict
NORM_THRESH_STRICT = 54
MIN_FACE = 20
DET_SCORE_THRESH = .95
SIM_THRESH = 0.61

show = False
show_face = False
wait_time = 1000 * 1
detector = SSHDetector('./kmodel/e2e', 0, ctx_id=gpuid)
logging.info('detector loader succ')

if show:
    cv2.namedWindow('test', cv2.WINDOW_NORMAL)

src_dir_gallery = '/home/xinglu/work/youeryuan/20180930 新大一班-林蝶老师-29、30/30.正.named/'
src_dir = '/data1/share/youeryuan/20180930 新大一班-林蝶老师-29、30/20180930 大一班9.30/9.30/'


def get_vseq(src_dir):
    assert osp.exists(src_dir), src_dir
    vs = [glob.iglob(src_dir + f'/**/*.{suffix}', recursive=True) for suffix in get_img_suffix()]
    vseq = itertools.chain(*vs)
    vseq = list(vseq)
    vseq = [v for v in vseq if 'proc' not in v and 'face' not in v]
    assert vseq, 'chk dir, empty?'
    return vseq


def detect_face(img, imgfn=None, save=False, dst='/tmp', det_score_thresh=DET_SCORE_THRESH):
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


def align_face(frame, imgfn, faces, drawon, dst='/tmp', min_face=MIN_FACE, det_score_thresh=DET_SCORE_THRESH):
    info = []
    frame = frame.copy()
    warp_faces = []
    for num, landmarks in enumerate(faces[:, 5:]):
        for index in range(len(landmarks) // 2):
            random_color = random_colors[index]
            cv2.circle(drawon, (landmarks[index * 2], landmarks[index * 2 + 1]), 5, random_color, -1)
        
        score = faces[num, 4]
        ## rule detection score
        logging.info(f'det score is {score}')
        if score < det_score_thresh:
            logging.info('skip because det score small')
            continue
        bbox = faces[num, 0:4]
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        ## rule face size
        if min(width, height) < min_face: continue
        
        kps = faces[num, 5:].reshape(5, 2)
        warp_face = preprocess(frame, bbox=bbox, landmark=kps)
        # plt_imshow(warp_face)
        # plt.show()
        warp_faces.append(warp_face)
        info.append({'kpt': kps, 'bbox': bbox,
                     'imgfn': imgfn
                     })
    if show:
        cvb.show_img(drawon, 'test', wait_time=wait_time)
    cvb.write_img(drawon, f"{dst}/proc/{imgfn}.jpg", )
    return warp_faces, drawon, info


def norm_face(warp_faces, info, norm_thresh):
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
        info_['warp_face'] = warp_face
        
        ind += 1
        infonew.append(info_)
    return res_faces, infonew


def extract_face_and_feature(src_dir, norm_thresh):
    vseq = get_vseq(src_dir)
    mkdir_p(src_dir + '/face', delete=False)
    mkdir_p(src_dir + '/proc', delete=False)
    all_info = []
    detect_meter = AverageMeter()
    align_meter = AverageMeter()
    norm_meter = AverageMeter()
    timer.since_last_check('start ')
    for ind_img, imgfp in enumerate(vseq):
        logging.info(f"--- {imgfp} ---")
        frame = cvb.read_img(imgfp)
        
        imgfn = osp.basename(imgfp)
        for suffix in get_img_suffix():
            imgfn = imgfn.replace(suffix, '')
        imgfn = imgfn.strip('.')
        
        faces_infos, drawon = detect_face(frame, imgfn, dst=src_dir)
        detect_meter.update(timer.since_last_check(verbose=False))
        
        faces_imgs, drawon, info = align_face(frame, imgfn, faces_infos, drawon, dst=src_dir)
        align_meter.update(timer.since_last_check(verbose=False))
        
        faces_imgs, info = norm_face(faces_imgs, info, norm_thresh)
        norm_meter.update(timer.since_last_check(verbose=False))
        all_info.extend(info)
        for ind, face in enumerate(faces_imgs):
            lz.mkdir_p(f'{src_dir}/face/', delete=False)
            cvb.write_img(face, f'{src_dir}/face/{imgfn}.{ind}.png')
            all_info[ind]['imgfn_from'] = osp.basename(imgfp)
            all_info[ind]['imgfn_face_to'] = f'{imgfn}.{ind}.png'
    lz.mkdir_p(f'{src_dir}/face/', delete=False)
    lz.msgpack_dump(all_info, f'{src_dir}/face/info.pk')
    
    print('final how many imgs', ind_img,
          'detect ', detect_meter.avg,
          'align ', align_meter.avg,
          'norm ', norm_meter.avg)


def info2face_imgfn(info_, src_dir):
    imgfn = info_['imgfn']
    ind = info_['ind']
    imgfn = glob.glob(f'{src_dir}/face/{imgfn}.{ind}*')
    imgfn = list(imgfn)
    assert len(imgfn) >= 1
    imgfn = imgfn[0]
    return imgfn


def info2face(info_, src_dir):
    imgfn = info2face_imgfn(info_, src_dir)
    img = cvb.read_img(imgfn)
    img = cvb.rgb2bgr(img)
    assert img is not None
    return img


def match_face(src_dir_gallery, src_dir):
    info_gallery = msgpack_load(src_dir_gallery + '/face/info.pk')
    info = msgpack_load(src_dir + '/face/info.pk')
    faces = []
    feas = []
    
    for info_ in info_gallery:
        img = info2face(info_, src_dir_gallery)
        feas.append(info_['fea'])
        faces.append(img)
    faces = np.array(faces)
    feas = np.asarray(feas)
    # plt_imshow_tensor(faces)
    
    from scipy.spatial.distance import cdist
    for info_ in info:
        fea = info_['fea'].reshape(1, -1)
        dists = cdist(fea, feas)
        ind_gallery = dists.argmin()
        face_gallery = faces[ind_gallery]
        face_probe = info2face(info_, src_dir)
        #     plt_imshow_tensor([face_gallery, face_probe])
        info_['ind_gallery'] = ind_gallery
        info_['gallery_img'] = face_gallery
        info_['sim'] = (2 - dists[0, ind_gallery]) / 2
        #     break
    
    msgpack_dump(info, src_dir + '/face/info.pk')


def clean_small_similarity(src_dir):
    info = msgpack_load(src_dir + '/face/info.pk')
    info_res = []
    for info_ in info:
        if info_['sim'] > SIM_THRESH and info_['norm'] > NORM_THRESH_STRICT:
            info_res.append(info_)
        else:
            imgfn = info2face_imgfn(info_, src_dir)
            _ = rm(imgfn)
    msgpack_dump(info_res, src_dir + '/face/info.pk')


extract_face_and_feature(src_dir_gallery, NORM_THRESH_GOOD)
extract_face_and_feature(src_dir, NORM_THRESH_STRICT)

match_face(src_dir_gallery, src_dir)
clean_small_similarity(src_dir)
