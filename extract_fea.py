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

extractor = FeaExtractor(day=param_day)
succs = []

for ind, faces in res.items():
    # ind = list(res.keys())[50]
    # faces = res[ind]
    frame = v[ind]
    frame = frame if not (param_type == 'mov' and param_day == 'yy2') else np.rot90(frame, ).copy()
    # plt_imshow(np.rot90(frame, 0), )
    # plt.show()

    frame = cvb.rgb2bgr(frame)
    # print(faces.shape)
    # img = frame.copy()
    frame_ori = frame.copy()
    frame_ori2 = frame.copy()
    img = frame
    # img = cvb.read_img('test_image/test_2.jpg').copy()

    for num in range(faces.shape[0]):
        score = faces[num, 4]
        if score < 0.9: continue
        # print(score)
        bbox = faces[num, 0:5]
        label_text = 'det {:.02f}'.format(bbox[4])
        cv2.putText(img, label_text, (int(bbox[0]),
                                      int(bbox[1] - 2)),
                    cv2.FONT_HERSHEY_COMPLEX, 1., (0, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        # kpoint = faces[num, 5:15]
        # for knum in range(5):
        #     cv2.circle(img, (kpoint[2 * knum], kpoint[2 * knum + 1]), 1, [0, 0, 255], 2)

    # cvb.show_img(img, win_name='', wait_time=1000 // 40)
    # cvb.show_img(img, win_name='', )
    # cv2.imwrite("res.jpg".format(ind), img)

    flag = False
    for num, landmarks in enumerate(faces[:, 5:]):
        bbox = faces[num, 0:5]
        imgpts, modelpts, rotate_degree, nose = face_orientation(frame, landmarks)  # roll, pitch, yaw
        pose_angle = pose_det.det(frame_ori, bbox, frame)  # yaw pitch roll


        def get_normalized_pnt(nose, pnt):
            global flag
            nose = np.asarray(nose).reshape(2, )
            pnt = np.asarray(pnt).reshape(2, )
            dir = pnt - nose
            norm = np.sqrt((dir ** 2).sum())
            if norm > 10000:
                print('pose norm is', norm)
                flag = True
            pnt = tuple(np.asarray(pnt).ravel())
            # not normalize in fact
            return pnt


        for imgpt in imgpts:
            get_normalized_pnt(nose, imgpt)
        if flag:
            break

        cv2.line(frame, nose, get_normalized_pnt(nose, imgpts[1]), (0, 255, 0), 3)  # GREEN
        cv2.line(frame, nose, get_normalized_pnt(nose, imgpts[0]), (255, 0, 0,), 3)  # BLUE
        cv2.line(frame, nose, get_normalized_pnt(nose, imgpts[2]), (0, 0, 255), 3)  # RED
        remapping = [2, 3, 0, 4, 1]

        for index in range(len(landmarks) // 2):
            random_color = random_colors[index]

            cv2.circle(frame, (landmarks[index * 2], landmarks[index * 2 + 1]), 5, random_color, -1)
            # cv2.circle(frame, tuple(modelpts[remapping[index]].ravel().astype(int)), 2, random_color, -1)

        for j in range(len(rotate_degree)):
            color = [(0, 255, 0), (255, 0, 0), (0, 0, 255)][j]
            cv2.putText(frame,
                        ('{:05.2f}').format(float(rotate_degree[j])),
                        (10, 30 + 50 * j + 170 * num), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        color, thickness=2, lineType=2)
        score = faces[num, 4]
        if score < 0.9: continue
        bbox = faces[num, 0:4]
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if min(width, height) < 15: continue
        rotate_degree = np.asarray(rotate_degree, int)
        rotate_degree = np.abs(rotate_degree)
        roll, pitch, yaw = rotate_degree
        if pitch > 15 or yaw > 15: continue
        print('roll pitch yaw ', roll, pitch, yaw)
        # crop_face = cvb.crop_img(frame_ori, bbox)
        # crop_face = cvb.crop_img(frame, bbox)
        # cvb.show_img(crop_face, wait_time=1000 // 20)

        kps = faces[num, 5:].reshape(5, 2)
        warp_face = preprocess(frame_ori, bbox=bbox, landmark=kps)
        # cvb.show_img(warp_face, wait_time=1000 // 20)
        # cvb.show_img(warp_face, )
        sim, norm = extractor.compare(warp_face, return_norm=True)
        ind_gallery = sim.argmax()
        # if sim[0, ind_gallery] > 0.1 and norm > 2.:
        if norm > 3.5 and sim[0, ind_gallery] > 0.5:
            print('!! sim is', sim[0, ind_gallery], 'norm ', norm)
            img_gallery = list(extractor.yy_imgs.values())[ind_gallery]
            img_gallery_norm = list(extractor.yy_feas_norms.values())[ind_gallery]
            cv2.putText(img, f'id {int(ind_gallery)}', (int(bbox[0]),
                                                        int(bbox[1] - 2 - 20)),
                        cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA)

            if len(succs) == 0 or ind_gallery != succs[-1][-1]:
                succs.append([img_gallery, warp_face, sim, norm, ind_gallery])

        # out_fn = f'face.{param_day}/gallery/{param_type}_{ind}_{num}.png'
        # assert not osp.exists(out_fn)
        # cvb.write_img(warp_face, out_fn)
    if flag:
        continue
    max_succ = frame.shape[1] // 224
    while len(succs) > max_succ:
        succs.pop(0)
    for ind, (img_gallery, warp_face, sim, norm, ind_gallery) in enumerate(succs[::-1]):
        frame[- 112:,  # row
        - (112 + 112 * ind * 2 + 10 * ind + 1):-(112 * ind * 2 + 10 * ind + 1),  # col
        :] = img_gallery

        frame[-112:,
        -(112 + 112 * (ind * 2 + 1) + 10 * ind + 1): -(112 * (ind * 2 + 1) + 10 * ind + 1),
        :] = warp_face

        cv2.putText(frame, f'sim {sim[0, ind_gallery]:.2f}',
                    (frame.shape[1] - 112 - 112 * (ind * 2 + 1),  # col
                     frame.shape[0] - 112),  # row
                    cv2.FONT_HERSHEY_COMPLEX, .9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'norm {norm:.2f}',
                    (frame.shape[1] - 112 - 112 * (ind * 2 + 1),  # col
                     frame.shape[0] - 112 - 112 // 4),  # row
                    cv2.FONT_HERSHEY_COMPLEX, .9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'ind {int(ind_gallery)}',
                    (frame.shape[1] - 112 - 112 * (ind * 2 + 1),  # col
                     frame.shape[0] - 112 - 112 // 4 * 2),  # row
                    cv2.FONT_HERSHEY_COMPLEX, .6, (0, 0, 255), 2, cv2.LINE_AA)

    cvb.show_img(frame, win_name='test', wait_time=1000 // 80)
    video_writer.write(frame)

video_writer.release()
