# -*- coding: future_fstrings -*-
import lz
from lz import *
from insightface.model import *
from insightface import model
import numpy as np
from insightface.verifacation import evaluate
from torch import optim
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
from insightface.utils import get_time, gen_plot, hflip_batch, separate_bn_paras, hflip
from PIL import Image
from torchvision import transforms as trans
import os, random, logging, numbers, math
from pathlib import Path
from torch.utils.data import DataLoader
from insightface.config import conf as gl_conf
import torch.autograd
from torch import nn
import torch.multiprocessing as mp
from torch.utils.data.sampler import Sampler
from torch.nn import functional as F

try:
    import mxnet as mx
    from mxnet import ndarray as nd
    from mxnet import recordio
except ImportError:
    logging.warning('if want to train, install mxnet for read rec data')
    gl_conf.training = False
try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp
    
    if gl_conf.fp16:
        # amp.register_half_function(torch, 'prelu')
        # amp.register_promote_function(torch, 'prelu')
        amp.register_float_function(torch, 'prelu')
        # amp.register_half_function(torch, 'pow')
        # amp.register_half_function(torch, 'norm') # dangerous
        pass
    amp_handle = amp.init(enabled=gl_conf.fp16)
except ImportError:
    logging.warning("if want to use fp16, install apex from https://www.github.com/nvidia/apex to run this example.")
    gl_conf.fp16 = False


def unpack_f64(s):
    from mxnet.recordio import IRHeader, _IR_FORMAT, _IR_SIZE, struct
    header = IRHeader(*struct.unpack(_IR_FORMAT, s[:_IR_SIZE]))
    s = s[_IR_SIZE:]
    if header.flag > 0:
        header = header._replace(label=np.frombuffer(s, np.float64, header.flag))
        s = s[header.flag * 8:]
    return header, s


def unpack_f32(s):
    from mxnet.recordio import IRHeader, _IR_FORMAT, _IR_SIZE, struct
    header = IRHeader(*struct.unpack(_IR_FORMAT, s[:_IR_SIZE]))
    s = s[_IR_SIZE:]
    if header.flag > 0:
        header = header._replace(label=np.frombuffer(s, np.float32, header.flag))
        s = s[header.flag * 4:]
    return header, s


def unpack_auto(s, fp):
    if 'f64' not in fp and 'alpha' not in fp:
        return unpack_f32(s)
    else:
        return unpack_f64(s)


# todo merge this
class Obj():
    pass


def get_rec(p='/data2/share/faces_emore/train.idx'):
    self = Obj()
    self.imgrecs = []
    path_imgidx = p
    path_imgrec = path_imgidx.replace('.idx', '.rec')
    self.imgrecs.append(
        recordio.MXIndexedRecordIO(
            path_imgidx, path_imgrec,
            'r')
    )
    self.lock = mp.Lock()
    self.imgrec = self.imgrecs[0]
    s = self.imgrec.read_idx(0)
    header, _ = recordio.unpack(s)
    assert header.flag > 0, 'ms1m or glint ...'
    print('header0 label', header.label)
    self.header0 = (int(header.label[0]), int(header.label[1]))
    self.id2range = {}
    self.imgidx = []
    self.ids = []
    ids_shif = int(header.label[0])
    for identity in list(range(int(header.label[0]), int(header.label[1]))):
        s = self.imgrecs[0].read_idx(identity)
        header, _ = recordio.unpack(s)
        a, b = int(header.label[0]), int(header.label[1])
        #     if b - a > gl_conf.cutoff:
        self.id2range[identity] = (a, b)
        self.ids.append(identity)
        self.imgidx += list(range(a, b))
    self.ids = np.asarray(self.ids)
    self.num_classes = len(self.ids)
    # self.ids_map = {identity - ids_shif: id2 for identity, id2 in zip(self.ids, range(self.num_classes))}
    ids_map_tmp = {identity: id2 for identity, id2 in zip(self.ids, range(self.num_classes))}
    self.ids = [ids_map_tmp[id_] for id_ in self.ids]
    self.ids = np.asarray(self.ids)
    self.id2range = {ids_map_tmp[id_]: range_ for id_, range_ in self.id2range.items()}
    return self


class DatasetIJBC2(torch.utils.data.Dataset):
    def __init__(self, ):
        self.test_transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        IJBC_path = '/data1/share/IJB_release/' if 'amax' in lz.hostname() else '/home/zl/zl_data/IJB_release/'
        img_list_path = IJBC_path + './IJBC/meta/ijbc_name_5pts_score.txt'
        img_list = open(img_list_path)
        files = img_list.readlines()
        img_path = '/share/data/loose_crop'
        if not osp.exists(img_path):
            img_path = IJBC_path + './IJBC/loose_crop'
        self.img_path = img_path
        self.IJBC_path = IJBC_path
        self.files = files
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, item):
        import cvbase as cvb
        from PIL import Image
        img_index = item
        each_line = self.files[img_index]
        name_lmk_score = each_line.strip().split(' ')
        img_name = os.path.join(self.img_path, name_lmk_score[0])
        img = cvb.read_img(img_name)
        img = cvb.bgr2rgb(img)  # this is RGB
        assert img is not None, img_name
        lmk = np.array([float(x) for x in name_lmk_score[1:-1]], dtype=np.float32)
        lmk = lmk.reshape((5, 2))
        warp_img = lz.preprocess(img, landmark=lmk)
        warp_img = Image.fromarray(warp_img)
        img = self.test_transform(warp_img)
        return img


class MegaFaceDisDS(torch.utils.data.Dataset):
    def __init__(self):
        megaface_path = '/data1/share/megaface/'
        # files_scrub = open(f'{megaface_path}/facescrub_lst') .readlines()
        # files_scrub = [f'{megaface_path}/facescrub_images/{f}' for f in files_scrub]
        files_scrub = []
        files_dis = open('f{megaface_path}/megaface_lst').readlines()
        files_dis = [f'{megaface_path}/facescrub_images/{f}' for f in files_dis]
        self.files = files_dis + files_scrub
        self.test_transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, item):
        import cvbase as cvb
        img = cvb.read_img(self.files[item])
        img = cvb.bgr2rgb(img)  # this is RGB
        img = self.test_transform(img)
        return img


class TestDataset(object):
    def __init__(self):
        assert gl_conf.use_test
        if gl_conf.use_test == 'ijbc':
            self.rec_test = DatasetIJBC2()
            self.imglen = len(self.rec_test)
        elif gl_conf.use_test == 'glint':
            self.rec_test = get_rec('/data2/share/glint_test/train.idx')
            self.imglen = max(self.rec_test.imgidx) + 1
        else:
            raise ValueError(f'{gl_conf.use_test}')
    
    def _get_single_item(self, index):
        if gl_conf.use_test == 'ijbc':
            imgs = self.rec_test[index]
        elif gl_conf.use_test == 'glint':
            self.rec_test.lock.acquire()
            s = self.rec_test.imgrec.read_idx(index)
            self.rec_test.lock.release()
            header, img = unpack_auto(s, 'glint_test')
            imgs = self.imdecode(img)
            imgs = self.preprocess_img(imgs)
        else:
            raise ValueError(f'{gl_conf.use_test}')
        return {'imgs': np.array(imgs, dtype=np.float32), 'labels': -1, 'indexes': index,
                'ind_inds': -1, 'is_trains': False}
    
    def __len__(self):
        return self.imglen
    
    def __getitem__(self, indices, ):
        res = self._get_single_item(indices)
        return res


class TorchDataset(object):
    def __init__(self,
                 path_ms1m,
                 ):
        self.flip = gl_conf.flip
        self.path_ms1m = path_ms1m
        self.root_path = Path(path_ms1m)
        path_imgrec = str(path_ms1m) + '/train.rec'
        path_imgidx = path_imgrec[0:-4] + ".idx"
        assert os.path.exists(path_imgidx), path_imgidx
        self.path_imgidx = path_imgidx
        self.path_imgrec = path_imgrec
        self.imgrecs = []
        self.locks = []
        
        if gl_conf.use_redis:
            import redis
            
            self.r = redis.Redis()
        else:
            self.r = None
        
        lz.timer.since_last_check('start timer for imgrec')
        for num_rec in range(gl_conf.num_recs):
            if num_rec == 1:
                path_imgrec = path_imgrec.replace('/data2/share/', '/share/data/')
            self.imgrecs.append(
                recordio.MXIndexedRecordIO(
                    path_imgidx, path_imgrec,
                    'r')
            )
            self.locks.append(mp.Lock())
        lz.timer.since_last_check(f'{gl_conf.num_recs} imgrec readers init')  # 27 s / 5 reader
        lz.timer.since_last_check('start cal dataset info')
        # try:
        #     self.imgidx, self.ids, self.id2range = lz.msgpack_load(str(path_ms1m) + f'/info.{gl_conf.cutoff}.pk')
        #     self.num_classes = len(self.ids)
        # except:
        s = self.imgrecs[0].read_idx(0)
        header, _ = unpack_auto(s, self.path_imgidx)
        assert header.flag > 0, 'ms1m or glint ...'
        logging.info(f'header0 label {header.label}')
        self.header0 = (int(header.label[0]), int(header.label[1]))
        self.id2range = {}
        self.imgidx = []
        self.ids = []
        ids_shif = int(header.label[0])
        for identity in list(range(int(header.label[0]), int(header.label[1]))):
            s = self.imgrecs[0].read_idx(identity)
            header, _ = unpack_auto(s, self.path_imgidx)
            a, b = int(header.label[0]), int(header.label[1])
            if b - a > gl_conf.cutoff:
                self.id2range[identity] = (a, b)
                self.ids.append(identity)
                self.imgidx += list(range(a, b))
        self.ids = np.asarray(self.ids)
        self.num_classes = len(self.ids)
        self.ids_map = {identity - ids_shif: id2 for identity, id2 in zip(self.ids, range(self.num_classes))}
        ids_map_tmp = {identity: id2 for identity, id2 in zip(self.ids, range(self.num_classes))}
        self.ids = [ids_map_tmp[id_] for id_ in self.ids]
        self.ids = np.asarray(self.ids)
        self.id2range = {ids_map_tmp[id_]: range_ for id_, range_ in self.id2range.items()}
        lz.msgpack_dump([self.imgidx, self.ids, self.id2range], str(path_ms1m) + f'/info.{gl_conf.cutoff}.pk')
        
        gl_conf.num_clss = self.num_classes
        gl_conf.explored = np.zeros(self.ids.max() + 1, dtype=int)
        if gl_conf.dop is None:
            if gl_conf.mining == 'dop':
                gl_conf.dop = np.ones(self.ids.max() + 1, dtype=int) * gl_conf.mining_init
                gl_conf.id2range_dop = {str(id_):
                                            np.ones((range_[1] - range_[0],)) *
                                            gl_conf.mining_init for id_, range_ in
                                        self.id2range.items()}
            elif gl_conf.mining == 'imp' or gl_conf.mining == 'rand.id':
                gl_conf.id2range_dop = {str(id_):
                                            np.ones((range_[1] - range_[0],)) *
                                            gl_conf.mining_init for id_, range_ in
                                        self.id2range.items()}
                gl_conf.dop = np.asarray([v.sum() for v in gl_conf.id2range_dop.values()])
        logging.info(f'update num_clss {gl_conf.num_clss} ')
        self.cur = 0
        lz.timer.since_last_check('finish cal dataset info')
        if gl_conf.kd and gl_conf.sftlbl_from_file:  # todo deprecated
            self.teacher_embedding_db = lz.Database('work_space/teacher_embedding.h5', 'r')
    
    def __len__(self):
        if gl_conf.local_rank is not None:
            return len(self.imgidx) // torch.distributed.get_world_size()
        else:
            #             logging.info(f'ask me len {len(self.imgidx)} {self.imgidx}')
            return len(self.imgidx) // gl_conf.epoch_less_iter
    
    def __getitem__(self, indices, ):
        # if isinstance(indices, (tuple, list)) and len(indices[0])==3:
        #    if self.r:
        #        pass
        #    else:
        #        return [self._get_single_item(index) for index in indices]
        res = self._get_single_item(indices)
        # for k, v in res.items():
        #     assert (
        #             isinstance(v, np.ndarray) or
        #             isinstance(v, str) or
        #             isinstance(v, int) or
        #             isinstance(v, np.int64) or
        #             torch.is_tensor(v)
        #     ), type(v)
        return res
    
    def imdecode(self, s):
        """Decodes a string or byte string to an NDArray.
        See mx.img.imdecode for more details."""
        img = mx.image.imdecode(s)  # mx.ndarray
        return img
    
    def postprocess_data(self, datum):
        """Final postprocessing step before image is loaded into the batch."""
        return nd.transpose(datum, axes=(2, 0, 1))
    
    def preprocess_img(self, imgs):
        if self.flip and random.randint(0, 1) == 1:
            imgs = mx.ndarray.flip(data=imgs, axis=1)
        imgs = imgs.asnumpy()
        if not gl_conf.fast_load:
            imgs = imgs / 255.
            imgs -= 0.5  # simply use 0.5 as mean
            imgs /= 0.5
            imgs = np.array(imgs, dtype=np.float32)
        imgs = imgs.transpose((2, 0, 1))
        return imgs
    
    def _get_single_item(self, index):
        # self.cur += 1
        # index += 1  # noneed,  here it index (imgidx) start from 1,.rec start from 1
        # assert index != 0 and index < len(self) + 1 # index can > len(self)
        succ = False
        index, pid, ind_ind = index
        if self.r:
            img = self.r.get(f'{gl_conf.dataset_name}/imgs/{index}')
            if img is not None:
                imgs = self.imdecode(img)
                imgs = self.preprocess_img(imgs)
                return {'imgs': np.array(imgs, dtype=np.float32), 'labels': pid, 'indexes': index,
                        'ind_inds': ind_ind, 'is_trains': True}
        ## rand until lock
        while True:
            for ind_rec in range(len(self.locks)):
                succ = self.locks[ind_rec].acquire(timeout=0)
                if succ: break
            if succ: break
        
        ##  locality based
        # if index < self.imgidx[len(self.imgidx) // 2]:
        #     ind_rec = 0
        # else:
        #     ind_rec = 1
        # succ = self.locks[ind_rec].acquire()
        
        s = self.imgrecs[ind_rec].read_idx(index)  # from [ 1 to 3804846 ]
        rls_succ = self.locks[ind_rec].release()
        header, img = unpack_auto(s, self.path_imgidx)  # this is RGB format
        imgs = self.imdecode(img)
        assert imgs is not None
        # label = header.label
        # if not isinstance(label, numbers.Number):
        #     assert label[-1] == 0. or label[-1] == 1., f'{label} {index} {imgs.shape}'
        #     label = label[0]
        # label = int(label)
        # assert label in self.ids_map
        # label = self.ids_map[label]
        # assert label == pid
        label = int(pid)
        imgs = self.preprocess_img(imgs)
        if gl_conf.use_redis and self.r and lz.get_mem() >= 20:
            self.r.set(f'{gl_conf.dataset_name}/imgs/{index}', img)
        res = {'imgs': imgs, 'labels': label,
               'ind_inds': ind_ind, 'indexes': index,
               'is_trains': True}
        if hasattr(self, 'teacher_embedding_db'):
            res['teacher_embedding'] = self.teacher_embedding_db[str(index)]
        return res


class Dataset_val(torch.utils.data.Dataset):
    def __init__(self, path, name, transform=None):
        from insightface.data.data_pipe import get_val_pair
        self.carray, self.issame = get_val_pair(path, name)
        self.carray = self.carray[:, ::-1, :, :]  # BGR 2 RGB!
        self.transform = transform
    
    def __getitem__(self, index):
        if self.transform:
            fliped_carray = self.transform(torch.tensor(self.carray[index]))
            return {'carray': self.carray[index], 'issame': 1.0 * self.issame[index], 'fliped_carray': fliped_carray}
        else:
            return {'carray': self.carray[index], 'issame': 1.0 * self.issame[index]}
    
    def __len__(self):
        return len(self.issame)


# improve locality and improve load speed!
class RandomIdSampler(Sampler):
    def __init__(self, imgidx, ids, id2range):
        path_ms1m = gl_conf.use_data_folder
        self.imgidx, self.ids, self.id2range = imgidx, ids, id2range
        # self.imgidx, self.ids, self.id2range = lz.msgpack_load(str(path_ms1m) + f'/info.{gl_conf.cutoff}.pk')
        # above is the imgidx of .rec file
        # remember -1 to convert to pytorch imgidx
        self.num_instances = gl_conf.instances
        self.batch_size = gl_conf.batch_size
        if gl_conf.tri_wei != 0:
            assert self.batch_size % self.num_instances == 0
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = {id: (np.asarray(list(range(idxs[0], idxs[1])))).tolist()
                          for id, idxs in self.id2range.items()}  # it index based on 1
        self.ids = list(self.ids)
        self.nimgs = np.asarray([
            range_[1] - range_[0] for id_, range_ in self.id2range.items()
        ])
        self.nimgs_normed = self.nimgs / self.nimgs.sum()
        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.ids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances
    
    def __len__(self):
        if gl_conf.local_rank is not None:
            return self.length // torch.distributed.get_world_size()
        else:
            return self.length
    
    def get_batch_ids(self):
        pids = []
        dop = gl_conf.dop
        if gl_conf.mining == 'imp' or gl_conf.mining == 'rand.id':
            # lz.logging.info(f'dop smapler {np.count_nonzero( dop == gl_conf.mining_init)} {dop}')
            pids = np.random.choice(self.ids,
                                    size=int(self.num_pids_per_batch),
                                    # p=gl_conf.dop / gl_conf.dop.sum(), # comment to balance
                                    replace=False)
        # todo dop with no replacement
        elif gl_conf.mining == 'dop':
            # lz.logging.info(f'dop smapler {np.count_nonzero( dop ==-1)} {dop}')
            nrand_ids = int(self.num_pids_per_batch * gl_conf.rand_ratio)
            pids_now = np.random.choice(self.ids,
                                        size=nrand_ids,
                                        replace=False)
            pids.append(pids_now)
            for _ in range(int(1 / gl_conf.rand_ratio) - 1):
                pids_next = dop[pids_now]
                pids_next[pids_next == -1] = np.random.choice(self.ids,
                                                              size=len(pids_next[pids_next == -1]),
                                                              replace=False)
                pids.append(pids_next)
                pids_now = pids_next
            pids = np.concatenate(pids)
            pids = np.unique(pids)
            if len(pids) < self.num_pids_per_batch:
                pids_now = np.random.choice(np.setdiff1d(self.ids, pids),
                                            size=self.num_pids_per_batch - len(pids),
                                            replace=False)
                pids = np.concatenate((pids, pids_now))
            else:
                pids = pids[: self.num_pids_per_batch]
        # assert len(pids) == np.unique(pids).shape[0]
        
        return pids
    
    def get_batch_idxs(self):
        pids = self.get_batch_ids()
        cnt = 0
        for pid in pids:
            if len(self.index_dic[pid]) < self.num_instances:
                replace = True
            else:
                replace = False
            if gl_conf.mining == 'imp':
                assert len(self.index_dic[pid]) == gl_conf.id2range_dop[str(pid)].shape[0]
                ind_inds = np.random.choice(
                    len(self.index_dic[pid]),
                    size=(self.num_instances,), replace=replace,
                    p=gl_conf.id2range_dop[str(pid)] / gl_conf.id2range_dop[str(pid)].sum()
                )
            else:
                ind_inds = np.random.choice(
                    len(self.index_dic[pid]),
                    size=(self.num_instances,), replace=replace, )
            if gl_conf.chs_first:
                if gl_conf.dataset_name == 'alpha_jk':
                    ind_inds = np.concatenate(([0], ind_inds))
                    ind_inds = np.unique(ind_inds)[:self.num_instances]  # 0 must be chsn
                elif gl_conf.dataset_name == 'alpha_f64':
                    if pid >= 112145:  # 112145 开始是证件-监控
                        ind_inds = np.concatenate(([0], ind_inds))
                        ind_inds = np.unique(ind_inds)[:self.num_instances]  # 0 must be chsn
            for ind_ind in ind_inds:
                ind = self.index_dic[pid][ind_ind]
                yield ind, pid, ind_ind
                cnt += 1
                if cnt == self.batch_size:
                    break
            if cnt == self.batch_size:
                break
    
    def __iter__(self):
        if gl_conf.mining == 'rand.img':  # quite slow
            for _ in range(len(self)):
                # lz.timer.since_last_check('s next id iter')
                pid = np.random.choice(
                    self.ids, p=self.nimgs_normed,
                )
                ind_ind = np.random.choice(
                    range(len(self.index_dic[pid])),
                )
                ind = self.index_dic[pid][ind_ind]
                # lz.timer.since_last_check('f next id iter')
                yield ind, pid, ind_ind
        else:
            cnt = 0
            while cnt < len(self):
                # logging.info(f'cnt {cnt}')
                for ind, pid, ind_ind in self.get_batch_idxs():
                    cnt += 1
                    yield (ind, pid, ind_ind)


class SeqSampler(Sampler):
    def __init__(self):
        path_ms1m = gl_conf.use_data_folder
        _, self.ids, self.id2range = lz.msgpack_load(path_ms1m / f'info.{gl_conf.cutoff}.pk')
        # above is the imgidx of .rec file
        # remember -1 to convert to pytorch imgidx
        self.num_instances = gl_conf.instances
        self.batch_size = gl_conf.batch_size
        assert self.batch_size % self.num_instances == 0
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = {id: (np.asarray(list(range(idxs[0], idxs[1])))).tolist()
                          for id, idxs in self.id2range.items()}  # it index based on 1
        self.ids = list(self.ids)
        
        # estimate number of examples in an epoch
        self.nimgs_lst = [len(idxs) for idxs in self.index_dic.values()]
        self.length = sum(self.nimgs_lst)
    
    def __len__(self):
        return self.length
    
    def __iter__(self):
        for pid in self.id2range:
            for ind_ind in range(len(self.index_dic[pid])):
                ind = self.index_dic[pid][ind_ind]
                yield ind, pid, ind_ind


def update_dop_cls(thetas, labels, dop):
    with torch.no_grad():
        bs = thetas.shape[0]
        # logging.info(f'min is {thetas.min()}')
        thetas[torch.arange(0, bs, dtype=torch.long), labels] = -1e4
        dop[labels.cpu().numpy()] = torch.argmax(thetas, dim=1).cpu().numpy()


class FaceInfer():
    def __init__(self, conf, gpuid=(0,)):
        if conf.net_mode == 'mobilefacenet':
            self.model = MobileFaceNet(conf.embedding_size)
            print('MobileFaceNet model generated')
        elif conf.net_mode == 'ir_se' or conf.net_mode == 'ir':
            self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode)
        else:
            raise ValueError(conf.net_mode)
        self.model = self.model.eval()
        self.model = torch.nn.DataParallel(self.model,
                                           device_ids=list(gpuid), output_device=gpuid[0]).to(gpuid[0])
    
    def load_model_only(self, fpath):
        model_state_dict = torch.load(fpath, map_location=lambda storage, loc: storage)
        model_state_dict = {k: v for k, v in model_state_dict.items() if 'num_batches_tracked' not in k}
        if list(model_state_dict.keys())[0].startswith('module'):
            self.model.load_state_dict(model_state_dict, strict=True, )
        else:
            self.model.module.load_state_dict(model_state_dict, strict=True, )
    
    def load_state(self, fixed_str=None,
                   resume_path=None, latest=True,
                   ):
        from pathlib import Path
        save_path = Path(resume_path)
        modelp = save_path / '{}'.format(fixed_str)
        if not modelp.exists():
            modelp = save_path / 'model_{}'.format(fixed_str)
        if not (modelp).exists():
            fixed_strs = [t.name for t in save_path.glob('model*_*.pth')]
            if latest:
                step = [fixed_str.split('_')[-2].split(':')[-1] for fixed_str in fixed_strs]
            else:  # best
                step = [fixed_str.split('_')[-3].split(':')[-1] for fixed_str in fixed_strs]
            step = np.asarray(step, dtype=float)
            step_ind = step.argmax()
            fixed_str = fixed_strs[step_ind].replace('model_', '')
            modelp = save_path / 'model_{}'.format(fixed_str)
        logging.info(f'you are using gpu, load model, {modelp}')
        model_state_dict = torch.load(str(modelp), map_location=lambda storage, loc: storage)
        model_state_dict = {k: v for k, v in model_state_dict.items() if 'num_batches_tracked' not in k}
        if gl_conf.cvt_ipabn:
            import copy
            model_state_dict2 = copy.deepcopy(model_state_dict)
            for k in model_state_dict2.keys():
                if 'running_mean' in k:
                    name = k.replace('running_mean', 'weight')
                    model_state_dict2[name] = torch.abs(model_state_dict[name])
            model_state_dict = model_state_dict2
        if list(model_state_dict.keys())[0].startswith('module'):
            self.model.load_state_dict(model_state_dict, strict=True, )
        else:
            self.model.module.load_state_dict(model_state_dict, strict=True, )


if gl_conf.fast_load:
    class data_prefetcher():
        def __init__(self, loader):
            self.loader = iter(loader)
            self.stream = torch.cuda.Stream()
            self.mean = torch.tensor([.5 * 255, .5 * 255, .5 * 255]).cuda().view(1, 3, 1, 1)
            self.std = torch.tensor([.5 * 255, .5 * 255, .5 * 255]).cuda().view(1, 3, 1, 1)
            # With Amp, it isn't necessary to manually convert data to half.
            # Type conversions are done internally on the fly within patched torch functions.
            if gl_conf.fp16:
                self.mean = self.mean.half()
                self.std = self.std.half()
            self.buffer = []
            self.preload()
        
        def preload(self):
            try:
                ind_loader, data_loader = next(self.loader)
                with torch.cuda.stream(self.stream):
                    data_loader['imgs'] = data_loader['imgs'].cuda()
                    data_loader['labels_cpu'] = data_loader['labels']
                    data_loader['labels'] = data_loader['labels'].cuda()
                    if gl_conf.fp16:
                        data_loader['imgs'] = data_loader['imgs'].half()
                    else:
                        data_loader['imgs'] = data_loader['imgs'].float()
                    data_loader['imgs'] = data_loader['imgs'].sub_(self.mean).div_(self.std)
                    self.buffer.append((ind_loader, data_loader))
            except StopIteration:
                self.buffer.append((None, None))
                return
        
        def next(self):
            torch.cuda.current_stream().wait_stream(self.stream)
            self.preload()
            res = self.buffer.pop(0)
            return res
        
        __next__ = next
        
        def __iter__(self):
            while True:
                ind, data = self.next()
                if ind is None:
                    raise StopIteration
                yield ind, data
else:
    data_prefetcher = lambda x: x


def fast_collate(batch):
    imgs = [img['imgs'] for img in batch]
    targets = torch.tensor([target['labels'] for target in batch], dtype=torch.int64)
    ind_inds = torch.tensor([target['ind_inds'] for target in batch], dtype=torch.int64)
    indexes = torch.tensor([target['indexes'] for target in batch], dtype=torch.int64)
    is_trains = torch.tensor([target['is_trains'] for target in batch], dtype=torch.int64)
    w = imgs[0].shape[1]
    h = imgs[0].shape[2]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)
        # nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)
    
    return {'imgs': tensor, 'labels': targets,
            'ind_inds': ind_inds, 'indexes': indexes,
            'is_trains': is_trains}


class face_learner(object):
    def __init__(self, conf, *args, **kwargs):
        self.milestones = conf.milestones
        ## torch reader
        self.dataset = TorchDataset(gl_conf.use_data_folder)
        self.val_loader_cache = {}
        self.loader = DataLoader(
            self.dataset, batch_size=conf.batch_size,
            num_workers=conf.num_workers,
            sampler=RandomIdSampler(self.dataset.imgidx,
                                    self.dataset.ids, self.dataset.id2range),
            drop_last=True, pin_memory=True,
            collate_fn=torch.utils.data.dataloader.default_collate if not gl_conf.fast_load else fast_collate
        )
        self.class_num = self.dataset.num_classes
        logging.info(f'{self.class_num} classes, load ok ')
        if conf.need_log:
            if torch.distributed.is_initialized():
                lz.set_file_logger(str(conf.log_path) + f'/proc{torch.distributed.get_rank()}')
                lz.set_file_logger_prt(str(conf.log_path) + f'/proc{torch.distributed.get_rank()}')
                lz.mkdir_p(conf.log_path, delete=False)
            else:
                lz.mkdir_p(conf.log_path, delete=True)
                lz.set_file_logger(str(conf.log_path))
                lz.set_file_logger_prt(str(conf.log_path))
                # lz.mkdir_p(conf.log_path, delete=True)
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            self.writer = SummaryWriter(str(conf.log_path))
        else:
            self.writer = None
        self.step = 0
        
        if conf.net_mode == 'mobilefacenet':
            self.model = MobileFaceNet(conf.embedding_size)
            logging.info('MobileFaceNet model generated')
        elif conf.net_mode == 'ir_se' or conf.net_mode == 'ir':
            self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode)
            logging.info('{}_{} model generated'.format(conf.net_mode, conf.net_depth))
        else:
            raise ValueError(conf.net_mode)
        if conf.kd:
            save_path = Path('work_space/emore.r152.cont/save/')
            fixed_str = [t.name for t in save_path.glob('model*_*.pth')][0]
            if not gl_conf.sftlbl_from_file:
                self.teacher_model = Backbone(152, conf.drop_ratio, 'ir_se')
                self.teacher_model = torch.nn.DataParallel(self.teacher_model).cuda()
                modelp = save_path / fixed_str
                model_state_dict = torch.load(modelp)
                model_state_dict = {k: v for k, v in model_state_dict.items() if 'num_batches_tracked' not in k}
                if list(model_state_dict.keys())[0].startswith('module'):
                    self.teacher_model.load_state_dict(model_state_dict, strict=True)
                else:
                    self.teacher_model.module.load_state_dict(model_state_dict, strict=True)
                self.teacher_model.eval()
            self.teacher_head = Arcface(embedding_size=conf.embedding_size, classnum=self.class_num).cuda()
            head_state_dict = torch.load(save_path / 'head_{}'.format(fixed_str.replace('model_', '')))
            self.teacher_head.load_state_dict(head_state_dict)
        if conf.loss == 'arcface':
            self.head = Arcface(embedding_size=conf.embedding_size, classnum=self.class_num)
        elif conf.loss == 'softmax':
            self.head = MySoftmax(embedding_size=conf.embedding_size, classnum=self.class_num)
        elif conf.loss == 'arcfaceneg':
            self.head = ArcfaceNeg(embedding_size=conf.embedding_size, classnum=self.class_num)
        else:
            raise ValueError(f'{conf.loss}')
        if conf.local_rank is None:
            self.model = torch.nn.DataParallel(self.model).cuda()
        else:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model.cuda(),
                                                                   device_ids=[conf.local_rank],
                                                                   output_device=conf.local_rank)
        
        if conf.head_init:
            kernel = lz.msgpack_load(conf.head_init).astype(np.float32).transpose()
            kernel = torch.from_numpy(kernel)
            assert self.head.kernel.shape == kernel.shape
            self.head.kernel.data = kernel
        if self.head is not None:
            self.head = self.head.cuda()
        if gl_conf.tri_wei != 0:
            self.head_triplet = TripletLoss().cuda()
        logging.info(' model heads generated')
        
        paras_only_bn, paras_wo_bn = separate_bn_paras(self.model)
        if conf.use_opt == 'adam':  # todo deprecated
            self.optimizer = optim.Adam([{'params': paras_wo_bn + [*self.head.parameters()], 'weight_decay': 0},
                                         {'params': paras_only_bn}, ],
                                        betas=(gl_conf.adam_betas1, gl_conf.adam_betas2),
                                        amsgrad=True,
                                        lr=conf.lr,
                                        )
        elif conf.net_mode == 'mobilefacenet' or conf.net_mode == 'csmobilefacenet':
            if conf.use_opt == 'sgd':
                self.optimizer = optim.SGD([
                    {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                    {'params': [paras_wo_bn[-1]] + [*self.head.parameters()], 'weight_decay': 4e-4},
                    {'params': paras_only_bn}], lr=conf.lr, momentum=conf.momentum)
                # embed()
            elif conf.use_opt == 'adabound':
                from tools.adabound import AdaBound
                self.optimizer = AdaBound([
                    {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                    {'params': [paras_wo_bn[-1]] + [*self.head.parameters()], 'weight_decay': 4e-4},
                    {'params': paras_only_bn}
                ], lr=conf.lr, betas=(gl_conf.adam_betas1, gl_conf.adam_betas2),
                    gamma=1e-3, final_lr=gl_conf.final_lr, )
        elif conf.use_opt == 'sgd':
            self.optimizer = optim.SGD([
                {'params': paras_wo_bn + [*self.head.parameters()],
                 'weight_decay': gl_conf.weight_decay},
                {'params': paras_only_bn},
            ], lr=conf.lr, momentum=conf.momentum)
        elif conf.use_opt == 'adabound':
            from tools.adabound import AdaBound
            self.optimizer = AdaBound([
                {'params': paras_wo_bn + [*self.head.parameters()],
                 'weight_decay': gl_conf.weight_decay},
                {'params': paras_only_bn},
            ], lr=conf.lr, betas=(gl_conf.adam_betas1, gl_conf.adam_betas2),
                gamma=1e-3, final_lr=gl_conf.final_lr,
            )
        else:
            raise ValueError(f'{conf.use_opt}')
        if gl_conf.fp16:
            if gl_conf.use_test:
                nloss = 2
            else:
                nloss = 1
            self.optimizer = amp_handle.wrap_optimizer(self.optimizer, num_loss=nloss)
        logging.info(f'optimizers generated {self.optimizer}')
        self.board_loss_every = gl_conf.board_loss_every
        # from data.data_pipe import get_val_data
        # self.agedb_30, self.cfp_fp, self.lfw, self.agedb_30_issame, self.cfp_fp_issame, self.lfw_issame = get_val_data(
        #     self.dataset.root_path)  # todo postpone load eval
        self.head.train()
        self.model.train()
    
    def train_dist(self, conf, epochs):
        self.model.train()
        loader = self.loader
        self.evaluate_every = gl_conf.other_every or len(loader) // 3
        self.save_every = gl_conf.other_every or len(loader) // 3
        self.step = gl_conf.start_step
        writer = self.writer
        lz.timer.since_last_check('start train')
        data_time = lz.AverageMeter()
        loss_time = lz.AverageMeter()
        accuracy = 0
        dist_need_log = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        
        if gl_conf.start_eval:
            for ds in ['cfp_fp', ]:  # 'lfw',  'agedb_30'
                accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(
                    conf,
                    self.loader.dataset.root_path,
                    ds)
                if dist_need_log:
                    self.board_val(ds, accuracy, best_threshold, roc_curve_tensor, writer)
                    logging.info(f'validation accuracy on {ds} is {accuracy} ')
        if dist_need_log:
            self.save_state(conf, 0)
        for e in range(conf.start_epoch, epochs):
            lz.timer.since_last_check('epoch {} started'.format(e))
            self.schedule_lr(e)
            loader_enum = data_prefetcher(enumerate(loader))
            while True:
                ind_data, data = next(loader_enum)
                if ind_data is None:
                    logging.info(f'one epoch finish {e} {ind_data}')
                    loader_enum = data_prefetcher(enumerate(loader))
                    ind_data, data = next(loader_enum)
                if (self.step + 1) % len(loader) == 0:
                    self.step += 1
                    break
                imgs = data['imgs'].cuda()
                assert imgs.max() < 2
                if 'labels_cpu' in data:
                    labels_cpu = data['labels_cpu'].cpu()
                else:
                    labels_cpu = data['labels'].cpu()
                labels = data['labels'].cuda()
                data_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                self.optimizer.zero_grad()
                embeddings = self.model(imgs, mode='train')
                thetas = self.head(embeddings, labels)
                loss_xent = conf.ce_loss(thetas, labels)
                if gl_conf.fp16:
                    with self.optimizer.scale_loss(loss_xent) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss_xent.backward()
                if dist_need_log:
                    with torch.no_grad():
                        if gl_conf.mining == 'dop':
                            update_dop_cls(thetas, labels_cpu, gl_conf.dop)
                        if gl_conf.mining == 'rand.id':
                            gl_conf.dop[labels_cpu.numpy()] = 1
                    gl_conf.explored[labels_cpu.numpy()] = 1
                    with torch.no_grad():
                        acc_t = (thetas.argmax(dim=1) == labels)
                        acc = ((acc_t.sum()).item() + 0.0) / acc_t.shape[0]
                self.optimizer.step()
                loss_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                if dist_need_log and self.step % self.board_loss_every == 0:
                    logging.info(f'epoch {e}/{epochs} step {self.step}/{len(loader)}: ' +
                                 f'xent: {loss_xent.item():.2e} ' +
                                 f'data time: {data_time.avg:.2f} ' +
                                 f'loss time: {loss_time.avg:.2f} ' +
                                 f'acc: {acc:.2e} ' +
                                 f'speed: {gl_conf.batch_size / (data_time.avg + loss_time.avg):.2f} imgs/s')
                    writer.add_scalar('info/lr', self.optimizer.param_groups[0]['lr'], self.step)
                    writer.add_scalar('loss/xent', loss_xent.item(), self.step)
                    writer.add_scalar('info/acc', acc, self.step)
                    writer.add_scalar('info/speed', gl_conf.batch_size / (data_time.avg + loss_time.avg), self.step)
                    writer.add_scalar('info/datatime', data_time.avg, self.step)
                    writer.add_scalar('info/losstime', loss_time.avg, self.step)
                    writer.add_scalar('info/epoch', e, self.step)
                    dop = gl_conf.dop
                    if dop is not None:
                        # writer.add_histogram('top_imp', dop, self.step)
                        writer.add_scalar('info/doprat',
                                          np.count_nonzero(gl_conf.explored == 0) / dop.shape[0], self.step)
                
                if not conf.no_eval and self.step % self.evaluate_every == 0 and self.step != 0:
                    for ds in ['cfp_fp', ]:  # 'lfw',  'agedb_30'
                        accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(
                            conf,
                            self.loader.dataset.root_path,
                            ds)
                        if dist_need_log:
                            self.board_val(ds, accuracy, best_threshold, roc_curve_tensor, writer)
                            logging.info(f'validation accuracy on {ds} is {accuracy} ')
                
                if dist_need_log and self.step % self.save_every == 0 and self.step != 0:
                    self.save_state(conf, accuracy)
                
                self.step += 1
                if gl_conf.prof and self.step % 29 == 28:
                    break
            if gl_conf.prof and e > 5:
                break
        self.save_state(conf, accuracy, to_save_folder=True, extra='final')
    
    def warmup(self, conf, epochs):
        self.model.train()
        loader = self.loader
        self.evaluate_every = gl_conf.other_every or len(loader) // 3
        self.save_every = gl_conf.other_every or len(loader) // 3
        self.step = gl_conf.start_step
        writer = SummaryWriter(str(conf.log_path) + '/warmup')
        lz.timer.since_last_check('start train')
        data_time = lz.AverageMeter()
        loss_time = lz.AverageMeter()
        accuracy = 0
        ttl_batch = int(epochs * len(loader))
        for e in range(conf.start_epoch, int(epochs) + 1):
            lz.timer.since_last_check('epoch {} started'.format(e))
            loader_enum = data_prefetcher(enumerate(loader))
            while True:
                now_lr = gl_conf.lr * (self.step + 1) / ttl_batch
                for params in self.optimizer.param_groups:
                    params['lr'] = now_lr
                ind_data, data = next(loader_enum)
                if ind_data is None:
                    logging.info(f'one epoch finish {e} {ind_data}')
                    loader_enum = data_prefetcher(enumerate(loader))
                    ind_data, data = next(loader_enum)
                if (self.step + 1) % len(loader) == 0 or self.step > ttl_batch:
                    self.step += 1
                    break
                imgs = data['imgs'].cuda()
                assert imgs.max() < 2
                if 'labels_cpu' in data:
                    labels_cpu = data['labels_cpu'].cpu()
                else:
                    labels_cpu = data['labels'].cpu()
                labels = data['labels'].cuda()
                data_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                self.optimizer.zero_grad()
                embeddings = self.model(imgs, mode='train')
                thetas = self.head(embeddings, labels)
                loss_xent = conf.ce_loss(thetas, labels)
                if gl_conf.fp16:
                    with self.optimizer.scale_loss(loss_xent) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss_xent.backward()
                with torch.no_grad():
                    if gl_conf.mining == 'dop':
                        update_dop_cls(thetas, labels_cpu, gl_conf.dop)
                    if gl_conf.mining == 'rand.id':
                        gl_conf.dop[labels_cpu.numpy()] = 1
                gl_conf.explored[labels_cpu.numpy()] = 1
                with torch.no_grad():
                    acc_t = (thetas.argmax(dim=1) == labels)
                    acc = ((acc_t.sum()).item() + 0.0) / acc_t.shape[0]
                self.optimizer.step()
                loss_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                if self.step % self.board_loss_every == 0:
                    logging.info(f'epoch {e}/{epochs} lr {now_lr}' +
                                 f'step {self.step}/{len(loader)}: ' +
                                 f'xent: {loss_xent.item():.2e} ' +
                                 f'data time: {data_time.avg:.2f} ' +
                                 f'loss time: {loss_time.avg:.2f} ' +
                                 f'acc: {acc:.2e} ' +
                                 f'speed: {gl_conf.batch_size / (data_time.avg + loss_time.avg):.2f} imgs/s')
                    writer.add_scalar('info/lr', self.optimizer.param_groups[0]['lr'], self.step)
                    writer.add_scalar('loss/xent', loss_xent.item(), self.step)
                    writer.add_scalar('info/acc', acc, self.step)
                    writer.add_scalar('info/speed', gl_conf.batch_size / (data_time.avg + loss_time.avg), self.step)
                    writer.add_scalar('info/datatime', data_time.avg, self.step)
                    writer.add_scalar('info/losstime', loss_time.avg, self.step)
                    writer.add_scalar('info/epoch', e, self.step)
                    dop = gl_conf.dop
                    if dop is not None:
                        writer.add_histogram('top_imp', dop, self.step)
                        writer.add_scalar('info/doprat',
                                          np.count_nonzero(gl_conf.explored == 0) / dop.shape[0], self.step)
                
                if not conf.no_eval and self.step % self.evaluate_every == 0 and self.step != 0:
                    for ds in ['cfp_fp', ]:  # 'lfw',  'agedb_30'
                        accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(
                            conf,
                            self.loader.dataset.root_path,
                            ds)
                        self.board_val(ds, accuracy, best_threshold, roc_curve_tensor, writer)
                        logging.info(f'validation accuracy on {ds} is {accuracy} ')
                if self.step % self.save_every == 0 and self.step != 0:
                    self.save_state(conf, accuracy)
                
                self.step += 1
                if gl_conf.prof and self.step % 29 == 28:
                    break
            if gl_conf.prof and e > 5:
                break
        self.save_state(conf, accuracy, to_save_folder=True, extra='final')
    
    def train_simple(self, conf, epochs):
        self.model.train()
        loader = self.loader
        self.evaluate_every = gl_conf.other_every or len(loader) // 3
        self.save_every = gl_conf.other_every or len(loader) // 3
        self.step = gl_conf.start_step
        writer = self.writer
        lz.timer.since_last_check('start train')
        data_time = lz.AverageMeter()
        loss_time = lz.AverageMeter()
        accuracy = 0
        if gl_conf.start_eval:
            for ds in ['cfp_fp', ]:
                accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(
                    conf,
                    self.loader.dataset.root_path,
                    ds)
                self.board_val(ds, accuracy, best_threshold, roc_curve_tensor, writer)
                logging.info(f'validation accuracy on {ds} is {accuracy} ')
        for e in range(conf.start_epoch, epochs):
            lz.timer.since_last_check('epoch {} started'.format(e))
            self.schedule_lr(e)
            loader_enum = data_prefetcher(enumerate(loader))
            while True:
                ind_data, data = next(loader_enum)
                if ind_data is None:
                    logging.info(f'one epoch finish {e} {ind_data}')
                    loader_enum = data_prefetcher(enumerate(loader))
                    ind_data, data = next(loader_enum)
                if (self.step + 1) % len(loader) == 0:
                    self.step += 1
                    break
                imgs = data['imgs'].cuda()
                assert imgs.max() < 2
                if 'labels_cpu' in data:
                    labels_cpu = data['labels_cpu'].cpu()
                else:
                    labels_cpu = data['labels'].cpu()
                labels = data['labels'].cuda()
                data_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                self.optimizer.zero_grad()
                embeddings = self.model(imgs, mode='train')
                thetas = self.head(embeddings, labels)
                loss_xent = conf.ce_loss(thetas, labels)
                if gl_conf.fp16:
                    with self.optimizer.scale_loss(loss_xent) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss_xent.backward()
                with torch.no_grad():
                    if gl_conf.mining == 'dop':
                        update_dop_cls(thetas, labels_cpu, gl_conf.dop)
                    if gl_conf.mining == 'rand.id':
                        gl_conf.dop[labels_cpu.numpy()] = 1
                gl_conf.explored[labels_cpu.numpy()] = 1
                with torch.no_grad():
                    acc_t = (thetas.argmax(dim=1) == labels)
                    acc = ((acc_t.sum()).item() + 0.0) / acc_t.shape[0]
                self.optimizer.step()
                loss_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                if self.step % self.board_loss_every == 0:
                    logging.info(f'epoch {e}/{epochs} step {self.step}/{len(loader)}: ' +
                                 f'xent: {loss_xent.item():.2e} ' +
                                 f'data time: {data_time.avg:.2f} ' +
                                 f'loss time: {loss_time.avg:.2f} ' +
                                 f'acc: {acc:.2e} ' +
                                 f'speed: {gl_conf.batch_size / (data_time.avg + loss_time.avg):.2f} imgs/s')
                    writer.add_scalar('info/lr', self.optimizer.param_groups[0]['lr'], self.step)
                    writer.add_scalar('loss/xent', loss_xent.item(), self.step)
                    writer.add_scalar('info/acc', acc, self.step)
                    writer.add_scalar('info/speed', gl_conf.batch_size / (data_time.avg + loss_time.avg), self.step)
                    writer.add_scalar('info/datatime', data_time.avg, self.step)
                    writer.add_scalar('info/losstime', loss_time.avg, self.step)
                    writer.add_scalar('info/epoch', e, self.step)
                    dop = gl_conf.dop
                    if dop is not None:
                        writer.add_histogram('top_imp', dop, self.step)
                        writer.add_scalar('info/doprat',
                                          np.count_nonzero(gl_conf.explored == 0) / dop.shape[0], self.step)
                
                if not conf.no_eval and self.step % self.evaluate_every == 0 and self.step != 0:
                    for ds in ['cfp_fp', ]:  # 'lfw',  'agedb_30'
                        accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(
                            conf,
                            self.loader.dataset.root_path,
                            ds)
                        self.board_val(ds, accuracy, best_threshold, roc_curve_tensor, writer)
                        logging.info(f'validation accuracy on {ds} is {accuracy} ')
                if self.step % self.save_every == 0 and self.step != 0:
                    self.save_state(conf, accuracy)
                
                self.step += 1
                if gl_conf.prof and self.step % 29 == 28:
                    break
            if gl_conf.prof and e > 5:
                break
        self.save_state(conf, accuracy, to_save_folder=True, extra='final')
    
    def train_ghm(self, conf, epochs):
        self.ghm_mom = 0.75
        self.gmax = 350
        self.ginterv = 30
        self.bins = int(self.gmax / self.ginterv) + 1
        self.gmax = self.bins * self.ginterv
        self.edges = np.asarray([self.ginterv * x for x in range(self.bins + 1)])
        self.acc_sum = np.zeros(self.bins)
        
        self.model.train()
        loader = self.loader
        self.evaluate_every = gl_conf.other_every or len(loader) // 3
        self.save_every = gl_conf.other_every or len(loader) // 3
        self.step = gl_conf.start_step
        writer = self.writer
        lz.timer.since_last_check('start train')
        data_time = lz.AverageMeter()
        loss_time = lz.AverageMeter()
        accuracy = 0
        for e in range(conf.start_epoch, epochs):
            lz.timer.since_last_check('epoch {} started'.format(e))
            self.schedule_lr(e)
            loader_enum = data_prefetcher(enumerate(loader))
            while True:
                ind_data, data = next(loader_enum)
                if ind_data is None:
                    logging.info(f'one epoch finish {e} {ind_data}')
                    loader_enum = data_prefetcher(enumerate(loader))
                    ind_data, data = next(loader_enum)
                if (self.step + 1) % len(loader) == 0:
                    self.step += 1
                    break
                imgs = data['imgs'].cuda()
                assert imgs.max() < 2
                if 'labels_cpu' in data:
                    labels_cpu = data['labels_cpu'].cpu()
                else:
                    labels_cpu = data['labels'].cpu()
                labels = data['labels'].cuda()
                data_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                self.optimizer.zero_grad()
                embeddings = self.model(imgs, mode='train')
                thetas = self.head(embeddings, labels)
                loss_xent = nn.CrossEntropyLoss(reduction='none')(thetas, labels)
                grad_xent = torch.autograd.grad(loss_xent.sum(),
                                                embeddings,
                                                retain_graph=True,
                                                create_graph=False, only_inputs=True,
                                                allow_unused=True)[0].detach()
                weights = torch.zeros_like(loss_xent)
                with torch.no_grad():
                    gnorm = grad_xent.norm(dim=1).cpu()
                    tot = grad_xent.shape[0]
                    n_valid_bins = 0
                    for i in range(self.bins):
                        inds = (gnorm >= self.edges[i]) & (gnorm < self.edges[i + 1])
                        num_in_bin = inds.sum().item()
                        if num_in_bin > 0:
                            # self.ghm_mom = 0
                            if self.ghm_mom > 0:
                                self.acc_sum[i] = self.ghm_mom * self.acc_sum[i] \
                                                  + (1 - self.ghm_mom) * num_in_bin
                                weights[inds] = tot / self.acc_sum[i]
                            else:
                                weights[inds] = tot / num_in_bin
                            n_valid_bins += 1
                    if n_valid_bins > 0:
                        weights /= n_valid_bins
                weights /= weights.sum()
                loss_xent = (weights * loss_xent).sum()
                if gl_conf.fp16:
                    with self.optimizer.scale_loss(loss_xent) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss_xent.backward()
                with torch.no_grad():
                    if gl_conf.mining == 'dop':
                        update_dop_cls(thetas, labels_cpu, gl_conf.dop)
                    if gl_conf.mining == 'rand.id':
                        gl_conf.dop[labels_cpu.numpy()] = 1
                gl_conf.explored[labels_cpu.numpy()] = 1
                with torch.no_grad():
                    acc_t = (thetas.argmax(dim=1) == labels)
                    acc = ((acc_t.sum()).item() + 0.0) / acc_t.shape[0]
                self.optimizer.step()
                loss_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                if self.step % self.board_loss_every == 0:
                    logging.info(f'epoch {e}/{epochs} step {self.step}/{len(loader)}: ' +
                                 f'xent: {loss_xent.item():.2e} ' +
                                 f'data time: {data_time.avg:.2f} ' +
                                 f'loss time: {loss_time.avg:.2f} ' +
                                 f'acc: {acc:.2e} ' +
                                 f'speed: {gl_conf.batch_size / (data_time.avg + loss_time.avg):.2f} imgs/s')
                    writer.add_scalar('info/lr', self.optimizer.param_groups[0]['lr'], self.step)
                    writer.add_scalar('loss/xent', loss_xent.item(), self.step)
                    writer.add_scalar('info/acc', acc, self.step)
                    writer.add_scalar('info/speed', gl_conf.batch_size / (data_time.avg + loss_time.avg), self.step)
                    writer.add_scalar('info/datatime', data_time.avg, self.step)
                    writer.add_scalar('info/losstime', loss_time.avg, self.step)
                    writer.add_scalar('info/epoch', e, self.step)
                    dop = gl_conf.dop
                    if dop is not None:
                        writer.add_histogram('top_imp', dop, self.step)
                        writer.add_scalar('info/doprat',
                                          np.count_nonzero(gl_conf.explored == 0) / dop.shape[0], self.step)
                
                if not conf.no_eval and self.step % self.evaluate_every == 0 and self.step != 0:
                    for ds in ['cfp_fp', ]:  # 'lfw',  'agedb_30'
                        accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(
                            conf,
                            self.loader.dataset.root_path,
                            ds)
                        self.board_val(ds, accuracy, best_threshold, roc_curve_tensor, writer)
                        logging.info(f'validation accuracy on {ds} is {accuracy} ')
                if self.step % self.save_every == 0 and self.step != 0:
                    self.save_state(conf, accuracy)
                
                self.step += 1
                if gl_conf.prof and self.step % 29 == 28:
                    break
            if gl_conf.prof and e > 5:
                break
        self.save_state(conf, accuracy, to_save_folder=True, extra='final')
    
    def train_with_wei(self, conf, epochs):
        self.model.train()
        loader = self.loader
        self.evaluate_every = gl_conf.other_every or len(loader) // 3
        self.save_every = gl_conf.other_every or len(loader) // 3
        self.step = gl_conf.start_step
        writer = self.writer
        lz.timer.since_last_check('start train')
        data_time = lz.AverageMeter()
        loss_time = lz.AverageMeter()
        accuracy = 0
        d = msgpack_load('work_space/wei.pk')
        weis = d['weis']
        edges = d['edges']
        iwidth = edges[1] - edges[0]
        for e in range(conf.start_epoch, epochs):
            lz.timer.since_last_check('epoch {} started'.format(e))
            self.schedule_lr(e)
            loader_enum = data_prefetcher(enumerate(loader))
            while True:
                ind_data, data = next(loader_enum)
                if ind_data is None:
                    logging.info(f'one epoch finish {e} {ind_data}')
                    loader_enum = data_prefetcher(enumerate(loader))
                    ind_data, data = next(loader_enum)
                if (self.step + 1) % len(loader) == 0:
                    self.step += 1
                    break
                imgs = data['imgs'].cuda()
                assert imgs.max() < 2
                if 'labels_cpu' in data:
                    labels_cpu = data['labels_cpu'].cpu()
                else:
                    labels_cpu = data['labels'].cpu()
                labels = data['labels'].cuda()
                data_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                self.optimizer.zero_grad()
                embeddings = self.model(imgs, mode='train')
                thetas = self.head(embeddings, labels)
                loss_xent = nn.CrossEntropyLoss(reduction='none')(thetas, labels)
                grad_xent = torch.autograd.grad(loss_xent.sum(),
                                                embeddings,
                                                retain_graph=True,
                                                create_graph=False, only_inputs=True,
                                                allow_unused=True)[0].detach()
                with torch.no_grad():
                    gnorm = grad_xent.norm(dim=1).cpu().numpy()
                    locs = np.ceil((gnorm - edges[0]) / iwidth)
                    locs = np.asarray(locs, int)
                    locs[locs > 99] = 99
                    weis_batch = weis[locs]
                    weis_batch += 1e-5
                    weis_batch /= weis_batch.sum()
                    # plt.plot(weis_batch,); plt.show()
                    weis_batch = to_torch(np.asarray(weis_batch, dtype=np.float32)).cuda()
                loss_xent = (weis_batch * loss_xent).sum()
                
                if gl_conf.fp16:
                    with self.optimizer.scale_loss(loss_xent) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss_xent.backward()
                with torch.no_grad():
                    if gl_conf.mining == 'dop':
                        update_dop_cls(thetas, labels_cpu, gl_conf.dop)
                    if gl_conf.mining == 'rand.id':
                        gl_conf.dop[labels_cpu.numpy()] = 1
                gl_conf.explored[labels_cpu.numpy()] = 1
                with torch.no_grad():
                    acc_t = (thetas.argmax(dim=1) == labels)
                    acc = ((acc_t.sum()).item() + 0.0) / acc_t.shape[0]
                self.optimizer.step()
                loss_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                if self.step % self.board_loss_every == 0:
                    logging.info(f'epoch {e}/{epochs} step {self.step}/{len(loader)}: ' +
                                 f'xent: {loss_xent.item():.2e} ' +
                                 f'data time: {data_time.avg:.2f} ' +
                                 f'loss time: {loss_time.avg:.2f} ' +
                                 f'acc: {acc:.2e} ' +
                                 f'speed: {gl_conf.batch_size / (data_time.avg + loss_time.avg):.2f} imgs/s')
                    writer.add_scalar('info/lr', self.optimizer.param_groups[0]['lr'], self.step)
                    writer.add_scalar('loss/xent', loss_xent.item(), self.step)
                    writer.add_scalar('info/acc', acc, self.step)
                    writer.add_scalar('info/speed', gl_conf.batch_size / (data_time.avg + loss_time.avg), self.step)
                    writer.add_scalar('info/datatime', data_time.avg, self.step)
                    writer.add_scalar('info/losstime', loss_time.avg, self.step)
                    writer.add_scalar('info/epoch', e, self.step)
                    dop = gl_conf.dop
                    if dop is not None:
                        writer.add_histogram('top_imp', dop, self.step)
                        writer.add_scalar('info/doprat',
                                          np.count_nonzero(gl_conf.explored == 0) / dop.shape[0], self.step)
                
                if not conf.no_eval and self.step % self.evaluate_every == 0 and self.step != 0:
                    for ds in ['cfp_fp', ]:  # 'lfw',  'agedb_30'
                        accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(
                            conf,
                            self.loader.dataset.root_path,
                            ds)
                        self.board_val(ds, accuracy, best_threshold, roc_curve_tensor, writer)
                        logging.info(f'validation accuracy on {ds} is {accuracy} ')
                if self.step % self.save_every == 0 and self.step != 0:
                    self.save_state(conf, accuracy)
                
                self.step += 1
                if gl_conf.prof and self.step % 29 == 28:
                    break
            if gl_conf.prof and e > 5:
                break
        self.save_state(conf, accuracy, to_save_folder=True, extra='final')
    
    def train_use_test(self, conf, epochs):
        self.model.train()
        loader = self.loader
        if gl_conf.use_test:
            loader_test = DataLoader(
                TestDataset(), batch_size=conf.batch_size,
                num_workers=conf.num_workers, shuffle=True,
                drop_last=True, pin_memory=True,
                collate_fn=torch.utils.data.dataloader.default_collate if not gl_conf.fast_load else fast_collate
            )
            loader_test_enum = data_prefetcher(enumerate(loader))
        self.evaluate_every = gl_conf.other_every or len(loader) // (3 * gl_conf.epoch_less_iter)
        self.save_every = gl_conf.other_every or len(loader) // (3 * gl_conf.epoch_less_iter)
        self.step = gl_conf.start_step
        writer = self.writer
        lz.timer.since_last_check('start train')
        data_time = lz.AverageMeter()
        loss_time = lz.AverageMeter()
        accuracy = 0
        
        for e in range(conf.start_epoch, epochs):
            lz.timer.since_last_check('epoch {} started'.format(e))
            self.schedule_lr(e)
            loader_enum = data_prefetcher(enumerate(loader))
            while True:
                ## get data
                ind_data, data = next(loader_enum)
                if ind_data is None:
                    logging.info(f'one epoch finish {e} {ind_data}')
                    loader_enum = data_prefetcher(enumerate(loader))
                    ind_data, data = next(loader_enum)
                if (self.step + 1) % len(loader) == 0:
                    self.step += 1
                    break
                imgs = data['imgs'].cuda()
                if 'labels_cpu' in data:
                    labels_cpu = data['labels_cpu'].cpu()
                else:
                    labels_cpu = data['labels'].cpu()
                labels = data['labels'].cuda()
                data_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                if gl_conf.use_test:
                    ind_data, imgs_test_data = next(loader_test_enum)
                    if ind_data is None:
                        logging.info(f'test finish {e} {ind_data}')
                        loader_test_enum = data_prefetcher(enumerate(loader_test))
                        ind_data, imgs_test_data = next(loader_enum)
                    imgs_test = imgs_test_data['imgs'].cuda()
                ## get loss and backward
                self.optimizer.zero_grad()  # todo why must put here # I do not know, but order matters!
                if gl_conf.use_test:
                    loss_vat = conf.vat_loss_func(self.model, self.head, imgs_test)
                    if loss_vat != 0:
                        if gl_conf.fp16:
                            with self.optimizer.scale_loss(loss_vat) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss_vat.backward()
                embeddings = self.model(imgs, mode='train')
                thetas = self.head(embeddings, labels)
                loss_xent = conf.ce_loss(thetas, labels)
                if gl_conf.fp16:
                    with self.optimizer.scale_loss(loss_xent) as scaled_loss:
                        scaled_loss.backward()  # todo skip error
                else:
                    loss_xent.backward()
                ## post process
                with torch.no_grad():
                    if gl_conf.mining == 'dop':
                        update_dop_cls(thetas, labels_cpu, gl_conf.dop)
                    if gl_conf.mining == 'rand.id':
                        gl_conf.dop[labels_cpu.numpy()] = 1
                gl_conf.explored[labels_cpu.numpy()] = 1
                with torch.no_grad():
                    acc_t = (thetas.argmax(dim=1) == labels)
                    acc = ((acc_t.sum()).item() + 0.0) / acc_t.shape[0]
                self.optimizer.step()
                loss_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                if self.step % self.board_loss_every == 0:
                    logging.info(f'epoch {e}/{epochs} step {self.step}/{len(loader)}: ' +
                                 f'xent: {loss_xent.item():.2e} ' +
                                 f'data time: {data_time.avg:.2f} ' +
                                 f'loss time: {loss_time.avg:.2f} ' +
                                 f'acc: {acc:.2e} ' +
                                 f'speed: {gl_conf.batch_size / (data_time.avg + loss_time.avg):.2f} imgs/s')
                    writer.add_scalar('info/lr', self.optimizer.param_groups[0]['lr'], self.step)
                    writer.add_scalar('loss/xent', loss_xent.item(), self.step)
                    if gl_conf.use_test:
                        logging.info(f'vat: {loss_vat.item():.2e} ')
                        writer.add_scalar('loss/vat', loss_vat.item(), self.step)
                    writer.add_scalar('info/acc', acc, self.step)
                    writer.add_scalar('info/speed', gl_conf.batch_size / (data_time.avg + loss_time.avg), self.step)
                    writer.add_scalar('info/datatime', data_time.avg, self.step)
                    writer.add_scalar('info/losstime', loss_time.avg, self.step)
                    writer.add_scalar('info/epoch', e, self.step)
                    dop = gl_conf.dop
                    if dop is not None:
                        writer.add_histogram('top_imp', dop, self.step)
                        writer.add_scalar('info/doprat',
                                          np.count_nonzero(gl_conf.explored == 0) / dop.shape[0], self.step)
                
                if not conf.no_eval and self.step % self.evaluate_every == 0 and self.step != 0:
                    for ds in ['cfp_fp', ]:  # 'lfw',  'agedb_30'
                        accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(
                            conf,
                            self.loader.dataset.root_path,
                            ds)
                        self.board_val(ds, accuracy, best_threshold, roc_curve_tensor, writer)
                        logging.info(f'validation accuracy on {ds} is {accuracy} ')
                if self.step % self.save_every == 0 and self.step != 0:
                    self.save_state(conf, accuracy)
                
                self.step += 1
                if gl_conf.prof and self.step % 29 == 28:
                    break
            if gl_conf.prof and e > 5:
                break
        self.save_state(conf, accuracy, to_save_folder=True, extra='final')
    
    # todo train_tri
    def train(self, conf, epochs, mode='train', name=None):
        self.model.train()
        if mode == 'train':
            loader = self.loader
            self.evaluate_every = gl_conf.other_every or len(loader) // 3
            self.save_every = gl_conf.other_every or len(loader) // 3
        elif mode == 'finetune':
            loader = DataLoader(
                self.dataset, batch_size=conf.batch_size * conf.ftbs_mult,
                num_workers=conf.num_workers,
                sampler=RandomIdSampler(self.dataset.imgidx,
                                        self.dataset.ids, self.dataset.id2range),
                drop_last=True, pin_memory=True,
                collate_fn=torch.utils.data.dataloader.default_collate if not gl_conf.fast_load else fast_collate
            )
            self.evaluate_every = gl_conf.other_every or len(loader) // 3
            self.save_every = gl_conf.other_every or len(loader) // 3
        else:
            raise ValueError(mode)
        self.step = gl_conf.start_step
        if name is None:
            writer = self.writer
        else:
            writer = SummaryWriter(str(conf.log_path) + '/ft')
        
        lz.timer.since_last_check('start train')
        data_time = lz.AverageMeter()
        loss_time = lz.AverageMeter()
        accuracy = 0
        tau = 0
        B_multi = 2
        Batch_size = gl_conf.batch_size * B_multi
        batch_size = gl_conf.batch_size
        tau_thresh = 1.2  # todo mv to conf
        #         tau_thresh = (Batch_size + 3 * batch_size) / (3 * batch_size)
        alpha_tau = .9
        for e in range(conf.start_epoch, epochs):
            lz.timer.since_last_check('epoch {} started'.format(e))
            self.schedule_lr(e)
            loader_enum = data_prefetcher(enumerate(loader))
            
            while True:
                ind_data, data = next(loader_enum)
                if ind_data is None:
                    logging.info(f'one epoch finish {e} {ind_data}')
                    loader_enum = data_prefetcher(enumerate(loader))
                    ind_data, data = next(loader_enum)
                if (self.step + 1) % len(loader) == 0:
                    self.step += 1
                    break
                imgs = data['imgs']
                labels_cpu = data['labels_cpu']
                labels = data['labels']
                ind_inds = data['ind_inds']
                # todo visualize it
                # import torchvision
                # imgs_thumb = torchvision.utils.make_grid(
                #     to_torch(imgs), normalize=True,
                #     nrow=int(np.sqrt(imgs.shape[0])) //4 * 4 ,
                #     scale_each=True).numpy()
                # imgs_thumb = to_img(imgs_thumb)
                # # imgs_thumb = cvb.resize_keep_ar( imgs_thumb, 1024,1024, )
                # plt_imshow(imgs_thumb)
                # plt.savefig(work_path+'t.png')
                # plt.close()
                # logging.info(f'this batch labes {labels} ')
                data_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                self.optimizer.zero_grad()
                #                 if not conf.fgg:
                #                 if np.random.rand()>0.5:
                writer.add_scalar('info/tau', tau, self.step)
                if gl_conf.online_imp and tau > tau_thresh and ind_data < len(loader) - B_multi:
                    writer.add_scalar('info/sampl', 1, self.step)
                    imgsl = [imgs]
                    labelsl = [labels]
                    for _ in range(B_multi - 1):
                        ind_data, data = next(loader_enum)
                        imgs = data['imgs'].cuda()
                        labels = data['labels'].cuda()
                        imgsl.append(imgs)
                        labelsl.append(labels)
                    imgs = torch.cat(imgsl, dim=0)
                    labels = torch.cat(labelsl, dim=0)
                    with torch.no_grad():
                        embeddings = self.model(imgs)
                    embeddings.requires_grad_(True)
                    thetas = self.head(embeddings, labels)
                    loss = conf.ce_loss(thetas, labels)
                    grad = torch.autograd.grad(loss, embeddings,
                                               retain_graph=False, create_graph=False,
                                               only_inputs=True)[0].detach()
                    grad.requires_grad_(False)
                    with torch.no_grad():
                        gi = torch.norm(grad, dim=1)
                        gi /= gi.sum()
                        G_ind = torch.multinomial(gi, gl_conf.batch_size, replacement=True)
                        imgs = imgs[G_ind]
                        labels = labels[G_ind]
                        gi_b = gi[G_ind]  # todo this is unbias
                        gi_b = gi_b / gi_b.sum()
                        wi = 1 / gl_conf.batch_size * (1 / gi_b)
                    embeddings = self.model(imgs)
                    thetas = self.head(embeddings, labels)
                    loss = (F.cross_entropy(thetas, labels, reduction='none') * wi).mean()
                    if gl_conf.fp16:
                        with amp_handle.scale_loss(loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                else:
                    writer.add_scalar('info/sampl', 0, self.step)
                    embeddings = self.model(imgs, mode=mode)
                    thetas = self.head(embeddings, labels)
                    if not gl_conf.kd:
                        loss = conf.ce_loss(thetas, labels)
                    else:
                        alpha = gl_conf.alpha
                        T = gl_conf.temperature
                        outputs = thetas
                        with torch.no_grad():
                            if not gl_conf.sftlbl_from_file:
                                teachers_embedding = self.teacher_model(imgs, )
                            else:
                                teachers_embedding = data['teacher_embedding']
                            teacher_outputs = self.teacher_head(teachers_embedding, labels)
                        # loss = -(F.softmax(teacher_outputs / T, dim=1) * F.log_softmax(outputs / T, dim=1)).sum(     dim=1).mean() *alpha
                        loss = F.kl_div(F.log_softmax(outputs / T, dim=1), F.softmax(teacher_outputs / T, dim=1)) * (
                                alpha * T * T)
                        loss += F.cross_entropy(outputs, labels) * (1. - alpha)
                    if gl_conf.tri_wei != 0:
                        loss_triplet, info = self.head_triplet(embeddings, labels, return_info=True)
                        grad_tri = torch.autograd.grad(loss_triplet, embeddings, retain_graph=True, create_graph=False,
                                                       only_inputs=True)[0].detach()
                        
                        writer.add_scalar('info/grad_tri', torch.norm(grad_tri, dim=1).mean().item(), self.step)
                        grad_xent = torch.autograd.grad(loss, embeddings, retain_graph=True, create_graph=False,
                                                        only_inputs=True)[0].detach()
                        writer.add_scalar('info/grad_xent', torch.norm(grad_xent, dim=1).mean().item(), self.step)
                        loss = ((1 - gl_conf.tri_wei) * loss + gl_conf.tri_wei * loss_triplet) / (1 - gl_conf.tri_wei)
                    # if gl_conf.online_imp:
                    #     # todo the order not correct
                    #     grad = torch.autograd.grad(loss, embeddings,
                    #                                retain_graph=True, create_graph=False,
                    #                                only_inputs=True)[0].detach()
                    #     grad.requires_grad_(False)
                    #     gi = torch.norm(grad, dim=1)
                    #     gi /= gi.sum()
                    if gl_conf.fp16:
                        with  self.optimizer.scale_loss(loss) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    with torch.no_grad():
                        if gl_conf.mining == 'dop':
                            update_dop_cls(thetas, labels_cpu, gl_conf.dop)
                        #  if gl_conf.mining == 'imp' :
                        #  for lable_, ind_ind_, gi_ in zip(labels_cpu.numpy(), ind_inds.numpy(), gi.cpu().numpy()):
                        #      gl_conf.id2range_dop[str(lable_)][ind_ind_] = gl_conf.id2range_dop[str(lable_)][
                        #      ind_ind_] * 0.9 + 0.1 * gi_
                        #                             gl_conf.dop[lable_] = gl_conf.id2range_dop[str(lable_)].sum()  # todo should be sum?
                        if gl_conf.mining == 'rand.id':
                            gl_conf.dop[labels_cpu.numpy()] = 1
                gl_conf.explored[labels_cpu.numpy()] = 1
                with torch.no_grad():
                    acc_t = (thetas.argmax(dim=1) == labels)
                    acc = ((acc_t.sum()).item() + 0.0) / acc_t.shape[0]
                if gl_conf.online_imp:
                    tau = alpha_tau * tau + \
                          (1 - alpha_tau) * (1 - (1 / (gi ** 2).sum()).item() * (
                            torch.norm(gi - 1 / len(gi), dim=0) ** 2).item()) ** (-1 / 2)
                
                #                 elif conf.fgg == 'g':
                #                     embeddings_o = self.model(imgs)
                #                     thetas_o = self.head(embeddings, labels)
                #                     loss_o = conf.ce_loss(thetas_o, labels)
                #                     grad = torch.autograd.grad(loss_o, embeddings_o,
                #                                                retain_graph=False, create_graph=False, allow_unused=True,
                #                                                only_inputs=True)[0].detach()
                #                     embeddings = embeddings_o + conf.fgg_wei * grad
                #                     thetas = self.head(embeddings, labels)
                #                     loss = conf.ce_loss(thetas, labels)
                #                     loss.backward()
                #                 elif conf.fgg == 'gg':
                #                     embeddings_o = self.model(imgs)
                #                     thetas_o = self.head(embeddings_o, labels)
                #                     loss_o = conf.ce_loss(thetas_o, labels)
                #                     grad = torch.autograd.grad(loss_o, embeddings_o,
                #                                                retain_graph=True, create_graph=True,
                #                                                only_inputs=True)[0]
                #                     embeddings = embeddings_o + conf.fgg_wei * grad
                #                     thetas = self.head(embeddings, labels)
                #                     loss = conf.ce_loss(thetas, labels)
                #                     loss.backward()
                #                 else:
                #                     raise ValueError(f'{conf.fgg}')
                self.optimizer.step()
                loss_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                if self.step % self.board_loss_every == 0:
                    logging.info(f'epoch {e}/{epochs} step {self.step}/{len(loader)}: ' +
                                 # f'img {imgs.mean()} {imgs.max()} {imgs.min()} '+
                                 f'loss: {loss.item():.2e} ' +
                                 f'data time: {data_time.avg:.2f} ' +
                                 f'loss time: {loss_time.avg:.2f} ' +
                                 f'acc: {acc:.2e} ' +
                                 f'speed: {gl_conf.batch_size / (data_time.avg + loss_time.avg):.2f} imgs/s')
                    writer.add_scalar('info/lr', self.optimizer.param_groups[0]['lr'], self.step)
                    writer.add_scalar('loss/xent', loss.item(), self.step)
                    if gl_conf.tri_wei != 0:
                        writer.add_scalar('loss/triplet', loss_triplet.item(), self.step)
                        writer.add_scalar('loss/dap', info['dap'], self.step)
                        writer.add_scalar('loss/dan', info['dan'], self.step)
                    
                    writer.add_scalar('info/acc', acc, self.step)
                    writer.add_scalar('info/speed',
                                      gl_conf.batch_size / (data_time.avg + loss_time.avg), self.step)
                    writer.add_scalar('info/datatime', data_time.avg, self.step)
                    writer.add_scalar('info/losstime', loss_time.avg, self.step)
                    writer.add_scalar('info/epoch', e, self.step)
                    dop = gl_conf.dop
                    writer.add_histogram('top_imp', dop, self.step)
                    writer.add_scalar('info/doprat',
                                      np.count_nonzero(gl_conf.explored == 0) / dop.shape[0], self.step)
                
                if not conf.no_eval and self.step % self.evaluate_every == 0 and self.step != 0:
                    for ds in ['cfp_fp', ]:  # 'lfw',  'agedb_30'
                        accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(
                            conf,
                            self.loader.dataset.root_path,
                            ds)
                        self.board_val(ds, accuracy, best_threshold, roc_curve_tensor, writer)
                        logging.info(f'validation accuracy on {ds} is {accuracy} ')
                if self.step % self.save_every == 0 and self.step != 0:
                    self.save_state(conf, accuracy)
                
                self.step += 1
                if gl_conf.prof and self.step % 29 == 28:
                    break
            if gl_conf.prof and e > 5:
                break
        self.save_state(conf, accuracy, to_save_folder=True, extra='final')
    
    def train_fgg(self, conf, epochs, mode='train', name=None):
        self.model.train()
        if mode == 'train':
            loader = self.loader
            self.evaluate_every = gl_conf.other_every or len(loader) // 3
            self.save_every = gl_conf.other_every or len(loader) // 3
        elif mode == 'finetune':
            loader = DataLoader(
                self.dataset, batch_size=conf.batch_size * conf.ftbs_mult,
                num_workers=conf.num_workers,
                sampler=RandomIdSampler(self.dataset.imgidx,
                                        self.dataset.ids, self.dataset.id2range),
                drop_last=True, pin_memory=True,
                collate_fn=torch.utils.data.dataloader.default_collate if not gl_conf.fast_load else fast_collate
            )
            self.evaluate_every = gl_conf.other_every or len(loader) // 3
            self.save_every = gl_conf.other_every or len(loader) // 3
        else:
            raise ValueError(mode)
        self.step = gl_conf.start_step
        if name is None:
            writer = self.writer
        else:
            writer = SummaryWriter(str(conf.log_path) + '/ft')
        
        lz.timer.since_last_check('start train')
        data_time = lz.AverageMeter()
        loss_time = lz.AverageMeter()
        accuracy = 0
        for e in range(conf.start_epoch, epochs):
            lz.timer.since_last_check('epoch {} started'.format(e))
            self.schedule_lr(e)
            loader_enum = data_prefetcher(enumerate(loader))
            
            while True:
                ind_data, data = next(loader_enum)
                if ind_data is None:
                    logging.info(f'one epoch finish {e} {ind_data}')
                    loader_enum = data_prefetcher(enumerate(loader))
                    ind_data, data = next(loader_enum)
                if (self.step + 1) % len(loader) == 0:
                    self.step += 1
                    break
                imgs = data['imgs']
                labels_cpu = data['labels_cpu']
                labels = data['labels']
                data_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                self.optimizer.zero_grad()
                embeddings = self.model(imgs, mode=mode)
                thetas = self.head(embeddings, labels)
                if not gl_conf.fgg:
                    loss = conf.ce_loss(thetas, labels)
                    loss.backward()
                elif conf.fgg == 'g':
                    embeddings_o = self.model(imgs)
                    thetas_o = self.head(embeddings, labels)
                    loss_o = conf.ce_loss(thetas_o, labels)
                    grad = torch.autograd.grad(loss_o, embeddings_o,
                                               retain_graph=False, create_graph=False, allow_unused=True,
                                               only_inputs=True)[0].detach()
                    embeddings = embeddings_o + conf.fgg_wei * grad
                    thetas = self.head(embeddings, labels)
                    loss = conf.ce_loss(thetas, labels)
                    loss.backward()
                elif conf.fgg == 'gg':
                    embeddings_o = self.model(imgs)
                    thetas_o = self.head(embeddings_o, labels)
                    loss_o = conf.ce_loss(thetas_o, labels)
                    grad = torch.autograd.grad(loss_o, embeddings_o,
                                               retain_graph=True, create_graph=True,
                                               only_inputs=True)[0]
                    embeddings = embeddings_o + conf.fgg_wei * grad
                    thetas = self.head(embeddings, labels)
                    loss = conf.ce_loss(thetas, labels)
                    loss.backward()
                else:
                    raise ValueError(f'{conf.fgg}')
                
                with torch.no_grad():
                    if gl_conf.mining == 'dop':
                        update_dop_cls(thetas, labels_cpu, gl_conf.dop)
                    if gl_conf.mining == 'rand.id':
                        gl_conf.dop[labels_cpu.numpy()] = 1
                
                gl_conf.explored[labels_cpu.numpy()] = 1
                with torch.no_grad():
                    acc_t = (thetas.argmax(dim=1) == labels)
                    acc = ((acc_t.sum()).item() + 0.0) / acc_t.shape[0]
                
                self.optimizer.step()
                loss_time.update(
                    lz.timer.since_last_check(verbose=False)
                )
                if self.step % self.board_loss_every == 0:
                    logging.info(f'epoch {e}/{epochs} step {self.step}/{len(loader)}: ' +
                                 # f'img {imgs.mean()} {imgs.max()} {imgs.min()} '+
                                 f'loss: {loss.item():.2e} ' +
                                 f'data time: {data_time.avg:.2f} ' +
                                 f'loss time: {loss_time.avg:.2f} ' +
                                 f'acc: {acc:.2e} ' +
                                 f'speed: {gl_conf.batch_size / (data_time.avg + loss_time.avg):.2f} imgs/s')
                    writer.add_scalar('info/lr', self.optimizer.param_groups[0]['lr'], self.step)
                    writer.add_scalar('loss/xent', loss.item(), self.step)
                    writer.add_scalar('info/acc', acc, self.step)
                    writer.add_scalar('info/speed',
                                      gl_conf.batch_size / (data_time.avg + loss_time.avg), self.step)
                    writer.add_scalar('info/datatime', data_time.avg, self.step)
                    writer.add_scalar('info/losstime', loss_time.avg, self.step)
                    writer.add_scalar('info/epoch', e, self.step)
                    dop = gl_conf.dop
                    writer.add_histogram('top_imp', dop, self.step)
                    writer.add_scalar('info/doprat',
                                      np.count_nonzero(gl_conf.explored == 0) / dop.shape[0], self.step)
                
                if not conf.no_eval and self.step % self.evaluate_every == 0 and self.step != 0:
                    for ds in ['cfp_fp', ]:  # 'lfw',  'agedb_30'
                        accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(
                            conf,
                            self.loader.dataset.root_path,
                            ds)
                        self.board_val(ds, accuracy, best_threshold, roc_curve_tensor, writer)
                        logging.info(f'validation accuracy on {ds} is {accuracy} ')
                if self.step % self.save_every == 0 and self.step != 0:
                    self.save_state(conf, accuracy)
                
                self.step += 1
                if gl_conf.prof and self.step % 29 == 28:
                    break
            if gl_conf.prof and e > 5:
                break
        self.save_state(conf, accuracy, to_save_folder=True, extra='final')
    
    def schedule_lr(self, e=0):
        from bisect import bisect_right
        
        e2lr = {epoch: gl_conf.lr * gl_conf.lr_gamma ** bisect_right(self.milestones, epoch) for epoch in
                range(gl_conf.epochs)}
        logging.info(f'map e to lr is {e2lr}')
        lr = e2lr[e]
        for params in self.optimizer.param_groups:
            params['lr'] = lr
        logging.info(f'lr is {lr}')
    
    init_lr = schedule_lr
    
    def save_state(self, conf, accuracy, to_save_folder=False, extra=None, model_only=False):
        # if gl_conf.local_rank is not None and gl_conf.local_rank != 0:
        #     return
        if to_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path
        time_now = get_time()
        lz.mkdir_p(save_path, delete=False)
        # self.model.cpu()
        torch.save(
            self.model.module.state_dict(),
            save_path /
            ('model_{}_accuracy:{}_step:{}_{}.pth'.format(time_now, accuracy, self.step,
                                                          extra)))
        # self.model.cuda()
        lz.msgpack_dump({'dop': gl_conf.dop,
                         'id2range_dop': gl_conf.id2range_dop,
                         }, str(save_path) + f'/extra_{time_now}_accuracy:{accuracy}_step:{self.step}_{extra}.pk')
        if not model_only:
            if self.head is not None:
                self.head.cpu()
                torch.save(
                    self.head.state_dict(),
                    save_path /
                    ('head_{}_accuracy:{}_step:{}_{}.pth'.format(time_now, accuracy, self.step,
                                                                 extra)))
                self.head.cuda()
            torch.save(
                self.optimizer.state_dict(),
                save_path /
                ('optimizer_{}_accuracy:{}_step:{}_{}.pth'.format(time_now, accuracy,
                                                                  self.step, extra)))
    
    def list_steps(self, resume_path):
        from pathlib import Path
        save_path = Path(resume_path)
        fixed_strs = [t.name for t in save_path.glob('model*_*.pth')]
        steps = [fixed_str.split('_')[-2].split(':')[-1] for fixed_str in fixed_strs]
        steps = np.asarray(steps, int)
        return steps
    
    def load_state(self, fixed_str='',
                   resume_path=None, latest=True,
                   load_optimizer=False, load_imp=False, load_head=False
                   ):
        from pathlib import Path
        save_path = Path(resume_path)
        modelp = save_path / '{}'.format(fixed_str)
        if not modelp.exists() or not modelp.is_file():
            modelp = save_path / 'model_{}'.format(fixed_str)
        if not modelp.exists() or not modelp.is_file():
            fixed_strs = [t.name for t in save_path.glob('model*_*.pth')]
            if latest:
                step = [fixed_str.split('_')[-2].split(':')[-1] for fixed_str in fixed_strs]
            else:  # best
                step = [fixed_str.split('_')[-3].split(':')[-1] for fixed_str in fixed_strs]
            step = np.asarray(step, dtype=float)
            assert step.shape[0] > 0, f"{resume_path} chk!"
            step_ind = step.argmax()
            fixed_str = fixed_strs[step_ind].replace('model_', '')
            modelp = save_path / 'model_{}'.format(fixed_str)
        logging.info(f'you are using gpu, load model, {modelp}')
        model_state_dict = torch.load(modelp)
        model_state_dict = {k: v for k, v in model_state_dict.items() if 'num_batches_tracked' not in k}
        if gl_conf.cvt_ipabn:
            import copy
            model_state_dict2 = copy.deepcopy(model_state_dict)
            for k in model_state_dict2.keys():
                if 'running_mean' in k:
                    name = k.replace('running_mean', 'weight')
                    model_state_dict2[name] = torch.abs(model_state_dict[name])
            model_state_dict = model_state_dict2
        if list(model_state_dict.keys())[0].startswith('module'):
            self.model.load_state_dict(model_state_dict, strict=True)
        else:
            self.model.module.load_state_dict(model_state_dict, strict=True)
        
        if load_head:
            assert osp.exists(save_path / 'head_{}'.format(fixed_str))
            logging.info(f'load head from {modelp}')
            head_state_dict = torch.load(save_path / 'head_{}'.format(fixed_str))
            self.head.load_state_dict(head_state_dict)
        if load_optimizer:
            logging.info(f'load opt from {modelp}')
            self.optimizer.load_state_dict(torch.load(save_path / 'optimizer_{}'.format(fixed_str)))
        if load_imp and (save_path / f'extra_{fixed_str.replace(".pth", ".pk")}').exists():
            extra = lz.msgpack_load(save_path / f'extra_{fixed_str.replace(".pth", ".pk")}')
            gl_conf.dop = extra['dop'].copy()
            gl_conf.id2range_dop = extra['id2range_dop'].copy()
    
    def board_val(self, db_name, accuracy, best_threshold, roc_curve_tensor, writer=None):
        writer = writer or self.writer
        writer.add_scalar('{}_accuracy'.format(db_name), accuracy, self.step)
        writer.add_scalar('{}_best_threshold'.format(db_name), best_threshold, self.step)
        # writer.add_image('{}_roc_curve'.format(db_name), roc_curve_tensor, self.step)
    
    def evaluate(self, conf, path, name, nrof_folds=10, tta=True):
        # from utils import ccrop_batch
        self.model.eval()
        idx = 0
        if name in self.val_loader_cache:
            carray, issame = self.val_loader_cache[name]
        else:
            from data.data_pipe import get_val_pair
            carray, issame = get_val_pair(path, name)
            self.val_loader_cache[name] = carray, issame
        carray = carray[:, ::-1, :, :]  # BGR 2 RGB!
        embeddings = np.zeros([len(carray), conf.embedding_size])
        with torch.no_grad():
            while idx + conf.batch_size <= len(carray):
                batch = torch.tensor(carray[idx:idx + conf.batch_size])
                if tta:
                    # batch = ccrop_batch(batch)
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.cuda()) + self.model(fliped.cuda())
                    emb_batch = emb_batch.cpu()
                    embeddings[idx:idx + conf.batch_size] = l2_norm(emb_batch)
                else:
                    embeddings[idx:idx + conf.batch_size] = self.model(batch.cuda()).cpu()
                idx += conf.batch_size
            if idx < len(carray):
                batch = torch.tensor(carray[idx:])
                if tta:
                    # batch = ccrop_batch(batch)
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.cuda()) + self.model(fliped.cuda())
                    emb_batch = emb_batch.cpu()
                    embeddings[idx:] = l2_norm(emb_batch)
                else:
                    embeddings[idx:] = self.model(batch.cuda()).cpu()
        tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
        roc_curve_tensor = None
        # buf = gen_plot(fpr, tpr)
        # roc_curve = Image.open(buf)
        # roc_curve_tensor = trans.ToTensor()(roc_curve)
        self.model.train()
        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor
    
    evaluate_accelerate = evaluate
    
    # todo this evaluate is depracated
    def evaluate_accelerate_dingyi(self, conf, path, name, nrof_folds=10, tta=True):
        lz.timer.since_last_check('start eval')
        self.model.eval()  # set the module in evaluation mode
        idx = 0
        if name in self.val_loader_cache:
            loader = self.val_loader_cache[name]
        else:
            if tta:
                dataset = Dataset_val(path, name, transform=hflip)
            else:
                dataset = Dataset_val(path, name)
            loader = DataLoader(dataset, batch_size=conf.batch_size, num_workers=0,
                                shuffle=False, pin_memory=False)  # todo why shuffle must false
            self.val_loader_cache[name] = loader  # because we have limited memory
        length = len(loader.dataset)
        embeddings = np.zeros([length, conf.embedding_size])
        issame = np.zeros(length)
        
        with torch.no_grad():
            for data in loader:
                carray_batch = data['carray']
                issame_batch = data['issame']
                if tta:
                    fliped = data['fliped_carray']
                    emb_batch = self.model(carray_batch.cuda()) + self.model(fliped.cuda())
                    emb_batch = emb_batch.cpu()
                    embeddings[idx: (idx + conf.batch_size > length) * (length) + (idx + conf.batch_size <= length) * (
                            idx + conf.batch_size)] = l2_norm(emb_batch)
                else:
                    embeddings[idx: (idx + conf.batch_size > length) * (length) + (idx + conf.batch_size <= length) * (
                            idx + conf.batch_size)] = self.model(carray_batch.cuda()).cpu()
                issame[idx: (idx + conf.batch_size > length) * (length) + (idx + conf.batch_size <= length) * (
                        idx + conf.batch_size)] = issame_batch
                idx += conf.batch_size
        
        # tpr/fpr is averaged over various fold division
        try:
            tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
        #     buf = gen_plot(fpr, tpr)
        #     roc_curve = Image.open(buf)
        #     roc_curve_tensor = trans.ToTensor()(roc_curve)
        except Exception as e:
            logging.error(f'{e}')
        roc_curve_tensor = torch.zeros(3, 100, 100)
        self.model.train()
        lz.timer.since_last_check('eval end')
        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor
    
    def validate_ori(self, conf, resume_path=None):
        res = {}
        if resume_path is not None:
            self.load_state(resume_path=resume_path)
        self.model.eval()
        for ds in ['agedb_30', 'lfw', 'cfp_fp', 'cfp_ff', 'calfw', 'cplfw', 'vgg2_fp', ]:
            accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.loader.dataset.root_path,
                                                                       ds)
            logging.info(f'validation accuracy on {ds} is {accuracy} ')
            res[ds] = accuracy
        
        self.model.train()
        return res
    
    # todo deprecated
    def validate(self, conf, resume_path=None):
        if resume_path is not None:
            self.load_state(resume_path=resume_path)
        self.model.eval()
        accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(conf, self.loader.dataset.root_path,
                                                                              'agedb_30')
        logging.info(f'validation accuracy on agedb_30 is {accuracy} ')
        
        accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(conf, self.loader.dataset.root_path,
                                                                              'lfw')
        logging.info(f'validation accuracy on lfw is {accuracy} ')
        
        accuracy, best_threshold, roc_curve_tensor = self.evaluate_accelerate(conf, self.loader.dataset.root_path,
                                                                              'cfp_fp')
        logging.info(f'validation accuracy on cfp_fp is {accuracy} ')
        self.model.train()
    
    def infer(self, conf, faces, target_embs, tta=False):
        '''
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        '''
        embs = []
        for img in faces:
            if tta:
                mirror = trans.functional.hflip(img)
                emb = self.model(conf.test_transform(img).cuda().unsqueeze(0))
                emb_mirror = self.model(conf.test_transform(mirror).cuda().unsqueeze(0))
                embs.append(l2_norm(emb + emb_mirror))
            else:
                embs.append(self.model(conf.test_transform(img).cuda().unsqueeze(0)))
        source_embs = torch.cat(embs)
        
        diff = source_embs.unsqueeze(-1) - target_embs.transpose(1, 0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > self.threshold] = -1  # if no match, set idx to -1
        return min_idx, minimum
    
    def push2redis(self, limits=6 * 10 ** 6 // 8):
        conf = gl_conf
        self.model.eval()
        loader = DataLoader(
            self.dataset, batch_size=conf.batch_size, num_workers=conf.num_workers,
            shuffle=False, sampler=SeqSampler(), drop_last=False,
            pin_memory=True, collate_fn=torch.utils.data.default_collate if not gl_conf.fast_load else fast_collate
        )
        meter = lz.AverageMeter()
        lz.timer.since_last_check(verbose=False)
        for ind_data, data in enumerate(loader):
            meter.update(lz.timer.since_last_check(verbose=False))
            if ind_data % 99 == 0:
                print(ind_data, meter.avg)
            if ind_data > limits:
                break
    
    def calc_teacher_logits(self, out='work_space/teacher_embedding.h5'):
        db = lz.Database(out, 'a')
        conf = gl_conf
        loader = DataLoader(
            self.dataset, batch_size=conf.batch_size, num_workers=conf.num_workers,
            shuffle=False, sampler=SeqSampler(), drop_last=False,
            pin_memory=True,
            collate_fn=torch.utils.data.dataloader.default_collate if not gl_conf.fast_load else fast_collate
        )
        for ind_data, data in data_prefetcher(enumerate(loader)):
            if ind_data % 99 == 3:
                logging.info(f'{ind_data} / {len(loader)}')
            indexes = data['indexes'].numpy()
            imgs = data['imgs']
            labels = data['labels']
            with torch.no_grad():
                if str(indexes.max()) in db:
                    print('skip', indexes.max())
                    continue
                embeddings = self.teacher_model(imgs, )
                outputs = embeddings.cpu().numpy()
                for index, output in zip(indexes, outputs):
                    db[str(index)] = output
        db.flush()
        db.close()
    
    def calc_img_feas(self, out='t.h5'):
        conf = gl_conf
        self.model.eval()
        loader = DataLoader(
            self.dataset, batch_size=conf.batch_size, num_workers=conf.num_workers,
            shuffle=False, sampler=SeqSampler(), drop_last=False,
            pin_memory=True,
            collate_fn=torch.utils.data.dataloader.default_collate if not gl_conf.fast_load else fast_collate
        )
        import h5py
        from sklearn.preprocessing import normalize
        
        f = h5py.File(out, 'w')
        chunksize = 80 * 10 ** 3
        dst = f.create_dataset("feas", (chunksize, 512), maxshape=(None, 512), dtype='f2')
        dst_gtri = f.create_dataset("gtri", (chunksize, 512), maxshape=(None, 512), dtype='f2')
        dst_gxent = f.create_dataset("gxent", (chunksize, 512), maxshape=(None, 512), dtype='f2')
        dst_tri = f.create_dataset("tri", (chunksize,), maxshape=(None,), dtype='f2')
        dst_xent = f.create_dataset("xent", (chunksize,), maxshape=(None,), dtype='f2')
        dst_gtri_norm = f.create_dataset("gtri_norm", (chunksize,), maxshape=(None,), dtype='f2')
        dst_gxent_norm = f.create_dataset("gxent_norm", (chunksize,), maxshape=(None,), dtype='f2')
        dst_img = f.create_dataset("img", (chunksize, 3, 112, 112), maxshape=(None, 3, 112, 112), dtype='f2')
        dst_xent[:chunksize] = -1
        ind_dst = 0
        for ind_data, data in data_prefetcher(enumerate(loader)):
            imgs = data['imgs'].cuda()
            labels = data['labels'].cuda()
            bs = imgs.shape[0]
            if ind_dst + bs > dst.shape[0]:
                dst.resize((dst.shape[0] + chunksize, 512), )
                dst_gxent.resize((dst.shape[0] + chunksize, 512), )
                dst_xent.resize((dst.shape[0] + chunksize,), )
                dst_xent[dst.shape[0]:dst.shape[0] + chunksize] = -1
                dst_gxent_norm.resize((dst.shape[0] + chunksize,), )
                dst_img.resize((dst.shape[0] + chunksize, 3, 112, 112))
            assert (data['indexes'].numpy() == np.arange(ind_dst + 1, ind_dst + bs + 1)).all()
            
            with torch.no_grad():
                embeddings = self.model(imgs, normalize=True)
            embeddings.requires_grad_(True)
            thetas = self.head(embeddings, labels)
            loss_xent = nn.CrossEntropyLoss(reduction='sum')(thetas, labels)  # for grad of each sample
            grad_xent = torch.autograd.grad(loss_xent,
                                            embeddings,
                                            retain_graph=True,
                                            create_graph=False, only_inputs=True,
                                            allow_unused=True)[0].detach()
            
            with torch.no_grad():
                dst[ind_dst:ind_dst + bs, :] = normalize(embeddings.cpu().numpy(), axis=1).astype(np.float16)
                # dst_img[ind_dst:ind_dst + bs, :, :, :] = imgs.cpu().numpy()
                dst_gxent[ind_dst:ind_dst + bs, :] = grad_xent.cpu().numpy().astype(np.float16)
                dst_gxent_norm[ind_dst:ind_dst + bs] = grad_xent.norm(dim=1).cpu().numpy()
                dst_xent[ind_dst:ind_dst + bs] = nn.CrossEntropyLoss(reduction='none')(thetas, labels).cpu().numpy()
            ind_dst += bs
            
            if ind_data % 99 == 0:
                logging.info(f'{ind_data} / {len(loader)}, {loss_xent.item()} {grad_xent.norm(dim=1)[0].item()}')
                # break
        f.flush()
        f.close()
    
    def calc_fc_feas(self, out='t.pk'):
        conf = gl_conf
        self.model.eval()
        loader = DataLoader(
            self.dataset, batch_size=conf.batch_size, num_workers=conf.num_workers,
            shuffle=False, sampler=SeqSampler(), drop_last=False,
            pin_memory=True,
            collate_fn=torch.utils.data.dataloader.default_collate if not gl_conf.fast_load else fast_collate
        )
        features = np.empty((self.dataset.num_classes, 512))
        import collections
        features_tmp = collections.defaultdict(list)
        features_wei = collections.defaultdict(list)
        for ind_data, data in data_prefetcher(enumerate(loader)):
            if ind_data % 99 == 3:
                logging.info(f'{ind_data} / {len(loader)}')
                # break
            imgs = data['imgs']
            labels = data['labels_cpu'].numpy()
            labels_unique = np.unique(labels)
            with torch.no_grad():
                # embeddings = self.model(imgs, normalize=False).half().cpu().numpy().astype(np.float16)
                embeddings = self.model(imgs, normalize=False).cpu().numpy()
            for la in labels_unique:
                features_tmp[la].append(embeddings[labels == la].mean(axis=0))
                features_wei[la].append(np.count_nonzero(labels == la))
        self.nimgs = np.asarray([
            range_[1] - range_[0] for id_, range_ in self.dataset.id2range.items()
        ])
        self.nimgs_normed = self.nimgs / self.nimgs.sum()
        for ind_fea in features_tmp:
            fea_tmp = features_tmp[ind_fea]
            fea_tmp = np.asarray(fea_tmp)
            fea_wei = features_wei[ind_fea]
            fea_wei = np.asarray(fea_wei)
            fea_wei = fea_wei / fea_wei.sum()
            fea_wei = fea_wei.reshape((-1, 1))
            fea = (fea_tmp * fea_wei).sum(axis=0)
            from sklearn.preprocessing import normalize
            print('how many norm', self.nimgs[ind_fea], np.sqrt((fea ** 2).sum()))
            fea = normalize(fea.reshape(1, -1)).flatten()
            features[ind_fea, :] = fea
        lz.msgpack_dump(features, out)
    
    def calc_importance(self, out):
        conf = gl_conf
        self.model.eval()
        loader = DataLoader(
            self.dataset, batch_size=conf.batch_size, num_workers=conf.num_workers,
            shuffle=False, sampler=SeqSampler(), drop_last=False,
            pin_memory=True,
        )
        gl_conf.dop = np.ones(ds.ids.max() + 1, dtype=int) * 1e-8
        gl_conf.id2range_dop = {str(id_):
                                    np.ones((range_[1] - range_[0],)) * 1e-8
                                for id_, range_ in
                                ds.id2range.items()}
        gl_conf.sub_imp_loss = {str(id_):
                                    np.ones((range_[1] - range_[0],)) * 1e-8
                                for id_, range_ in
                                ds.id2range.items()}
        for ind_data, data in enumerate(loader):
            if ind_data % 999 == 0:
                logging.info(f'{ind_data} / {len(loader)}')
            imgs = data['imgs']
            labels_cpu = data['labels']
            ind_inds = data['ind_inds']
            imgs = imgs.cuda()
            labels = labels_cpu.cuda()
            
            with torch.no_grad():
                embeddings = self.model(imgs)
            embeddings.requires_grad_(True)
            thetas = self.head(embeddings, labels)
            losses = nn.CrossEntropyLoss(reduction='none')(thetas, labels)
            loss = losses.mean()
            if gl_conf.tri_wei != 0:
                loss_triplet = self.head_triplet(embeddings, labels)
                loss = ((1 - gl_conf.tri_wei) * loss + gl_conf.tri_wei * loss_triplet) / (1 - gl_conf.tri_wei)
            grad = torch.autograd.grad(loss, embeddings,
                                       retain_graph=True, create_graph=False,
                                       only_inputs=True)[0].detach()
            gi = torch.norm(grad, dim=1)
            for lable_, ind_ind_, gi_, loss_ in zip(labels_cpu.numpy(), ind_inds.numpy(), gi.cpu().numpy(),
                                                    losses.detach().cpu().numpy()):
                gl_conf.id2range_dop[str(lable_)][ind_ind_] = gi_
                gl_conf.sub_imp_loss[str(lable_)][ind_ind_] = loss_
                gl_conf.dop[lable_] = gl_conf.id2range_dop[str(lable_)].mean()
        lz.msgpack_dump({'dop': gl_conf.dop,
                         'id2range_dop': gl_conf.id2range_dop,
                         'sub_imp_loss': gl_conf.sub_imp_loss
                         }, out)
    
    def find_lr(self,
                conf,
                init_value=1e-5,
                final_value=10.,
                beta=0.98,
                bloding_scale=3.,
                num=None):
        if not num:
            num = len(self.loader)
        mult = (final_value / init_value) ** (1 / num)
        lr = init_value
        for params in self.optimizer.param_groups:
            params['lr'] = lr
        self.model.train()
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        log_lrs = []
        loader_enum = data_prefetcher(enumerate(self.loader))
        for i, data in loader_enum:
            imgs = data['imgs']
            labels = data['labels']
            if i % 100 == 0:
                logging.info(f'ok {i}')
            imgs = imgs.cuda()
            labels = labels.cuda()
            batch_num += 1
            
            self.optimizer.zero_grad()
            
            embeddings = self.model(imgs)
            thetas = self.head(embeddings, labels)
            
            if not gl_conf.kd:
                loss = conf.ce_loss(thetas, labels)
            else:
                alpha = gl_conf.alpha
                T = gl_conf.temperature
                outputs = thetas
                with torch.no_grad():
                    teachers_embedding = self.teacher_model(imgs, )
                    teacher_outputs = self.head(teachers_embedding, labels)
                loss = -(F.softmax(teacher_outputs / T, dim=1) * F.log_softmax(outputs / T, dim=1)).sum(
                    dim=1).mean() * T * T * alpha + \
                       F.cross_entropy(outputs, labels) * (1. - alpha)  # todo wrong here
            if gl_conf.tri_wei != 0:
                loss_triplet = self.head_triplet(embeddings, labels)
                loss = (1 - gl_conf.tri_wei) * loss + gl_conf.tri_wei * loss_triplet  # todo
            # Compute the smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            self.writer.add_scalar('avg_loss', avg_loss, batch_num)
            smoothed_loss = avg_loss / (1 - beta ** batch_num)
            self.writer.add_scalar('smoothed_loss', smoothed_loss, batch_num)
            # Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > bloding_scale * best_loss:
                logging.info('exited with best_loss at {}'.format(best_loss))
                plt.plot(log_lrs[10:-5], losses[10:-5])
                return log_lrs, losses
            # Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss
            # Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))
            self.writer.add_scalar('log_lr', math.log10(lr), batch_num)
            # Do the SGD step
            # Update the lr for the next step
            if gl_conf.fp16:
                with amp_handle.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self.optimizer.step()
            
            lr *= mult
            for params in self.optimizer.param_groups:
                params['lr'] = lr
            if batch_num > num:
                print('finish', batch_num, num)
                plt.plot(log_lrs[10:-5], losses[10:-5])
                plt.show()
                plt.savefig('/tmp/tmp.png')
                # from IPython import embed ;   embed()
                return log_lrs, losses


if __name__ == '__main__':
    pass
    # # test thread safe
    ds = TorchDataset(lz.share_path2 + '/faces_ms1m_112x112')
    print(len(ds))
    loader = torch.utils.data.DataLoader(ds, sampler=SeqSampler(), batch_size=32,
                                         num_workers=12, shuffle=False)
    for data in loader:
        print(data['indexes'])
        time.sleep(1)
    # # test random id smpler
    # lz.timer.since_last_check('start')
    # smpler = RandomIdSampler()
    # for idx in smpler:
    #     print(idx)
    #     break
    # print(len(smpler))
    # lz.timer.since_last_check('construct ')
    # flag = False
    # for ids in smpler:
    #     # print(ids)
    #     ids = np.asarray(ids)
    #     assert np.min(ids) >= 0
    #     if np.isclose(ids, 0):
    #         flag = True
    # print(flag)
