# -*- coding: future_fstrings -*-

from pathlib import Path
import lz
from lz import *
from torch.nn import CrossEntropyLoss
from torchvision import transforms as trans

# todo label smooth
# todo batch read redis

dist = False
num_devs = 1
if dist:
    num_devs = 1
else:
    # lz.init_dev(3)
    lz.init_dev(lz.get_dev(num_devs))

conf = edict()
conf.num_workers = 24 if not dist else 5
conf.num_devs = num_devs
conf.no_eval = False
conf.start_eval = True
conf.loss = 'arcface'  # softmax arcface arcfaceneg

conf.local_rank = None
conf.num_clss = None
conf.dop = None  # top_imp
conf.id2range_dop = None  # sub_imp
conf.explored = None

conf.data_path = Path('/data2/share/')
conf.work_path = Path('work_space/bak')
conf.model_path = conf.work_path / 'models'
conf.log_path = conf.work_path / 'log'
conf.save_path = conf.work_path / 'save'
vgg_folder = conf.data_path / 'faces_vgg_112x112'
ms1m_folder = conf.data_path / 'faces_ms1m_112x112'
glint_folder = conf.data_path / 'glint'
emore_folder = conf.data_path / 'faces_emore'
asia_emore = conf.data_path / 'asia_emore'
glint_test = conf.data_path / 'glint_test'
alpha_f64 = conf.data_path / 'alpha_f64'
alpha_jk = conf.data_path / 'alpha_jk'

conf.use_data_folder = ms1m_folder  # asia_emore emore_folder glint_folder ms1m_folder alpha_f64
conf.dataset_name = str(conf.use_data_folder).split('/')[-1]

if conf.use_data_folder == ms1m_folder:
    conf.cutoff = 0
elif conf.use_data_folder == glint_folder:
    conf.cutoff = 15
elif conf.use_data_folder == emore_folder:
    conf.cutoff = 0
elif conf.use_data_folder == asia_emore:
    conf.cutoff = 10
else:
    conf.cutoff = 10
conf.mining = 'rand.id'  # todo balance opt # 'dop' 'imp' rand.img(slow) rand.id # todo imp.grad imp.loss
conf.mining_init = 1  # imp 1.6; rand.id 1; dop -1
conf.rand_ratio = 9 / 27

conf.margin = .5
conf.margin2 = .0
conf.topk = 0
conf.fgg = ''  # g gg ''
conf.fgg_wei = 0  # 1
conf.tri_wei = 0
conf.scale = 64.
conf.instances = 4

conf.input_size = [112, 112]
conf.embedding_size = 512

conf.drop_ratio = 0.4
conf.net_mode = 'mobilefacenet'  # csmobilefacenet mobilefacenet ir_se resnext densenet widerresnet
conf.net_depth = 50  # 100 121 169 201 264

conf.test_transform = trans.Compose([
    trans.ToTensor(),
    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

conf.flip = True
conf.upgrade_irse = True
conf.use_redis = False
conf.use_chkpnt = False
conf.chs_first = True
conf.prof = False
conf.fast_load = False
conf.fp16 = False
conf.ipabn = False
conf.cvt_ipabn = False

conf.kd = False
conf.sftlbl_from_file = False
conf.alpha = .95
conf.temperature = 6

conf.online_imp = False
conf.use_test = False  # 'ijbc' 'glint' False
# conf.train_ratio = .7  # todo
conf.device='cuda:0'
conf.batch_size = 210 * num_devs
conf.ftbs_mult = 2
conf.board_loss_every = 10  # 100
conf.other_every = None if not conf.prof else 51
conf.num_recs = 1
# --------------------Training Config ------------------------
conf.log_path = conf.work_path / 'log'
conf.save_path = conf.work_path / 'save'
conf.weight_decay = 5e-4  # 5e-4 , 1e-6 for 1e-3, 0.3 for 3e-3
conf.start_step = 0
conf.use_opt = 'sgd'  # adabound
conf.adam_betas1 = .9  # .85 to .95
conf.adam_betas2 = .999  # 0.999 0.99
conf.final_lr = 1e-1
conf.lr = 1e-1
conf.lr_gamma = 0.1
conf.start_epoch = 0
conf.epochs = 12
conf.warmup = 0  # conf.epochs/25 # 1 0
conf.milestones = [5, 8, 10]
conf.epoch_less_iter = 1
conf.momentum = 0.9
conf.pin_memory = True


# todo may use kl_div to speed up
class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    
    def __init__(self, epsilon=0.1, ):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets1 = torch.zeros(log_probs.size()).scatter_(
            1, targets.unsqueeze(1).data.cpu(), 1).cuda()
        # targets2 = torch.cuda.FloatTensor(inputs.size()).fill_(0).scatter_(1, targets.unsqueeze(1).detach(), 1)
        targets3 = (1 - self.epsilon) * targets1 + \
                   self.epsilon / inputs.shape[1]
        loss = (-targets3 * log_probs).mean(0).sum()
        return loss


conf.ce_loss = CrossEntropyLoss()
# conf.ce_loss = CrossEntropyLabelSmooth()

training = True  # False means test
if not training:
    conf.batch_size *= 2
if not training:
    conf.need_log = False
else:
    conf.need_log = True
conf.batch_size = conf.batch_size // conf.instances * conf.instances
conf.head_init = ''  # work_space/glint.15.fc7.pk
