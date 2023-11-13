import os

from os.path import join as ospj
from os.path import expanduser
from munch import Munch as mch
import numpy as np

from ds import prepare_coco_dataloaders, prepare_flickr_dataloaders, prepare_cub_dataloaders, prepare_flo_dataloaders

from utils import *
from networks import *
from train_probVLM import *

import matplotlib.pyplot as plt


dataset = 'CUB' # coco or flickr
data_dir = ospj('D:/Download/Datasets/', dataset) # e.g. ospj(expanduser('~'), 'Documents', 'jm', 'data', dataset)
dataloader_config = mch({
    'batch_size': 64,
    'random_erasing_prob': 0.,
    'traindata_shuffle': True
})
loaders,vocab = load_data_loader(dataset, data_dir, dataloader_config)
cub_train_loader, cub_valid_loader, cub_test_loader = loaders['train'], loaders['val'], loaders['test']

# clip_net = load_model('cuda')
CLIP_Net = load_model(device='cuda', model_path=None)
ProbVLM_Net = BayesCap_for_CLIP(
    inp_dim=512,
    out_dim=512,
    hid_dim=256,
    num_layers=3,
    p_drop=0.05,
)

train_ProbVLM(
    CLIP_Net,
    ProbVLM_Net,
    cub_train_loader,
    cub_valid_loader,
    Cri = TempCombLoss(),
    device='cuda',
    dtype=torch.float,
    init_lr=8e-5,
    num_epochs=500,
    eval_every=5,
    ckpt_path='../ckpt/ProbVLM_Net',
    T1=1e0,
    T2=1e-4
)