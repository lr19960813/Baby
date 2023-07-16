import os
import torch

class _dict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


# path params
path = _dict({
    'root_dir': '/home/usrs/shuhei.imai.q4/shuhei.imai/lecture/PBL_ALPS_ALPINE/base_code'
})
path.img_data         = os.path.join(path.root_dir, 'data')
path.all_filelist     = os.path.join(path.root_dir, 'filelist/all.txt')
path.train_filelist   = os.path.join(path.root_dir, 'filelist/train.txt')
path.valid_filelist   = os.path.join(path.root_dir, 'filelist/valid.txt')
path.test_filelist    = os.path.join(path.root_dir, 'filelist/test.txt')
path.log_dir          = os.path.join(path.root_dir, 'log')
path.final_result     = os.path.join(path.root_dir, 'result.txt')
path.pretrained_model = os.path.join(path.root_dir, 'pretrain/model.pt')
path.trained_model    = os.path.join(path.log_dir,  'baby_33.pt')

# training params
train = _dict({
    'num_class': 2,
    'loss_function': torch.nn.functional.cross_entropy,
    'num_epoch': 30,
    'batch_size': 20,
    'learning_rate': 1e-5,
    'betas': [0.9, 0.999],
    'num_worker': 4,
    'tolerance': 5,
    'min_delta': 1e-3
})
