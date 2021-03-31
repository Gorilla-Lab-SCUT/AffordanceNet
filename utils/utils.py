import numpy as np
import torch
import torch.nn.functional as F
import random


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


class PN2_Scheduler(object):
    def __init__(self, init_lr, step, decay_rate, min_lr):
        super().__init__()
        self.init_lr = init_lr
        self.step = step
        self.decay_rate = decay_rate
        self.min_lr = min_lr
        return

    def __call__(self, epoch):
        factor = self.decay_rate**(epoch//self.step)
        if self.init_lr*factor < self.min_lr:
            factor = self.min_lr / self.init_lr
        return factor


class PN2_BNMomentum(object):
    def __init__(self, origin_m, m_decay, step):
        super().__init__()
        self.origin_m = origin_m
        self.m_decay = m_decay
        self.step = step
        return

    def __call__(self, m, epoch):
        momentum = self.origin_m * (self.m_decay**(epoch//self.step))
        if momentum < 0.01:
            momentum = 0.01
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum
        return


def set_random_seed(seed):
    # random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
