"""训练仅加载官方 train，增强与 worker 调度、模型随机流完全解耦。"""

import math

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision.datasets import CIFAR100, SVHN
from torchvision.transforms import functional as TF, InterpolationMode

from .common import ROOT, derive_seed, load_json


def official_train(dataset):
    if dataset == 'cifar100':
        return CIFAR100(str(ROOT / 'data'), train=True, download=False)
    if dataset == 'svhn':
        return SVHN(str(ROOT / 'data/svhn'), split='train', download=False)
    raise ValueError('未注册的数据集')


def raw_arrays(raw, dataset):
    return (raw.data, np.asarray(raw.targets)) if dataset == 'cifar100' else (
        raw.data.transpose(0, 2, 3, 1), np.asarray(raw.labels))


def transform(image, dataset, model_key, training, seed, epoch, sample_id):
    protocol = load_json(ROOT / 'configs/protocol.json') if not hasattr(transform, '_protocol') else transform._protocol
    transform._protocol = protocol
    image = TF.resize(image, 256, interpolation=InterpolationMode.BICUBIC, antialias=True)
    if training:
        rng = np.random.default_rng(derive_seed(seed, epoch, sample_id, 0))
        top = int(rng.integers(0, image.height - 224 + 1))
        left = int(rng.integers(0, image.width - 224 + 1))
        image = TF.crop(image, top, left, 224, 224)
        if dataset == 'cifar100' and rng.random() < 0.5:
            image = TF.hflip(image)
    else:
        image = TF.center_crop(image, [224, 224])
    value = TF.to_tensor(image)
    return TF.normalize(value, protocol['models'][model_key]['mean'], protocol['models'][model_key]['std'])


class TrainValData(Dataset):
    def __init__(self, dataset, model_key, split, seed, raw=None):
        if split not in ['train', 'validation']:
            raise ValueError('训练入口仅允许 train/validation')
        self.dataset, self.model_key, self.split, self.seed = dataset, model_key, split, seed
        self.raw = raw if raw is not None else official_train(dataset)
        self.images, self.labels = raw_arrays(self.raw, dataset)
        manifest = load_json(ROOT / 'records' / f'{dataset}_split.json')
        self.indices = manifest['train_indices' if split == 'train' else 'validation_indices']

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        # sampler 将 epoch 随索引发送给 persistent workers，避免隐含共享 RNG 进度。
        epoch, offset = position if isinstance(position, tuple) else (0, position)
        index = self.indices[offset]
        sample_id = f'{self.dataset}/train/{index}'
        image = Image.fromarray(self.images[index])
        tensor = transform(image, self.dataset, self.model_key, self.split == 'train',
                           self.seed, epoch, sample_id)
        return tensor, int(self.labels[index]), sample_id


class EpochSampler(Sampler):
    def __init__(self, dataset, seed, epoch=0, shuffle=True):
        self.dataset, self.seed, self.epoch, self.shuffle = dataset, seed, epoch, shuffle

    def __iter__(self):
        order = np.arange(len(self.dataset))
        if self.shuffle:
            rng = np.random.default_rng(derive_seed('sampler', self.seed, self.dataset.dataset, self.epoch))
            order = rng.permutation(order)
        return iter((self.epoch, int(index)) for index in order)

    def __len__(self):
        return len(self.dataset)


def loader(dataset, seed, training, workers=4):
    sampler = EpochSampler(dataset, seed, shuffle=training)
    # DataLoader 的 generator 只负责 worker seed，不消费模型随机流。
    generator = torch.Generator().manual_seed(derive_seed('workers', seed))
    return DataLoader(dataset, batch_size=32 if training else 128, sampler=sampler,
                      num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
                      generator=generator, drop_last=False), sampler


def epoch_order_hash(sampler):
    import hashlib
    return hashlib.sha256(np.array([index for _, index in sampler], dtype='<i8').tobytes()).hexdigest()
