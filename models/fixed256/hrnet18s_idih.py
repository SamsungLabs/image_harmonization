from functools import partial

import torch
from torchvision import transforms
from easydict import EasyDict as edict
from albumentations import HorizontalFlip, Resize, RandomResizedCrop

from iharm.data.compose import ComposeDataset
from iharm.data.hdataset import HDataset
from iharm.data.transforms import HCompose
from iharm.engine.simple_trainer import SimpleHTrainer
from iharm.mconfigs import BMCONFIGS
from iharm.model import initializer
from iharm.model.backboned import HRNetIHModel
from iharm.model.losses import MaskWeightedMSE
from iharm.model.metrics import DenormalizedMSEMetric, DenormalizedPSNRMetric
from iharm.utils.log import logger


def main(cfg):
    model, model_cfg = init_model(cfg)
    train(model, cfg, model_cfg, start_epoch=cfg.start_epoch)


def init_model(cfg):
    model_cfg = edict()
    model_cfg.crop_size = (256, 256)
    model_cfg.input_normalization = {
        'mean': [.485, .456, .406],
        'std': [.229, .224, .225]
    }

    model_cfg.input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(model_cfg.input_normalization['mean'], model_cfg.input_normalization['std']),
    ])

    model = HRNetIHModel(BMCONFIGS['improved_dih256'])

    model.to(cfg.device)
    model.apply(initializer.XavierGluon(rnd_type='gaussian', magnitude=2.0))
    model.backbone.load_pretrained_weights(cfg.IMAGENET_PRETRAINED_MODELS.HRNETV2_W18_SMALL)

    return model, model_cfg


def train(model, cfg, model_cfg, start_epoch=0):
    cfg.batch_size = 16 if cfg.batch_size < 1 else cfg.batch_size
    cfg.val_batch_size = cfg.batch_size

    cfg.input_normalization = model_cfg.input_normalization
    crop_size = model_cfg.crop_size

    loss_cfg = edict()
    loss_cfg.pixel_loss = MaskWeightedMSE(min_area=100)
    loss_cfg.pixel_loss_weight = 1.0

    num_epochs = 180
    train_augmentator = HCompose([
        RandomResizedCrop(*crop_size, scale=(0.5, 1.0)),
        HorizontalFlip()
    ])
    val_augmentator = HCompose([
        Resize(*crop_size)
    ])

    trainset = ComposeDataset(
        [
            HDataset(cfg.HFLICKR_PATH, split='train'),
            HDataset(cfg.HDAY2NIGHT_PATH, split='train'),
            HDataset(cfg.HCOCO_PATH, split='train'),
            HDataset(cfg.HADOBE5K_PATH, split='train'),
        ],
        augmentator=train_augmentator,
        input_transform=model_cfg.input_transform,
        keep_background_prob=0.05,
    )

    valset = ComposeDataset(
        [
            HDataset(cfg.HFLICKR_PATH, split='test'),
            HDataset(cfg.HDAY2NIGHT_PATH, split='test'),
            HDataset(cfg.HCOCO_PATH, split='test'),
        ],
        augmentator=val_augmentator,
        input_transform=model_cfg.input_transform,
        keep_background_prob=-1,
    )

    optimizer_params = {
        'lr': 1e-3,
        'betas': (0.9, 0.999), 'eps': 1e-8
    }

    lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR,
                           milestones=[160, 175], gamma=0.1)
    trainer = SimpleHTrainer(
        model, cfg, model_cfg, loss_cfg,
        trainset, valset,
        optimizer='adam',
        optimizer_params=optimizer_params,
        lr_scheduler=lr_scheduler,
        metrics=[
            DenormalizedPSNRMetric(
                'images', 'target_images',
                mean=torch.tensor(cfg.input_normalization['mean'], dtype=torch.float32).view(1, 3, 1, 1),
                std=torch.tensor(cfg.input_normalization['std'], dtype=torch.float32).view(1, 3, 1, 1),
            ),
            DenormalizedMSEMetric(
                'images', 'target_images',
                mean=torch.tensor(cfg.input_normalization['mean'], dtype=torch.float32).view(1, 3, 1, 1),
                std=torch.tensor(cfg.input_normalization['std'], dtype=torch.float32).view(1, 3, 1, 1),
            )
        ],
        checkpoint_interval=10,
        image_dump_interval=1000
    )

    logger.info(f'Starting Epoch: {start_epoch}')
    logger.info(f'Total Epochs: {num_epochs}')
    for epoch in range(start_epoch, num_epochs):
        trainer.training(epoch)
        trainer.validation(epoch)
