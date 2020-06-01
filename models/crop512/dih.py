import torch
from torchvision import transforms
from easydict import EasyDict as edict
from albumentations import HorizontalFlip, PadIfNeeded, RandomCrop

from iharm.data.compose import ComposeDataset
from iharm.data.hdataset import HDataset
from iharm.data.transforms import HCompose
from iharm.engine.simple_trainer import SimpleHTrainer
from iharm.model import initializer
from iharm.model.base import DeepImageHarmonization
from iharm.model.losses import MSE
from iharm.model.metrics import DenormalizedMSEMetric, DenormalizedPSNRMetric
from iharm.utils.log import logger


def main(cfg):
    model, model_cfg = init_model(cfg)
    train(model, cfg, model_cfg, start_epoch=cfg.start_epoch)


def init_model(cfg):
    model_cfg = edict()
    model_cfg.crop_size = (512, 512)
    model_cfg.input_normalization = {
        'mean': [.485, .456, .406],
        'std': [.229, .224, .225]
    }

    model_cfg.input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(model_cfg.input_normalization['mean'], model_cfg.input_normalization['std']),
    ])

    model = DeepImageHarmonization(depth=8)

    model.to(cfg.device)
    model.apply(initializer.XavierGluon(rnd_type='gaussian', magnitude=2.0))

    return model, model_cfg


def train(model, cfg, model_cfg, start_epoch=0):
    cfg.batch_size = 16 if cfg.batch_size < 1 else cfg.batch_size
    cfg.val_batch_size = cfg.batch_size

    cfg.input_normalization = model_cfg.input_normalization
    crop_size = model_cfg.crop_size

    loss_cfg = edict()
    loss_cfg.pixel_loss = MSE()
    loss_cfg.pixel_loss_weight = 1.0

    num_epochs = 120

    train_augmentator = HCompose([
        HorizontalFlip(),
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size)
    ])

    val_augmentator = HCompose([
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size)
    ])

    trainset = ComposeDataset(
        [
            HDataset(cfg.HFLICKR_PATH, split='train'),
            HDataset(cfg.HDAY2NIGHT_PATH, split='train'),
            HDataset(cfg.HCOCO_PATH, split='train'),
            HDataset(cfg.HADOBE5K_PATH, split='train'),
        ],
        augmentator=train_augmentator,
        input_transform=model_cfg.input_transform
    )

    valset = ComposeDataset(
        [
            HDataset(cfg.HFLICKR_PATH, split='test'),
            HDataset(cfg.HDAY2NIGHT_PATH, split='test'),
            HDataset(cfg.HCOCO_PATH, split='test'),
        ],
        augmentator=val_augmentator,
        input_transform=model_cfg.input_transform
    )

    optimizer_params = {
        'lr': 1e-3,
        'betas': (0.9, 0.999), 'eps': 1e-8
    }

    trainer = SimpleHTrainer(
        model, cfg, model_cfg, loss_cfg,
        trainset, valset,
        optimizer='adam',
        optimizer_params=optimizer_params,
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
        checkpoint_interval=5,
        image_dump_interval=500
    )

    logger.info(f'Starting Epoch: {start_epoch}')
    logger.info(f'Total Epochs: {num_epochs}')
    for epoch in range(start_epoch, num_epochs):
        trainer.training(epoch)
        trainer.validation(epoch)
