import argparse
import sys

sys.path.insert(0, '.')

import torch
from pathlib import Path
from tqdm import trange

from albumentations import Resize, NoOp
from iharm.data.hdataset import HDataset
from iharm.data.transforms import HCompose, LongestMaxSizeIfLarger
from iharm.inference.predictor import Predictor
from iharm.inference.metrics import MetricsHub, MSE, fMSE, PSNR, N
from iharm.inference.utils import load_model, find_checkpoint
from iharm.mconfigs import ALL_MCONFIGS
from iharm.utils.exp import load_config_file
from iharm.utils.log import logger, add_new_file_output_to_logger


RESIZE_STRATEGIES = {
    'None': NoOp(),
    'LimitLongest1024': LongestMaxSizeIfLarger(1024),
    'Fixed256': Resize(256, 256),
    'Fixed512': Resize(512, 512)
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('model_type', choices=ALL_MCONFIGS.keys())
    parser.add_argument('checkpoint', type=str,
                        help='The path to the checkpoint. '
                             'This can be a relative path (relative to cfg.MODELS_PATH) '
                             'or an absolute path. The file extension can be omitted.')
    parser.add_argument('--datasets', type=str, default='HFlickr,HDay2Night,HCOCO,HAdobe5k',
                        help='Each dataset name must be one of the prefixes in config paths, '
                             'which look like DATASET_PATH.')
    parser.add_argument('--resize-strategy', type=str, choices=RESIZE_STRATEGIES.keys(), default='Fixed256')
    parser.add_argument('--use-flip', action='store_true', default=False,
                        help='Use horizontal flip test-time augmentation.')
    parser.add_argument('--gpu', type=str, default=0, help='ID of used GPU.')
    parser.add_argument('--config-path', type=str, default='./config.yml',
                        help='The path to the config file.')
    parser.add_argument('--results-path', type=str, default='',
                        help='The path to the evaluation results. '
                             'Default path: cfg.EXPS_PATH/evaluation_results.')

    parser.add_argument('--eval-prefix', type=str, default='')

    args = parser.parse_args()
    cfg = load_config_file(args.config_path, return_edict=True)
    return args, cfg


def main():
    args, cfg = parse_args()
    checkpoint_path = find_checkpoint(cfg.MODELS_PATH, args.checkpoint)
    add_new_file_output_to_logger(
        logs_path=Path(cfg.EXPS_PATH) / 'evaluation_results',
        prefix=f'{Path(checkpoint_path).stem}_',
        only_message=True
    )
    logger.info(vars(args))

    device = torch.device(f'cuda:{args.gpu}')
    net = load_model(args.model_type, checkpoint_path, verbose=True)
    predictor = Predictor(net, device, with_flip=args.use_flip)

    fg_ratio_intervals = [(0.0, 0.05), (0.05, 0.15), (0.15, 1.0), (0.0, 1.00)]

    datasets_names = args.datasets.split(',')
    datasets_metrics = [[] for _ in fg_ratio_intervals]
    for dataset_indx, dataset_name in enumerate(datasets_names):
        dataset = HDataset(
            cfg.get(f'{dataset_name.upper()}_PATH'), split='test',
            augmentator=HCompose([RESIZE_STRATEGIES[args.resize_strategy]]),
            keep_background_prob=-1
        )

        dataset_metrics = []
        for fg_ratio_min, fg_ratio_max in fg_ratio_intervals:
            dataset_metrics.append(MetricsHub([N(), MSE(), fMSE(), PSNR()],
                                   name=f'{dataset_name} ({fg_ratio_min:.0%}-{fg_ratio_max:.0%})',
                                   name_width=28))

        for sample_i in trange(len(dataset), desc=f'Testing on {dataset_name}'):
            sample = dataset.get_sample(sample_i)
            sample = dataset.augment_sample(sample)

            sample_mask = sample['object_mask']
            sample_fg_ratio = (sample_mask > 0.5).sum() / (sample_mask.shape[0] * sample_mask.shape[1])
            pred = predictor.predict(sample['image'], sample_mask, return_numpy=False)

            target_image = torch.as_tensor(sample['target_image'], dtype=torch.float32).to(predictor.device)
            sample_mask = torch.as_tensor(sample_mask, dtype=torch.float32).to(predictor.device)
            with torch.no_grad():
                for metrics_hub, (fg_ratio_min, fg_ratio_max) in zip(dataset_metrics, fg_ratio_intervals):
                    if fg_ratio_min <= sample_fg_ratio <= fg_ratio_max:
                        metrics_hub.compute_and_add(pred, target_image, sample_mask)

        for indx, metrics_hub in enumerate(dataset_metrics):
            datasets_metrics[indx].append(metrics_hub)
        if dataset_indx == 0:
            logger.info(dataset_metrics[-1].get_table_header())
        for metrics_hub in dataset_metrics:
            logger.info(metrics_hub)

    if len(datasets_metrics) > 1:
        overall_metrics = [sum(x, MetricsHub([], f'Overall ({fg_ratio_min:.0%}-{fg_ratio_max:.0%})', name_width=28))
                           for x, (fg_ratio_min, fg_ratio_max) in zip(datasets_metrics, fg_ratio_intervals)]
        logger.info('-' * len(str(overall_metrics[-1])))
        for x in overall_metrics:
            logger.info(x)


if __name__ == '__main__':
    main()
