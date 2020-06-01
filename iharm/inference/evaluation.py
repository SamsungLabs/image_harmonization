from time import time
from tqdm import trange
import torch


def evaluate_dataset(dataset, predictor, metrics_hub):
    for sample_i in trange(len(dataset), desc=f'Testing on {metrics_hub.name}'):
        sample = dataset.get_sample(sample_i)
        sample = dataset.augment_sample(sample)

        sample_mask = sample['object_mask']
        predict_start = time()
        pred = predictor.predict(sample['image'], sample_mask, return_numpy=False)
        torch.cuda.synchronize()
        metrics_hub.update_time(time() - predict_start)

        target_image = torch.as_tensor(sample['target_image'], dtype=torch.float32).to(predictor.device)
        sample_mask = torch.as_tensor(sample_mask, dtype=torch.float32).to(predictor.device)
        with torch.no_grad():
            metrics_hub.compute_and_add(pred, target_image, sample_mask)
