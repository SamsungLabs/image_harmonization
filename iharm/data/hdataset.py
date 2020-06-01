from pathlib import Path

import cv2
import numpy as np

from .base import BaseHDataset


class HDataset(BaseHDataset):
    def __init__(self, dataset_path, split, blur_target=False, **kwargs):
        super(HDataset, self).__init__(**kwargs)

        self.dataset_path = Path(dataset_path)
        self.blur_target = blur_target
        self._split = split
        self._real_images_path = self.dataset_path / 'real_images'
        self._composite_images_path = self.dataset_path / 'composite_images'
        self._masks_path = self.dataset_path / 'masks'

        images_lists_paths = [x for x in self.dataset_path.glob('*.txt') if x.stem.endswith(split)]
        assert len(images_lists_paths) == 1

        with open(images_lists_paths[0], 'r') as f:
            self.dataset_samples = [x.strip() for x in f.readlines()]

    def get_sample(self, index):
        composite_image_name = self.dataset_samples[index]
        real_image_name = composite_image_name.split('_')[0] + '.jpg'
        mask_name = '_'.join(composite_image_name.split('_')[:-1]) + '.png'

        composite_image_path = str(self._composite_images_path / composite_image_name)
        real_image_path = str(self._real_images_path / real_image_name)
        mask_path = str(self._masks_path / mask_name)

        composite_image = cv2.imread(composite_image_path)
        composite_image = cv2.cvtColor(composite_image, cv2.COLOR_BGR2RGB)

        real_image = cv2.imread(real_image_path)
        real_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB)

        object_mask_image = cv2.imread(mask_path)
        object_mask = object_mask_image[:, :, 0].astype(np.float32) / 255.
        if self.blur_target:
            object_mask = cv2.GaussianBlur(object_mask, (7, 7), 0)

        return {
            'image': composite_image,
            'object_mask': object_mask,
            'target_image': real_image,
            'image_id': index
        }
