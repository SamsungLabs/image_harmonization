from albumentations import Compose, LongestMaxSize, DualTransform
import albumentations.augmentations.functional as F
import cv2


class HCompose(Compose):
    def __init__(self, transforms, *args, additional_targets=None, no_nearest_for_masks=True, **kwargs):
        if additional_targets is None:
            additional_targets = {
                'target_image': 'image',
                'object_mask': 'mask'
            }
        self.additional_targets = additional_targets
        super().__init__(transforms, *args, additional_targets=additional_targets, **kwargs)
        if no_nearest_for_masks:
            for t in transforms:
                if isinstance(t, DualTransform):
                    t._additional_targets['object_mask'] = 'image'


class LongestMaxSizeIfLarger(LongestMaxSize):
    """
    Rescale an image so that maximum side is less or equal to max_size, keeping the aspect ratio of the initial image.
    If image sides are smaller than the given max_size, no rescaling is applied.

    Args:
        max_size (int): maximum size of smallest side of the image after the transformation.
        interpolation (OpenCV flag): interpolation method. Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """
    def apply(self, img, interpolation=cv2.INTER_LINEAR, **params):
        if max(img.shape[:2]) < self.max_size:
            return img
        return F.longest_max_size(img, max_size=self.max_size, interpolation=interpolation)

    def apply_to_keypoint(self, keypoint, **params):
        height = params["rows"]
        width = params["cols"]

        scale = self.max_size / max([height, width])
        if scale > 1.0:
            return keypoint
        return F.keypoint_scale(keypoint, scale, scale)
