import torch
import torch.nn.functional as F


class TrainMetric(object):
    def __init__(self, pred_outputs, gt_outputs, epsilon=1e-6):
        self.pred_outputs = pred_outputs
        self.gt_outputs = gt_outputs
        self.epsilon = epsilon
        self._last_batch_metric = 0.0
        self._epoch_metric_sum = 0.0
        self._epoch_batch_count = 0

    def compute(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        self._last_batch_metric = self.compute(*args, **kwargs)
        self._epoch_metric_sum += self._last_batch_metric
        self._epoch_batch_count += 1

    def get_epoch_value(self):
        if self._epoch_batch_count > 0:
            return self._epoch_metric_sum / self._epoch_batch_count
        else:
            return 0.0

    def reset_epoch_stats(self):
        self._epoch_metric_sum = 0.0
        self._epoch_batch_count = 0

    def log_states(self, sw, tag_prefix, global_step):
        sw.add_scalar(tag=tag_prefix, value=self._last_batch_metric, global_step=global_step)

    @property
    def name(self):
        return type(self).__name__


class PSNRMetric(TrainMetric):
    def __init__(self, pred_output='instances', gt_output='instances'):
        super(PSNRMetric, self).__init__((pred_output, ), (gt_output, ))

    def compute(self, pred, gt):
        mse = F.mse_loss(pred, gt)
        squared_max = gt.max() ** 2
        psnr = 10 * torch.log10(squared_max / (mse + self.epsilon))
        return psnr.item()


class DenormalizedTrainMetric(TrainMetric):
    def __init__(self, pred_outputs, gt_outputs, mean=None, std=None):
        super(DenormalizedTrainMetric, self).__init__(pred_outputs, gt_outputs)
        self.mean = torch.zeros(1) if mean is None else mean
        self.std = torch.ones(1) if std is None else std
        self.device = None

    def init_device(self, input_device):
        if self.device is None:
            self.device = input_device
            self.mean = self.mean.to(self.device)
            self.std = self.std.to(self.device)

    def denormalize(self, tensor):
        self.init_device(tensor.device)
        return tensor * self.std + self.mean

    def update(self, *args, **kwargs):
        self._last_batch_metric = self.compute(*args, **kwargs)
        self._epoch_metric_sum += self._last_batch_metric
        self._epoch_batch_count += 1


class DenormalizedPSNRMetric(DenormalizedTrainMetric):
    def __init__(
        self,
        pred_output='instances', gt_output='instances',
        mean=None, std=None,
    ):
        super(DenormalizedPSNRMetric, self).__init__((pred_output, ), (gt_output, ), mean, std)

    def compute(self, pred, gt):
        denormalized_pred = torch.clamp(self.denormalize(pred), 0, 1)
        denormalized_gt = self.denormalize(gt)
        return PSNRMetric.compute(self, denormalized_pred, denormalized_gt)


class DenormalizedMSEMetric(DenormalizedTrainMetric):
    def __init__(
        self,
        pred_output='instances', gt_output='instances',
        mean=None, std=None,
    ):
        super(DenormalizedMSEMetric, self).__init__((pred_output, ), (gt_output, ), mean, std)

    def compute(self, pred, gt):
        denormalized_pred = self.denormalize(pred) * 255
        denormalized_gt = self.denormalize(gt) * 255
        return F.mse_loss(denormalized_pred, denormalized_gt).item()
