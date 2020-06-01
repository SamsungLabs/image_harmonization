import torch
from iharm.inference.transforms import NormalizeTensor, PadToDivisor, ToTensor, AddFlippedTensor


class Predictor(object):
    def __init__(self, net, device, with_flip=False,
                 mean=(.485, .456, .406), std=(.229, .224, .225)):
        self.device = device
        self.net = net.to(self.device)
        self.net.eval()

        if hasattr(net, 'depth'):
            size_divisor = 2 ** (net.depth + 1)
        else:
            size_divisor = 1

        mean = torch.tensor(mean, dtype=torch.float32)
        std = torch.tensor(std, dtype=torch.float32)
        self.transforms = [
            PadToDivisor(divisor=size_divisor, border_mode=0),
            ToTensor(self.device),
            NormalizeTensor(mean, std, self.device),
        ]
        if with_flip:
            self.transforms.append(AddFlippedTensor())

    def predict(self, image, mask, return_numpy=True):
        with torch.no_grad():
            for transform in self.transforms:
                image, mask = transform.transform(image, mask)

            predicted_image = self.net(image, mask)['images']

            for transform in reversed(self.transforms):
                predicted_image = transform.inv_transform(predicted_image)

            predicted_image = torch.clamp(predicted_image, 0, 255)

        if return_numpy:
            return predicted_image.cpu().numpy()
        else:
            return predicted_image
