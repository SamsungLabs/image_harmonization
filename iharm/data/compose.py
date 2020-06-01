from .base import BaseHDataset


class ComposeDataset(BaseHDataset):
    def __init__(self, datasets, **kwargs):
        super(ComposeDataset, self).__init__(**kwargs)

        self._datasets = datasets
        self.dataset_samples = []
        for dataset_indx, dataset in enumerate(self._datasets):
            self.dataset_samples.extend([(dataset_indx, i) for i in range(len(dataset))])

    def get_sample(self, index):
        dataset_indx, sample_indx = self.dataset_samples[index]
        return self._datasets[dataset_indx].get_sample(sample_indx)
