from torch import multinomial, Tensor
from torch.utils.data import Sampler


class ImbalancedDatasetSampler(Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = dataset.label_count
        total = sum(label_to_count.values())
        # weight for each sample
        # weights = [1.0 - (label_to_count[self._get_label(dataset, idx)] / total)
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = Tensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.label(idx)

    def __iter__(self):
        return (self.indices[i] for i in multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
