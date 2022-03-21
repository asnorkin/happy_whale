from collections import Counter

import numpy as np
from torch.utils.data.sampler import BatchSampler


class BalancedSpeciesSampler(BatchSampler):
    def __init__(self, items, batch_size, min_count=100):
        self.batch_size = batch_size
        self.min_count = min_count
        self.weights = self._calculate_weights(items)
        self.indices = np.arange(len(items))

    def __len__(self):
        return int(np.ceil(self.weights.shape[0] / self.batch_size))

    def __iter__(self):
        for batch_i in range(len(self)):
            yield np.random.choice(self.indices, size=self.batch_size, replace=False)

    def _calculate_weights(self, items):
        counts = Counter([item["specie_label"] for item in items])
        weights = [1. / np.sqrt(max(self.min_count, counts[item["specie_label"]])) for item in items]
        return np.asarray(weights)



