from torch.utils.data import Dataset


class DummyDataset(Dataset):
    def __init__(self, dataset, dummy_batches=1):
        self.dataset = dataset
        self.samples = [self.dataset[i] for i in range(dummy_batches)]
        self.i = -1
        self.dummy_batches = dummy_batches

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        if hasattr(self.dataset, item):
            return getattr(self.dataset, item)
        super(DummyDataset).__getattr__(item)

    def worker_init_function(self, worker_index):
        pass

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        self.i = (self.i + 1) % self.dummy_batches
        return self.samples[self.i]
