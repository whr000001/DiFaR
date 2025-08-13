import json
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler


class MyDataset(Dataset):
    def __init__(self, text, image, labels, device):
        self.device = device
        self.labels = labels
        self.image = image
        self.text = text

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        return self.text[index], self.image[index], self.labels[index]

    def get_collate_fn(self):
        def collate_fn(batch):
            text, image, labels = zip(*batch)
            text = torch.stack(text).to(self.device)
            image = torch.stack(image).to(self.device)
            labels = torch.stack(labels).to(self.device)
            return text, image, labels

        return collate_fn


class MySampler(Sampler):
    def __init__(self, indices, shuffle):
        super().__init__(None)
        self.indices = indices
        if not torch.is_tensor(self.indices):
            self.indices = torch.tensor(self.indices, dtype=torch.long)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            indices = self.indices[torch.randperm(self.indices.shape[0])]
        else:
            indices = self.indices
        for item in indices:
            yield item

    def __len__(self):
        return len(self.indices)

