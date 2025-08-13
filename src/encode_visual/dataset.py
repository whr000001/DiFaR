import json
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler


class EncodeDataset(Dataset):
    def __init__(self, dataset_name):
        data = json.load(open(f'../../datasets/{dataset_name}/data.json'))
        self.images = []
        self.text = []
        self.labels = []
        for item in data:
            self.text.append(item['text'])
            self.labels.append(item['label'])
            self.images.append('../../datasets/{}/images/{}'.format(dataset_name, item['image_name']))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.text[index], Image.open(self.images[index]), self.labels[index]

    def get_collate_fn(self):
        def collate_fn(batch):
            return zip(*batch)
        return collate_fn
