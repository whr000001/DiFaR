import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.text_fc = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        self.image_fc = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        self.cls = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 2)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, data):
        text, image, labels = data
        batch_size = len(labels)
        text_reps = self.text_fc(text)
        image_reps = self.image_fc(image)
        reps = self.fc(torch.cat([text_reps, image_reps], dim=-1))

        pred = self.cls(reps)
        loss = self.loss_fn(pred, labels)

        return pred, loss, labels, batch_size
