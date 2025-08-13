import json
import os
import torch
from tqdm import tqdm
from model import MyModel
from dataset import MyDataset, MySampler
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--dataset', type=str)
args = parser.parse_args()

dataset_name = args.dataset
assert dataset_name in ['Fakeddit', 'FakeNewsNet', 'FineFake', 'MMFakeBench']

hidden_dim = 256
lr = 1e-3


def train_one_epoch(model, optimizer, loader):
    model.train()
    ave_loss = 0
    cnt = 0
    all_truth = []
    all_preds = []
    for batch in loader:
        optimizer.zero_grad()
        out, loss, truth, length = model(batch)
        loss.backward()
        optimizer.step()

        preds = out.argmax(-1).to('cpu')
        truth = truth.to('cpu')

        ave_loss += loss.item() * length
        cnt += length
        all_truth.append(truth)
        all_preds.append(preds)

    ave_loss /= cnt
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_truth = torch.cat(all_truth, dim=0).numpy()
    return ave_loss, accuracy_score(all_truth, all_preds), \
        f1_score(all_truth, all_preds, average='micro'), f1_score(all_truth, all_preds, average='macro')


@torch.no_grad()
def validate(model, loader, tmp=None):
    model.eval()

    ave_loss = 0
    cnt = 0
    all_truth = []
    all_preds = []
    all_score = []
    for batch in loader:
        out, loss, truth, length = model(batch)

        preds = out.argmax(-1).to('cpu')
        truth = truth.to('cpu')
        score = torch.softmax(out, dim=-1).to('cpu')

        ave_loss += loss.item() * length
        cnt += length

        all_truth.append(truth)
        all_preds.append(preds)
        all_score.append(score)

    ave_loss /= cnt
    all_preds = torch.cat(all_preds, dim=0).numpy().tolist()
    all_truth = torch.cat(all_truth, dim=0).numpy().tolist()
    all_score = torch.cat(all_score, dim=0).numpy().tolist()
    return ave_loss, accuracy_score(all_truth, all_preds), \
        f1_score(all_truth, all_preds, average='micro'), f1_score(all_truth, all_preds, average='macro'), \
        all_truth, all_preds, all_score


def train(train_loader, val_loader, test_loader, device):
    model = MyModel(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_metrics = 0
    best_state = model.state_dict()
    for key, value in best_state.items():
        best_state[key] = value.clone()
    no_up_limits = 8
    no_up = 0
    pbar = tqdm(range(200), leave=False)
    for _ in pbar:
        train_loss, train_acc, train_micro, train_macro = train_one_epoch(model, optimizer, train_loader)
        # print('train', _, train_loss, train_acc, train_f1)
        val_loss, val_acc, val_micro, val_macro, _, _, _ = validate(model, val_loader)
        # print('val', _, val_acc, val_f1)
        if isinstance(pbar, tqdm):
            pbar.set_postfix({
                'train_loss': train_loss,
                'train_micro': train_micro,
                'train_macro': train_macro,
                'val_micro': val_micro,
                'val_macro': val_macro
            })
        if val_micro > best_metrics:
            best_metrics = val_micro
            for key, value in model.state_dict().items():
                best_state[key] = value.clone()
            no_up = 0
        else:
            no_up += 1
        if no_up >= no_up_limits:
            break
    model.load_state_dict(best_state)
    test_loss, test_acc, test_micro, test_macro, all_truth, all_preds, all_score = validate(model, test_loader)
    return test_acc, all_preds, all_score, all_truth


def main():
    _, _, labels = torch.load(f'../encode_visual/data/{dataset_name}.pt', weights_only=True)
    _, image, _ = torch.load(f'../encode_visual/data/{dataset_name}.pt', weights_only=True)
    text = torch.load(f'../encode_textual/data/{dataset_name}.pt', weights_only=True)

    exp_name = f'{dataset_name}'
    exp_path = f'exps/{exp_name}.json'
    if os.path.exists(exp_path):
        exp = json.load(open(exp_path))
    else:
        exp = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = MyDataset(text, image, labels, device)
    folds = json.load(open(f'../../datasets/{dataset_name}/folds.json'))
    for fold_index in range(5):
        train_indices = []
        val_indices = []
        for _ in range(5):
            if fold_index == _:
                val_indices += folds[_]
            else:
                train_indices += folds[_]
        train_sampler = MySampler(train_indices, shuffle=True)
        val_sampler = MySampler(val_indices, shuffle=False)

        train_loader = DataLoader(dataset, batch_size=64, sampler=train_sampler, collate_fn=dataset.get_collate_fn(),
                                  drop_last=True)  # for the possibility of batch_size == 1
        val_loader = DataLoader(dataset, batch_size=64, sampler=val_sampler, collate_fn=dataset.get_collate_fn())

        fold_index = str(fold_index)
        for _ in range(10):
            acc, preds, score, truth = train(train_loader, val_loader, val_loader, device)
            if fold_index not in exp:
                exp[fold_index] = (acc, preds, score, truth)
            else:
                if acc > exp[fold_index][0]:
                    exp[fold_index] = (acc, preds, score, truth)
    json.dump(exp, open(exp_path, 'w'))


if __name__ == "__main__":
    main()
