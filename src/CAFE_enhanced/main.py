import os
import json
import copy
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import MyDataset, MySampler
from model import SimilarityModule, DetectionModule
from argparse import ArgumentParser
from sklearn.metrics import accuracy_score, f1_score

parser = ArgumentParser()
parser.add_argument('--dataset', type=str)
args = parser.parse_args()

dataset_name = args.dataset
assert dataset_name in ['Fakeddit', 'FakeNewsNet', 'FineFake', 'MMFakeBench']

lr = 1e-3
l2 = 1e-5


def prepare_data(text, image, label):
    nr_index = [i for i, la in enumerate(label) if la == 1]
    if len(nr_index) == 1:
        nr_index = [nr_index[0], nr_index[0]]  # for batch_size == 1
    text_nr = text[nr_index]
    image_nr = image[nr_index]
    fixed_text = copy.deepcopy(text_nr)
    matched_image = copy.deepcopy(image_nr)
    unmatched_image = copy.deepcopy(image_nr).roll(shifts=3, dims=0)
    return fixed_text, matched_image, unmatched_image


def train_one_epoch(similarity_module, detection_module, optim_task_similarity, optim_task_detection, loader, device):
    loss_func_similarity = torch.nn.CosineEmbeddingLoss()
    loss_func_detection = torch.nn.CrossEntropyLoss()

    similarity_module.train()
    detection_module.train()

    all_preds, all_truth = [], []

    for i, (text, image, label) in enumerate(loader):
        text = text.to(device)
        image = image.to(device)
        label = label.to(device)

        fixed_text, matched_image, unmatched_image = prepare_data(text, image, label)
        fixed_text.to(device)
        matched_image.to(device)
        unmatched_image.to(device)

        text_aligned_match, image_aligned_match, pred_similarity_match = similarity_module(fixed_text, matched_image)
        text_aligned_unmatch, image_aligned_unmatch, pred_similarity_unmatch = similarity_module(fixed_text,
                                                                                                 unmatched_image)
        similarity_pred = torch.cat([pred_similarity_match.argmax(1), pred_similarity_unmatch.argmax(1)], dim=0)
        similarity_label_0 = torch.cat(
            [torch.ones(pred_similarity_match.shape[0]), torch.zeros(pred_similarity_unmatch.shape[0])], dim=0).to(
            device)
        similarity_label_1 = torch.cat(
            [torch.ones(pred_similarity_match.shape[0]), -1 * torch.ones(pred_similarity_unmatch.shape[0])], dim=0).to(
            device)

        text_aligned_4_task1 = torch.cat([text_aligned_match, text_aligned_unmatch], dim=0)
        image_aligned_4_task1 = torch.cat([image_aligned_match, image_aligned_unmatch], dim=0)
        loss_similarity = loss_func_similarity(text_aligned_4_task1, image_aligned_4_task1, similarity_label_1)

        optim_task_similarity.zero_grad()
        loss_similarity.backward()
        optim_task_similarity.step()

        # corrects_pre_similarity += similarity_pred.eq(similarity_label_0).sum().item()

        # ---  TASK2 Detection  ---

        text_aligned, image_aligned, _ = similarity_module(text, image)
        pre_detection = detection_module(text, image, text_aligned, image_aligned)

        loss_detection = loss_func_detection(pre_detection, label)

        optim_task_detection.zero_grad()
        loss_detection.backward()
        optim_task_detection.step()

        pre_label_detection = pre_detection.argmax(1)
        all_preds.append(pre_label_detection.to('cpu'))
        all_truth.append(label.to('cpu'))

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_truth = torch.cat(all_truth, dim=0).numpy()
    return 0, accuracy_score(all_truth, all_preds), \
        f1_score(all_truth, all_preds, average='micro'), f1_score(all_truth, all_preds, average='macro')


@torch.no_grad()
def validate(similarity_module, detection_module, loader, device):
    similarity_module.eval()
    detection_module.eval()

    # loss_func_detection = torch.nn.CrossEntropyLoss()
    # loss_func_similarity = torch.nn.CosineEmbeddingLoss()

    all_truth = []
    all_preds = []
    all_score = []

    for i, (text, image, label) in enumerate(loader):
        text = text.to(device)
        image = image.to(device)
        label = label.to(device)

        fixed_text, matched_image, unmatched_image = prepare_data(text, image, label)
        fixed_text.to(device)
        matched_image.to(device)
        unmatched_image.to(device)

        # ---  TASK1 Similarity  ---

        text_aligned_match, image_aligned_match, pred_similarity_match = similarity_module(fixed_text,
                                                                                           matched_image)
        text_aligned_unmatch, image_aligned_unmatch, pred_similarity_unmatch = similarity_module(fixed_text,
                                                                                                 unmatched_image)
        similarity_pred = torch.cat([pred_similarity_match.argmax(1), pred_similarity_unmatch.argmax(1)], dim=0)
        similarity_label_0 = torch.cat(
            [torch.ones(pred_similarity_match.shape[0]), torch.zeros(pred_similarity_unmatch.shape[0])], dim=0).to(
            device)
        similarity_label_1 = torch.cat(
            [torch.ones(pred_similarity_match.shape[0]), -1 * torch.ones(pred_similarity_unmatch.shape[0])],
            dim=0).to(device)

        text_aligned_4_task1 = torch.cat([text_aligned_match, text_aligned_unmatch], dim=0)
        image_aligned_4_task1 = torch.cat([image_aligned_match, image_aligned_unmatch], dim=0)
        # loss_similarity = loss_func_similarity(text_aligned_4_task1, image_aligned_4_task1, similarity_label_1)

        # ---  TASK2 Detection  ---

        text_aligned, image_aligned, _ = similarity_module(text, image)
        pre_detection = detection_module(text, image, text_aligned, image_aligned)
        # loss_detection = loss_func_detection(pre_detection, label)
        pre_label_detection = pre_detection.argmax(1)

        all_truth.append(label.to('cpu'))
        all_preds.append(pre_label_detection.to('cpu'))
        score = torch.softmax(pre_detection, dim=-1).to('cpu')
        all_score.append(score)

    all_preds = torch.cat(all_preds, dim=0).numpy().tolist()
    all_truth = torch.cat(all_truth, dim=0).numpy().tolist()
    all_score = torch.cat(all_score, dim=0).numpy().tolist()
    return 0, accuracy_score(all_truth, all_preds), \
        f1_score(all_truth, all_preds, average='micro'), f1_score(all_truth, all_preds, average='macro'), \
        all_truth, all_preds, all_score


def train(train_loader, val_loader, test_loader, device):
    similarity_module = SimilarityModule().to(device)
    detection_module = DetectionModule().to(device)
    optim_task_similarity = torch.optim.Adam(
        similarity_module.parameters(), lr=lr, weight_decay=l2
    )  # also called task1
    optim_task_detection = torch.optim.Adam(
        detection_module.parameters(), lr=lr, weight_decay=l2
    )  # also called task2

    best_metrics = 0
    best_similarity_state = similarity_module.state_dict()
    for key, value in best_similarity_state.items():
        best_similarity_state[key] = value.clone()
    best_detection_state = detection_module.state_dict()
    for key, value in best_detection_state.items():
        best_detection_state[key] = value.clone()
    no_up_limits = 20
    no_up = 0
    pbar = tqdm(range(200), leave=False)

    for epoch in pbar:
        train_loss, train_acc, train_micro, train_macro = train_one_epoch(similarity_module, detection_module,
                                                                          optim_task_similarity, optim_task_detection,
                                                                          train_loader, device)
        val_loss, val_acc, val_micro, val_macro, _, _, _ = validate(similarity_module, detection_module,
                                                                    val_loader, device)
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
            for key, value in similarity_module.state_dict().items():
                best_similarity_state[key] = value.clone()
            for key, value in detection_module.state_dict().items():
                best_detection_state[key] = value.clone()
            no_up = 0
        else:
            no_up += 1
        if no_up >= no_up_limits:
            break
    similarity_module.load_state_dict(best_similarity_state)
    detection_module.load_state_dict(best_detection_state)
    test_loss, test_acc, test_micro, test_macro, all_truth, all_preds, all_score = validate(similarity_module,
                                                                                            detection_module,
                                                                                            test_loader, device)
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

        train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler,
                                  collate_fn=dataset.get_collate_fn(),
                                  drop_last=True)  # for the possibility of batch_size == 1
        val_loader = DataLoader(dataset, batch_size=32, sampler=val_sampler, collate_fn=dataset.get_collate_fn())

        fold_index = str(fold_index)
        for _ in range(5):
            acc, preds, score, truth = train(train_loader, val_loader, val_loader, device)
            if fold_index not in exp:
                exp[fold_index] = (acc, preds, score, truth)
            else:
                if acc > exp[fold_index][0]:
                    exp[fold_index] = (acc, preds, score, truth)
    json.dump(exp, open(exp_path, 'w'))


if __name__ == "__main__":
    main()
