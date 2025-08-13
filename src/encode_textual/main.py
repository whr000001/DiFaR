import json
from torch.utils.data import Dataset
from transformers import DebertaV2Model, DebertaV2Tokenizer
from tqdm import tqdm
import torch
from nltk import sent_tokenize


class TextEncoder:
    def __init__(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        lm_path = 'microsoft/deberta-v3-large'
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(lm_path)
        self.encoder = DebertaV2Model.from_pretrained(lm_path).to(device)
        self.device = device

    @torch.no_grad()
    def encode_single(self, text):
        sents = sent_tokenize(text)
        batch_size = 32
        batch_cnt = (len(sents) + batch_size - 1) // batch_size
        text_reps = []
        for _ in range(batch_cnt):
            batch_sent = sents[_ * batch_size: (_ + 1) * batch_size]
            input_tensor = self.tokenizer.batch_encode_plus(batch_sent, return_tensors='pt', padding=True,
                                                            max_length=512, truncation=True).to(self.device)
            out = self.encoder(**input_tensor)
            reps = out.last_hidden_state
            attention_mask = input_tensor['attention_mask']
            reps = torch.einsum('ijk,ij->ijk', reps, attention_mask)
            reps = torch.sum(reps, dim=1)
            attention_mask = torch.sum(attention_mask, dim=1).unsqueeze(-1)
            reps = reps / attention_mask
            text_reps.append(reps)
        text_reps = torch.cat(text_reps, dim=0)
        text_reps = torch.mean(text_reps, dim=0).to('cpu')
        return text_reps

    def __call__(self, texts):
        reps = []
        for text in tqdm(texts):
            reps.append(self.encode_single(text))
        reps = torch.stack(reps)
        return reps


def main():
    encoder = TextEncoder()
    for dataset_name in ['FineFake', 'FakeNewsNet', 'Fakeddit', 'MMFakeBench']:
        data = json.load(open(f'../../datasets/{dataset_name}/data.json'))
        explanation = json.load(open(f'../../DiFaR/rationales/{dataset_name}.json'))
        inputs = []
        for index in range(len(data)):
            text = data[index]['text'] + ' ' + explanation[index]
            inputs.append(text)
        reps = encoder(inputs)
        torch.save(reps, f'data/{dataset_name}.pt')


if __name__ == '__main__':
    main()
