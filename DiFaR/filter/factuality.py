import json
import torch
import numpy as np
from nltk import sent_tokenize
from transformers import AutoTokenizer, pipeline
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer1 = AutoTokenizer.from_pretrained('tals/albert-xlarge-vitaminc-mnli')
tokenizer2 = AutoTokenizer.from_pretrained('FacebookAI/roberta-base')
vitaminc_model = pipeline('text-classification', model='tals/albert-xlarge-vitaminc-mnli',
                          tokenizer=tokenizer1, return_all_scores=True, max_length=512, truncation=True, device=device)
factkb_model = pipeline('text-classification', model='bunsenfeng/FactKB', tokenizer=tokenizer2,
                        device=device, return_all_scores=True, max_length=512, truncation=True)


@torch.no_grad()
def main():
    dataset_names = ['Fakeddit', 'FakeNewsNet', 'FineFake', 'MMFakeBench']
    for dataset_name in dataset_names:
        explanation = json.load(open(f'../diversity/multiple/{dataset_name}.json'))
        wiki_data = json.load(open(f'knowledge_base/{dataset_name}.json'))
        out = []
        for item, wiki in tqdm(zip(explanation, wiki_data), total=len(explanation)):
            sents = sent_tokenize(item)
            factuality_score = []
            for sent, docs in zip(sents, wiki):
                candidate = [sent + ' ' + doc for doc in docs]
                vitaminc_scores = vitaminc_model(candidate)
                factkb_scores = factkb_model(candidate)
                candidate_scores = []
                for i in range(len(docs)):
                    vitaminc_score = (vitaminc_scores[i][0]['score'] - vitaminc_scores[i][1]['score'] + 0 *
                                      vitaminc_scores[i][2]['score'] + 1) / 2  # 0 to 1
                    factkb_score = factkb_scores[i][1]['score']  # 0 to 1
                    candidate_scores.append((vitaminc_score + factkb_score) / 2)
                score = np.max(candidate_scores)
                factuality_score.append(score)
            out.append(factuality_score)
        json.dump(out, open(f'scores/factuality_{dataset_name}.json', 'w'))


if __name__ == '__main__':
    main()
