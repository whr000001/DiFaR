import json
from nltk import sent_tokenize
from transformers import pipeline, AutoTokenizer
import torch
import numpy as np
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name_or_path = "microsoft/mpnet-base"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
sim_model = pipeline('feature-extraction', tokenizer=tokenizer, model=model_name_or_path,
                     device=device, max_length=512, truncation=True)


def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def main():
    dataset_names = ['Fakeddit', 'FakeNewsNet', 'FineFake', 'MMFakeBench']
    for dataset_name in dataset_names:
        data = json.load(open(f'../../datasets/{dataset_name}/data.json'))
        explanation = json.load(open(f'../diversity/multiple/{dataset_name}.json'))
        out = []
        for sample, item in tqdm(zip(data, explanation), total=len(data)):
            sample = sample['text']
            sents = sent_tokenize(item)
            inputs = [sample] + sents
            features = sim_model(inputs)
            query_feature = np.mean(features[0], axis=1)[0]

            text_features = []
            for i in range(1, len(features)):
                text_features.append(np.mean(features[i], axis=1)[0])

            scores = []
            for i in range(len(sents)):
                scores.append(cosine_similarity(query_feature, text_features[i]))
            out.append(scores)
        json.dump(out, open(f'scores/relevance_{dataset_name}.json', 'w'))


if __name__ == '__main__':
    main()
