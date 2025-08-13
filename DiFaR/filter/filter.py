import json
import numpy as np
from nltk import sent_tokenize


def main():
    dataset_names = ['Fakeddit', 'FakeNewsNet', 'FineFake', 'MMFakeBench']
    for dataset_name in dataset_names:
        explanation_data = json.load(open(f'../diversity/multiple/{dataset_name}.json'))
        factuality_scores = json.load(open(f'scores/factuality_{dataset_name}.json'))
        similarity_scores = json.load(open(f'scores/similarity_{dataset_name}.json'))
        out = []
        for exp, factuality, similarity in zip(explanation_data, factuality_scores, similarity_scores):
            sents = sent_tokenize(exp)
            if len(sents) == 0:
                out.append(' ')
                continue
            f_lim = np.percentile(factuality, 50)
            s_lim = np.percentile(similarity, 50)
            filtered_exp = ''
            for index, (sent, each_f, each_s) in enumerate(zip(sents, factuality, similarity)):
                if index != 0 and each_f < f_lim and each_s < s_lim:
                    continue
                filtered_exp += sent + '\n'
            out.append(filtered_exp)
        json.dump(out, open(f'filtered/{dataset_name}.json', 'w'))


if __name__ == '__main__':
    main()
