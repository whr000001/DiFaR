import json
from tqdm import tqdm
from nltk import sent_tokenize
import wikipedia as wp
import os
import time


os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''  # fill your proxy if available


def factuality(text):
    docs = [""]
    try:
        for ent in wp.search(text[:100], results=3):
            try:
                docs.append(wp.summary(ent, sentences=5))
            except:
                pass
    except:
        time.sleep(2)
        pass
    return docs


def main():
    dataset_names = ['Fakeddit', 'FakeNewsNet', 'FineFake', 'MMFakeBench']
    for dataset_name in dataset_names:
        data = json.load(open(f'../diversity/multiple/{dataset_name}.json'))
        save_path = f'knowledge_base/{dataset_name}.json'
        if os.path.exists(save_path):
            res = json.load(open(save_path))
        else:
            res = []
        for index, item in enumerate(tqdm(data)):
            if index < len(res):
                continue
            sents = sent_tokenize(item)
            each = []
            for sent in sents:
                each.append(factuality(sent))
            res.append(each)
            json.dump(res, open(save_path, 'w'))


if __name__ == '__main__':
    main()
