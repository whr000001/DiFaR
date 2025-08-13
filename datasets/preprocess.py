import json
import os
import random
import pandas
from PIL import Image
import pickle


def sample_fakeddit():
    random.seed(20250501)
    data = json.load(open('../raw_data/Fakeddit/cleaned_data.json'))
    res = []
    for item in data:
        instance_id = item['id']
        try:
            image = Image.open(f'../raw_data/Fakeddit/images/{instance_id}.jpg')
        except Exception as e:
            continue
        res.append(item)
    random.shuffle(res)
    real, fake = [], []
    for item in res:
        instance_id = item['id']
        item['label'] = item['2_way_label']
        candidate = fake if item['label'] == 1 else real
        candidate.append(item)
        item['image_name'] = f'{instance_id}.jpg'
    res = real[:500] + fake[:500]
    random.shuffle(res)
    for item in res:
        if item['label'] == 0:
            item['label'] = 1
        else:
            item['label'] = 0
    json.dump(res, open('Fakeddit/data.json', 'w'))
    for item in res:
        instance_id = item['id']
        cmd = f'cp ../raw_data/Fakeddit/images/{instance_id}.jpg Fakeddit/images/'
        os.system(cmd)


def sample_cosmos():
    random.seed(20250501)
    data = []
    with open('../raw_data/COSMOS/test_data.json') as f:
        for line in f.readlines():
            data.append(json.loads(line))
    res = []
    for item in data:
        try:
            image = Image.open('../raw_data/COSMOS/{}'.format(item['img_local_path']))
        except Exception as e:
            continue
        res.append(item)
    random.shuffle(res)
    real, fake = [], []
    for item in res:
        item['text'] = item['caption1']
        item['label'] = item['context_label']
        candidate = fake if item['label'] == 1 else real
        candidate.append(item)
        item['image_name'] = item['img_local_path'].split('/')[-1]
    res = real[:500] + fake[:500]
    random.shuffle(res)
    json.dump(res, open('COSMOS/data.json', 'w'))
    for item in res:
        cmd = 'cp ../raw_data/COSMOS/{} COSMOS/images/{}'.format(item['img_local_path'], item['image_name'])
        os.system(cmd)


def sample_mmfakebench():
    random.seed(20250501)
    test_data = json.load(open('../raw_data/MMFakeBench/MMFakeBench_test.json'))
    for item in test_data:
        item['image_path'] = '../raw_data/MMFakeBench/MMFakeBench_test/{}'.format(item['image_path'])
    val_data = json.load(open('../raw_data/MMFakeBench/MMFakeBench_val.json'))
    for item in val_data:
        item['image_path'] = '../raw_data/MMFakeBench/MMFakeBench_val/{}'.format(item['image_path'])
    data = test_data + val_data
    random.shuffle(data)
    res = []
    for item in data:
        try:
            image = Image.open(item['image_path'])
        except Exception as e:
            continue
        res.append(item)
    print(len(res))
    real, fake = [], []
    for item in res:
        item['label'] = int(item['gt_answers'] == 'Fake')
        candidate = fake if item['label'] == 1 else real
        candidate.append(item)
    print(len(real), len(fake))
    res = real[:500] + fake[:500]
    random.shuffle(res)
    for index, item in enumerate(res):
        image_path = item['image_path']
        item['image_name'] = '{}.{}'.format(index, image_path.split('.')[-1])
        cmd = 'cp {} MMFakeBench/images/{}'.format(item['image_path'], item['image_name'])
        os.system(cmd)
    json.dump(res, open('MMFakeBench/data.json', 'w'))


def sample_pheme():
    random.seed(20250501)
    data = []
    domains = ['charliehebdo', 'ferguson', 'germanwings-crash', 'ottawashooting', 'sydneysiege']
    for domain in domains:
        for cate in ['rumours', 'non-rumours']:
            path = f'../raw_data/Pheme/pheme-rnr-dataset/{domain}/{cate}'
            names = sorted(os.listdir(path))
            res = []
            for item in names:
                each = json.load(open(f'{path}/{item}/source-tweet/{item}.json'))
                image_path_0 = f'../raw_data/Pheme/image/{item}.jpg'
                image_path_1 = f'../raw_data/Pheme/image_new/{item}.jpg'
                raw_image_path = None
                if os.path.exists(image_path_0):
                    try:
                        image = Image.open(image_path_0)
                        raw_image_path = image_path_0
                    except:
                        continue
                elif os.path.exists(image_path_1):
                    try:
                        image = Image.open(image_path_1)
                        raw_image_path = image_path_1
                    except:
                        continue
                else:
                    continue
                res.append({
                    'text': each['text'],
                    'label': int(cate == 'rumours'),
                    'image_name': f'{item}.jpg',
                    'raw_image_path': raw_image_path
                })
            random.shuffle(res)
            data += res[:100]
    random.shuffle(data)
    json.dump(data, open('Pheme/data.json', 'w'))
    for item in data:
        cmd = 'cp {} Pheme/images/{}'.format(item['raw_image_path'], item['image_name'])
        os.system(cmd)


def sample_fakenewsnet():
    random.seed(20250501)
    out = []
    for domain in ['gossip', 'politi']:
        data = []
        train_data = pandas.read_csv(f'../raw_data/FakeNewsNet/{domain}_train.csv')
        for index, item in train_data.iterrows():
            data.append({
                'text': item['content'],
                'raw_image_path': '../raw_data/FakeNewsNet/Images/{}_train/{}'.format(domain, item['image']),
                'label': item['label'],
                'image_name': item['image']
            })
        test_data = pandas.read_csv(f'../raw_data/FakeNewsNet/{domain}_test.csv')
        for index, item in test_data.iterrows():
            data.append({
                'text': item['content'],
                'raw_image_path': '../raw_data/FakeNewsNet/Images/{}_test/{}'.format(domain, item['image']),
                'label': item['label'],
                'image_name': item['image']
            })
        res = []
        for item in data:
            try:
                image = Image.open(item['raw_image_path'])
            except:
                continue
            res.append(item)
        random.shuffle(res)
        res = res[:500]
        out += res
    random.shuffle(out)
    for item in out:
        if item['label'] == 0:
            item['label'] = 1
        else:
            item['label'] = 0
    json.dump(out, open('FakeNewsNet/data.json', 'w'))
    for item in out:
        cmd = 'cp {} FakeNewsNet/images/{}'.format(item['raw_image_path'], item['image_name'])
        os.system(cmd)


def sample_finefake():
    random.seed(20250501)

    # # save pkl file as json file
    # with open('../raw_data/FineFake/FineFake.pkl', "rb") as f:
    #     data_df = pickle.load(f)  # data_df is in dataframe
    # json_data = []
    # for index, item in data_df.iterrows():
    #     if not isinstance(item['text'], str):
    #         continue
    #     json_data.append({
    #         'text': item['text'],
    #         'image_path': item['image_path'],
    #         'label': 1 - item['label'],
    #         'fine-grained label': item['fine-grained label']
    #     })
    # print(len(json_data))
    # json.dump(json_data, open('../raw_data/FineFake/json_data.json', 'w', encoding='utf-8'))

    data = json.load(open('../raw_data/FineFake/json_data.json'))
    res = {}
    for item in data:
        path = '../raw_data/FineFake/{}'.format(item['image_path'])
        try:
            image = Image.open(path)
        except Exception as e:
            continue
        fine_label = item['fine-grained label']
        if fine_label not in res:
            res[fine_label] = []
        res[fine_label].append(item)
    out = []
    for label in [0, 1, 2, 3, 4, 5]:
        each = res[label]
        random.shuffle(each)
        sample_size = 500 if label == 0 else 100
        out += each[:sample_size]
    random.shuffle(out)
    for item in out:
        path = '../raw_data/FineFake/{}'.format(item['image_path'])
        image_name = '_'.join(item['image_path'].split('/'))
        item['image_name'] = image_name
        cmd = 'cp {} FineFake/images/{}'.format(path, item['image_name'])
        os.system(cmd)
    json.dump(out, open('FineFake/data.json', 'w', encoding='utf-8'))


def multi_fold():
    random.seed(20250501)
    dataset_names = ['Pheme', 'Fakeddit', 'FakeNewsNet', 'FineFake', 'MMFakeBench']
    for dataset in dataset_names:
        data = json.load(open(f'{dataset}/data.json'))
        indices = list(range(len(data)))
        random.shuffle(indices)
        fold_cnt = 5
        fold_size = (len(indices) + fold_cnt - 1) // fold_cnt
        folds = []
        for fold_index in range(fold_cnt):
            each = indices[fold_index * fold_size: (fold_index + 1) * fold_size]
            folds.append(each)
        json.dump(folds, open(f'{dataset}/folds.json', 'w'))


def main():
    # sample_fakeddit()
    # sample_cosmos()
    # sample_mmfakebench()
    # sample_pheme()
    # sample_fakenewsnet()
    # sample_finefake()
    multi_fold()


if __name__ == '__main__':
    main()
