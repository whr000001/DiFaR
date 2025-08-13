import json
import os
from tqdm import tqdm
from utils import construct_length, encode_image, obtain_response


def main():
    for dataset_name in ['Fakeddit', 'FakeNewsNet', 'FineFake', 'MMFakeBench']:
        data = json.load(open(f'../../datasets/{dataset_name}/data.json'))
        save_path = f'explanation/{dataset_name}_object.json'
        if os.path.exists(save_path):
            out = json.load(open(save_path))
        else:
            out = []
        for item in tqdm(data[len(out):], desc=dataset_name):
            text = construct_length(item['text'])
            image = encode_image('../../datasets/{}/images/{}'.format(dataset_name, item['image_name']))
            inst = 'news text: {}'.format(text)
            inst += 'Please analyze the object that appears in the image of this piece of news.'
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": inst},
                        {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image}'}},
                    ],
                },
            ]
            analysis = obtain_response(conversation)
            inst = 'Based on the analysis, determine whether this news with text and image is fake or real. ' \
                   'Meanwhile, provide a comprehensive explanation.'
            conversation += [
                {
                    'role': 'assistant',
                    'content': [{'type': 'text', 'text': analysis}]
                },
                {
                    'role': 'user',
                    'content': [{'type': 'text', 'text': inst}]
                }
            ]
            explain = obtain_response(conversation)
            out.append([analysis, explain])
            json.dump(out, open(save_path, 'w'))


if __name__ == '__main__':
    main()
