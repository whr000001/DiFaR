import json


def combine_multiple():
    dataset_names = ['Fakeddit', 'FakeNewsNet', 'FineFake', 'MMFakeBench']
    experts = ['sentiment', 'propaganda', 'consistency', 'object', 'description']
    for dataset in dataset_names:
        out = []
        expert_data = {}
        for expert in experts:
            expert_data[expert] = json.load(open(f'explanation/{dataset}_{expert}.json'))
        for index in range(len(expert_data['sentiment'])):
            explanation = ''
            for expert in expert_data:
                sample = expert_data[expert][index]
                explanation += '\n' + sample[0] + '\n' + sample[1]
            out.append(explanation)
        json.dump(out, open(f'multiple/{dataset}.json', 'w'))


def main():
    combine_multiple()


if __name__ == '__main__':
    main()
