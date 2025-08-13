import torch
from dataset import EncodeDataset
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm


@torch.no_grad()
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'openai/clip-vit-large-patch14'
    model = CLIPModel.from_pretrained(model_path).to(device)
    processor = CLIPProcessor.from_pretrained(model_path)
    for dataset_name in ['Pheme', 'Fakeddit', 'FakeNewsNet', 'FineFake', 'MMFakeBench']:
        dataset = EncodeDataset(dataset_name)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=dataset.get_collate_fn())
        text_reps = []
        image_reps = []
        all_labels = []
        for batch in tqdm(loader, desc=dataset_name):
            text, image, labels = batch
            inputs = processor(text=text, images=image, return_tensors='pt',
                               truncation=True, padding=True, max_length=77).to(device)
            outputs = model(**inputs)
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds
            text_reps.append(text_features.to('cpu'))
            image_reps.append(image_features.to('cpu'))
            all_labels.append(torch.tensor(labels, dtype=torch.long).to('cpu'))
        text_reps = torch.cat(text_reps, dim=0)
        image_reps = torch.cat(image_reps, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        torch.save([text_reps, image_reps, all_labels], f'data/{dataset_name}.pt')


if __name__ == '__main__':
    main()
