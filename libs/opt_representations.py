from models.classifier_module import classfier_module
from libs.dataset import dataset
from torch.utils.data import Dataset, DataLoader
import torch
from typing import Tuple
from tqdm import tqdm

def get_representations(
        model: classfier_module,
        data: Dataset
) -> Tuple[torch.Tensor, list]:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.eval()
    model.to(device)
    dataloader = DataLoader(data,
                            batch_size=32,
                            num_workers=0,
                            collate_fn = dataset.batch_collector_)
    re_tensors = []
    labels = []
    with tqdm(dataloader, total=len(data)//32) as batches:
        for inputs in batches:
            representation = model.get_representation(
                input_ids= inputs['input_ids'].to(device),
                attention_mask= inputs['attention_mask'].to(device)
            ).cpu()
            labels.extend(inputs['labels'].tolist())
            re_tensors.append(representation)
    return torch.cat(re_tensors, dim=0), labels
