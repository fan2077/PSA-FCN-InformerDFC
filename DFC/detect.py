import os
import sys
import torch
from torch import nn
from torch.utils.data import Dataset
import scipy.io as sio
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import Informer_nodis

mat_label={"N":0,"Y":1}

class Matdataset(Dataset):
    def __init__(self, data_dir):
        self.label_name = {"N": 0, "Y": 1}
        self.data_info = self.get_mat_info(data_dir)

    def __getitem__(self, index):
        path_mat, label = self.data_info[index]
        matdata = sio.loadmat(path_mat)
        data = matdata['augmented_data1']
        mat = torch.Tensor(data)
        mat = mat.transpose(1, 0)
        return mat, label

    def __len__(self):
        return len(self.data_info)

    def get_mat_info(self, data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            for sub_dir in dirs:
                mat_names = os.listdir(os.path.join(root, sub_dir))
                mat_names = list(filter(lambda x: x.endswith('.mat'), mat_names))
                for i in range(len(mat_names)):
                    mat_name = mat_names[i]
                    path_mat = os.path.join(root, sub_dir, mat_name)
                    label = mat_label[sub_dir]
                    data_info.append((path_mat, int(label)))
        return data_info


def evaluate_model(loader, model, device):
    all_labels = []
    all_predictions = []

    model.eval()
    with torch.no_grad():
        for mats, labels in tqdm(loader, file=sys.stdout):
            outputs, _ = model(mats.to(device))
            predictions = torch.max(outputs, dim=1)[1]
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    return accuracy, precision, recall, f1


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_root = "D:/DFCdata"
    mat_path = os.path.join(data_root, "DFC-40-6o")
    assert os.path.exists(mat_path), "{} path does not exist.".format(mat_path)

    val_data = os.path.join(mat_path, 'train')
    val_dataset = Matdataset(val_data)

    batch_size = 1
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size, shuffle=False,
                                             num_workers=nw)

    model = Informer_nodis.Informerencoder().to(device)
    model_weight_path = "Informer40-6-2.pth"
    print(f"Loading model weights from {model_weight_path}")
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.linear = nn.Sequential()

    accuracy, precision, recall, f1 = evaluate_model(val_loader, model, device)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")


if __name__ == '__main__':
    main()
