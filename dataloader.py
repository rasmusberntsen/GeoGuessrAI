import os
from matplotlib import pyplot as plt
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class ImageDataset(Dataset):
    def __init__(self,
                 annotations_file='countries.csv',
                 img_dir='images/'):

        self.img_labels = pd.read_csv(
            annotations_file, sep=";")
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.img_dir,
            self.img_labels.iloc[idx, 1]
        )
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 0]
        # transformer = transforms.Compose(
        #     [transforms.Resize((520, 520))]
        # )
        # image = transformer(image)
        return image, label


if __name__ == "__main__":
    dataset = ImageDataset()
    data = iter(dataset)
    img, label = next(data)
    # Plot image
    plt.figure()
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
    plt.title(label)