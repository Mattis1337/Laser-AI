import os

from torch.utils.data import Dataset

# TODO: Adjust the custom datset to actually taking bitboards as input and not pictures:
#  look at what output the transform type ToTensor offers and adjust the bitboards accordingly
class WhiteMovesDataset(Dataset):
    def __init__(self, img_labels, img_dir, transform=None, target_transform=None):
        # Get the file containing all white moves
        self.img_labels = img_labels 
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class BlackMovesDataset(Dataset):
    def __init__(self, img_labels, img_dir, transform=None, target_transform=None):
        # Get the file containing all black moves 
        self.img_labels = img_labels
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
