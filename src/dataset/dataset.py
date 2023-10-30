import os
from torch.utils.data import Dataset
import pandas as pd
import torch
from PIL import ImageFile, Image
from torchvision import transforms as T
from torchvision.transforms import functional as TF

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ClassificationDataset(Dataset):
    def __init__(
        self,
        csv_file,
        root_dir,
        image_dim=(350, 350),
        precropped=False,
        raw=False,
    ):

        self.dataframe = pd.read_csv(csv_file, encoding='utf-16')
        self.root_dir = root_dir
        self.image_dim = image_dim
        self.precropped = precropped
        self.raw = raw

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.dataframe["img_path"][idx])
        image = Image.open(img_name)
        if not self.precropped:
            image= image.crop(( self.dataframe["nozzle_tip_y"][idx]-self.image_dim[0]//2,
                                self.dataframe["nozzle_tip_x"][idx]-self.image_dim[1]//2,
                                self.dataframe["nozzle_tip_y"][idx]+self.image_dim[0]//2,
                                self.dataframe["nozzle_tip_x"][idx]+self.image_dim[1]//2))

        image = T.ToTensor()(image)

        y = self.get_labels(idx)
        sample = (image, y, self.dataframe["img_path"][idx])
        return sample

    def get_labels(self, idx):
        target = int(self.dataframe["flow_rate_class"][idx])
        y = torch.tensor(target, dtype=torch.long).unsqueeze(0)
        return y, idx

class RegressionDataset(ClassificationDataset):
    def __init__(
        self,
        csv_file,
        root_dir,
        image_dim=(350, 350),
        precropped=True,
        raw=False,
    ):
        super().__init__(
            csv_file=csv_file,
            root_dir=root_dir,
            image_dim=image_dim,
            precropped=precropped,
            raw=raw,
        )

    def get_labels(self, idx):
        target = float(self.dataframe["flow_rate"][idx])
        y = torch.tensor(target, dtype=torch.float).unsqueeze(0)
        if self.raw:
            y = torch.log(y / 100)
        return y
