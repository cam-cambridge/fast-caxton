import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class Preprocess(nn.Module):
    def __init__(self, test):
        super().__init__()

        self.mean = [0.16963361, 0.1761256, 0.16955556]
        self.std = [0.27373913, 0.27497604, 0.27459216]

        if test:
            self.transforms = T.Compose([
                T.Lambda(lambda crop: TF.crop(crop,100,85,180,180)),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),     
            ])
        else:
            self.transforms = T.Compose([
                T.RandomAffine(degrees=30, scale=(0.95, 1.05), translate=(0.03,0.03)),
                T.Lambda(lambda crop: TF.crop(crop,100,85,180,180)),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                T.RandomHorizontalFlip(0.5),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),     
            ])

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x):
        x = self.transforms(x)
        return x
    
class TestAugmentation(nn.Module):

    def __init__(self):
        super().__init__()

        self.mean = [0.16963361, 0.1761256, 0.16955556]
        self.std = [0.27373913, 0.27497604, 0.27459216]

        self.transforms = T.Compose(
            [
                T.Normalize(self.mean, self.std),
                T.Lambda(lambda crop: TF.crop(crop,90,85,200,200)),
                T.FiveCrop(180),
                T.Lambda(
                    lambda crops: torch.concat(
                        [torch.stack([crop, TF.hflip(crop)]) for crop in crops]
                    )
                ),
            ]
        )

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x):
        x = self.transforms(x)  # n, b, c, h, w
        x = torch.swapaxes(x, 0, 1)  # b, n, c, h, w
        return x

class TrainAugmentation(nn.Module):

    def __init__(self):
        super().__init__()

        self.mean = [0.16963361, 0.1761256, 0.16955556]
        self.std = [0.27373913, 0.27497604, 0.27459216]

        self.transforms = T.Compose([
                T.RandomAffine(degrees=30, scale=(0.95, 1.05), translate=(0.03,0.03)),
                T.Lambda(lambda crop: TF.crop(crop,100,85,180,180)),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                T.RandomHorizontalFlip(0.5),
                T.Normalize(self.mean, self.std),
            ]
        )

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x):
        x = self.transforms(x)  # n, b, c, h, w
        return x

class TransferAugmentation(nn.Module):

    def __init__(self):
        super().__init__()

        self.mean = [0.16963361, 0.1761256, 0.16955556]
        self.std = [0.27373913, 0.27497604, 0.27459216]

        self.transforms = T.Compose([
                T.RandomAffine(degrees=10, scale=(0.95, 1.05), translate=(0.03,0.03)),
                T.Lambda(lambda crop: TF.crop(crop,100,85,180,180)),
                T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
                T.Normalize(self.mean, self.std),
            ]
        )

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x):
        x = self.transforms(x)  # n, b, c, h, w
        return x

class SimpleTestAugmentation(nn.Module):

    def __init__(self):
        super().__init__()

        self.mean = [0.16963361, 0.1761256, 0.16955556]
        self.std = [0.27373913, 0.27497604, 0.27459216]
        
        self.transforms = T.Compose(
            [
                T.Lambda(lambda crop: TF.crop(crop,100,85,180,180)),
                T.Normalize(self.mean, self.std),

            ]
        )

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x):
        x = self.transforms(x)  # n, b, c, h, w
        return x

