import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class Preprocess(nn.Module):
    # Initialize the Preprocess class.
    def __init__(self, test):
        super().__init__()  # Call to the constructor of the parent class (nn.Module).

        # Set the mean and standard deviation for normalization. These values typically
        # correspond to the mean and standard deviation of the dataset used for training.
        self.mean = [0.16963361, 0.1761256, 0.16955556]
        self.std = [0.27373913, 0.27497604, 0.27459216]

        # If in test mode, define a specific set of transformations, otherwise define
        # a different set that includes data augmentation for training.
        if test:
            # For testing, crops the image to a fixed size and normalizes pixel values.
            self.transforms = T.Compose([
                T.Lambda(lambda crop: TF.crop(crop, 100, 85, 180, 180)),  # Crop operation.
                T.ToTensor(),  # Convert image to PyTorch tensor.
                T.Normalize(self.mean, self.std),  # Normalize with predefined mean and std.     
            ])
        else:
            # For training, adds data augmentation techniques such as random affine transformations,
            # color jitter, and random horizontal flips, along with the crop and normalization.
            self.transforms = T.Compose([
                T.RandomAffine(degrees=30, scale=(0.95, 1.05), translate=(0.03,0.03)),  # Random affine transformation for data augmentation.
                T.Lambda(lambda crop: TF.crop(crop, 100, 85, 180, 180)),  # Crop operation.
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # Random color jitter for data augmentation.
                T.RandomHorizontalFlip(0.5),  # Random horizontal flip with a 50% probability.
                T.ToTensor(),  # Convert image to PyTorch tensor.
                T.Normalize(self.mean, self.std),  # Normalize with predefined mean and std.     
            ])

    @torch.no_grad()  # Decorator to disable gradient calculations, improving efficiency.
    def forward(self, x):
        # Define the forward pass through the transformation pipeline.
        x = self.transforms(x)  # Apply the transformation to the input x.
        return x
    
class TestAugmentation(nn.Module):
    # Initialize the TestAugmentation class.
    def __init__(self):
        super().__init__()

        # Set the mean and standard deviation for normalization. These values typically
        # correspond to the mean and standard deviation of the dataset used for training.
        self.mean = [0.16963361, 0.1761256, 0.16955556]
        self.std = [0.27373913, 0.27497604, 0.27459216]

        self.transforms = T.Compose(
            [
                T.Normalize(self.mean, self.std),  # Normalize with predefined mean and std.
                T.Lambda(lambda crop: TF.crop(crop, 90, 85, 200, 200)),  # Crop operation.
                T.FiveCrop(180),  # Crop the image into four corners and the center.
                T.Lambda(
                    lambda crops: torch.concat(
                        [torch.stack([crop, TF.hflip(crop)]) for crop in crops]  # Apply a horizontal flip to each crop.
                    )
                ),
            ]
        )

    @torch.no_grad()  # Decorator to disable gradient calculations, improving efficiency.
    def forward(self, x):
        # Define the forward pass through the transformation pipeline.
        x = self.transforms(x)  # n, b, c, h, w
        x = torch.swapaxes(x, 0, 1)  # b, n, c, h, w
        return x

class TrainAugmentation(nn.Module):
    # Initialize the TrainAugmentation class.
    def __init__(self):
        super().__init__()

        # Set the mean and standard deviation for normalization. These values typically
        # correspond to the mean and standard deviation of the dataset used for training.
        self.mean = [0.16963361, 0.1761256, 0.16955556]
        self.std = [0.27373913, 0.27497604, 0.27459216]

        self.transforms = T.Compose([
                T.RandomAffine(degrees=30, scale=(0.95, 1.05), translate=(0.03,0.03)),  # Random affine transformation for data augmentation.
                T.Lambda(lambda crop: TF.crop(crop, 100, 85, 180, 180)),  # Crop operation.
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # Random color jitter for data augmentation.
                T.RandomHorizontalFlip(0.5),  # Random horizontal flip with a 50% probability.
                T.Normalize(self.mean, self.std),  # Normalize with predefined mean and std.
            ]
        )

    @torch.no_grad()  # Decorator to disable gradient calculations, improving efficiency.
    def forward(self, x):
        # Define the forward pass through the transformation pipeline.
        x = self.transforms(x)  # n, b, c, h, w
        return x

class TransferAugmentation(nn.Module):
    # Initialize the TransferAugmentation class.
    def __init__(self):
        super().__init__()

        # Set the mean and standard deviation for normalization. These values typically
        # correspond to the mean and standard deviation of the dataset used for training.
        self.mean = [0.16963361, 0.1761256, 0.16955556]
        self.std = [0.27373913, 0.27497604, 0.27459216]

        self.transforms = T.Compose([
                T.RandomAffine(degrees=10, scale=(0.95, 1.05), translate=(0.03,0.03)),  # Random affine transformation for data augmentation.
                T.Lambda(lambda crop: TF.crop(crop, 100, 85, 180, 180)),  # Crop operation.
                T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),  # Random color jitter for data augmentation.
                T.Normalize(self.mean, self.std),  # Normalize with predefined mean and std.
            ]
        )

    @torch.no_grad()  # Decorator to disable gradient calculations, improving efficiency.
    def forward(self, x):
        # Define the forward pass through the transformation pipeline.
        x = self.transforms(x)  # n, b, c, h, w
        return x

class SimpleTestAugmentation(nn.Module):
    # Initialize the SimpleTestAugmentation class.
    def __init__(self):
        super().__init__()

        # Set the mean and standard deviation for normalization. These values typically
        # correspond to the mean and standard deviation of the dataset used for training.
        self.mean = [0.16963361, 0.1761256, 0.16955556]
        self.std = [0.27373913, 0.27497604, 0.27459216]
        
        self.transforms = T.Compose(
            [
                T.Lambda(lambda crop: TF.crop(crop, 100, 85, 180, 180)),  # Crop operation.
                T.Normalize(self.mean, self.std),  # Normalize with predefined mean and std.
            ]
        )

    @torch.no_grad()  # Decorator to disable gradient calculations, improving efficiency.
    def forward(self, x):
        # Define the forward pass through the transformation pipeline.
        x = self.transforms(x)  # n, b, c, h, w
        return x

