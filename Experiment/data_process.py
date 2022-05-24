import torch

from noise_utils import *
from torchvision import datasets, transforms as T

def select_noise(args):
    """A simple selection function to decide which type object to return

    Args:
        args (dict): a dictionary containing the configuration file specifications

    Returns:
        ImageNoise: one of the supported noise types
    """
    
    args["noise_type"] = args["noise_type"].lower()
    
    if args["noise_type"] == "gaussian":
        return GaussianNoise(args["mean"], args["std"])
    
    elif args["noise_type"] == "speckle":
        return SpeckleNoise(args["mean"], args["std"])
    
    elif args["noise_type"] == "s&p":
        return SaltPepperNoise(args["amount"])
    
    elif args["noise_type"] == "poisson":
        return PoissonNoise()
    
    elif args["noise_type"] == "eraser":
        return RandomEraser (
            args["init_erasing_prob"],
            args["subarea_low"],
            args["subarea_high"],
            args["aspect_ratio_low"],
            args["aspect_ratio_high"],
            args["attempts"],
            args["cmap"]
        )
        
    else:
        return NoNoise()

class DataProcessor(object):
    def __init__(self, args=None) -> None:
        self.train_dir = args["data_dir"] + '/train'
        self.valid_dir = args["data_dir"] + '/valid'
        self.test_dir = args["data_dir"] + '/test'
        self.T_Noise = select_noise(args)
        
    def load_transform(self):
        """Loads the data and applies the specified transformations on them

        Returns:
            tuple: three dataloaders (train, validation and test)
        """
        train_transform = T.Compose ([
            self.T_Noise,
            T.RandomRotation(30),
            T.Resize(255),
            T.CenterCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        valid_transform = T.Compose ([
            T.Resize(255),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        test_transform = T.Compose ([
            T.Resize(255),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load the datasets with ImageFolder
        train_data = datasets.ImageFolder(self.train_dir, transform=train_transform)
        valid_data = datasets.ImageFolder(self.valid_dir, transform=valid_transform)
        test_data = datasets.ImageFolder(self.test_dir, transform=test_transform)

        # Using the image datasets, define the dataloaders
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
        validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

        return trainloader, validloader, testloader