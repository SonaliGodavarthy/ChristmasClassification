import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class ChristmasImages(Dataset) :

    def __init__(self, path, training=True):
        super().__init__()
        self.training = training
        # If training == True, path contains subfolders
        # containing images of the corresponding classes
        # If training == False, path directly contains
        # the test images for testing the classifier
        
        # The path to the dataset
        self.path = path
        
        # The transformations to be applied to the images
        self.transform = transforms.Compose([transforms.Resize((224,224)),
                                             transforms.RandomHorizontalFlip(p=0.5),
                                             transforms.RandomVerticalFlip(p=0.5),
                                             transforms.ToTensor()])
        
        if(self.training):
            # Creating an ImageFolder dataset
            self.data = datasets.ImageFolder(self.path,transform=self.transform)
        
        else:
            # Directly loading the images from the path
            # Getting the paths of all images in the test directory
            image_paths = [self.path+'/'+ i for i in os.listdir(self.path)]
            # Applying transformations to each image and stack them into a tensor
            image_list = [self.transform(Image.open(image).convert('RGB')) for image in image_paths]
            self.data = torch.stack(image_list,dim=0)
                

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # If self.training == False, output (image, )
        # where image will be used as input for your model
        
        if(self.training):
            # Returning the image and its corresponding label
            return (self.data[index][0], self.data[index][1]) 
        else:
            # Returning only the transformed image
            return (self.data[index],) 