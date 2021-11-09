from torch.utils.data import DataLoader as dl
from torchvision import datasets, transforms, models

class DataLoader:
    def __init__(self, train_data_url, test_data_url):
        self.batch_size = 10

        self.__train_transform = transforms.Compose([
            transforms.RandomRotation(10),      # rotate +/- 10 degrees
            transforms.RandomHorizontalFlip(),  # reverse 50% of images
            transforms.Resize(224),             # resize shortest side to 224 pixels
            transforms.CenterCrop(224),         # crop longest side to 224 pixels at center
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
            ])

        self.__test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                [0.229,0.224,0.225])
            ])

        self.__train_data = datasets.ImageFolder(root = (train_data_url), transform = self.__train_transform)
        self.__test_data = datasets.ImageFolder(root = (test_data_url), transform = self.__test_transform)

        self.__train_loader = dl(self.__train_data, self.batch_size, shuffle = True)
        self.__test_loader = dl(self.__test_data, self.batch_size)

    def get_train_loader(self):
        return self.__train_loader

    def get_test_loader(self):
        return self.__test_loader

