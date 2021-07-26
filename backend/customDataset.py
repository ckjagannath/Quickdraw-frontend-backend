from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class customDatasetClass(Dataset):

    def __init__(self, path):

        """
        Custom Dataset Iterator class
        :param path: Path to the directory containing the images
        Data
        ├───Class0
        │       image0.png
        │       image1.png
        │       image2.png
        │       image3.png
        │
        ├───Class1
        │       image0.png
        │       image1.png
        │       image2.png
        │       image3.png
        │
        ├───Class2
        │       image0.png
        │       image1.png
        │       image2.png
        │       image3.png
        │
        └───Class3
                image0.png
                image1.png
                image2.png
                image3.png
        """
        self.path = path
        self.allImages = []
        self.allTargets = []
        self.allClasses = sorted(os.listdir(self.path))

        for targetNo, targetI in enumerate(self.allClasses):
            for imageI in sorted(os.listdir(self.path + '/' + targetI)):
                imageI=self.process(self.path + '/' + targetI + '/' + imageI)
                self.allImages.append(imageI)
                self.allTargets.append(targetNo)

        

    def process(self, path):
        image = Image.fromarray(plt.imread(path)[:, :, 3])  # read alpha channel
        image = image.resize((96, 96))
        image = (np.array(image) > 0.1).astype(np.float32)[None, :, :]
        return image
    

    def __getitem__(self, item):

        image= self.allImages[item]
        target=self.allTargets[item]

        return image, target

    def __len__(self):

        return len(self.allImages)