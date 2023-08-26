import matplotlib.pyplot as plt
from PIL import Image
import pathlib
from DehazingDataset import DatasetType, DehazingDataset

# TODO: Abstract this in DehazingModel.py
import torch.utils.data as tu_data
import torchvision.transforms as transforms
import torchvision.transforms.functional as tv_functional

import cv2.ximgproc

def GetProjectDir() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent

def Preprocess(image: Image.Image) -> Image.Image:
    # Contrast Enhancement
    transform = transforms.Compose([transforms.PILToTensor(), transforms.functional.equalize])
    transformedImage = transform(image)

    # Gamma Correction
    gammaCorrectedImage = transforms.functional.adjust_gamma(transformedImage, 2.2)

    # Histogram Stretching
    min_val = gammaCorrectedImage.min()
    max_val = gammaCorrectedImage.max()
    stretchedImage = (gammaCorrectedImage - min_val) / (max_val - min_val)

    # Guided Filtering
    gFilter = cv2.ximgproc.createGuidedFilter(guide=stretchedImage.permute(1, 2, 0).numpy(), radius=3, eps=0.01)
    filteredImage = gFilter.filter(src=stretchedImage.permute(1, 2, 0).numpy())
    return filteredImage


if __name__ == '__main__':
    datasetPath = GetProjectDir() / 'dataset/SS594_Multispectral_Dehazing/Haze1k/Haze1k'
    dehazingDataset = DehazingDataset(dehazingDatasetPath=datasetPath, _type=DatasetType.Train, transformFn=Preprocess, verbose=True)
    print(len(dehazingDataset))

    tensor_image = dehazingDataset[0][0]
    plt.imshow(tensor_image)
    plt.title('Sample Preprocessed Image')
    plt.show()

    # TODO: Abstract this in DehazingModel.py
    dataLoader = tu_data.DataLoader(dehazingDataset, batch_size=32, shuffle=True, num_workers=2)
