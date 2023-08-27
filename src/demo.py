import torch
import torch.utils.data as tu_data
from PIL import Image

import matplotlib.pyplot as plt

from DehazingDataset import DatasetType, DehazingDataset
from main import AODnet
from main import Preprocess
from main import GetProjectDir

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.has_mps else 'cpu')

    datasetPath = GetProjectDir() / 'dataset/SS594_Multispectral_Dehazing/Haze1k/Haze1k'
    testingDataset = DehazingDataset(dehazingDatasetPath=datasetPath, _type=DatasetType.Train, transformFn=Preprocess,
                                      verbose=False)
    testingDataLoader = tu_data.DataLoader(testingDataset, batch_size=32, shuffle=True, num_workers=3)

    input_image_path = "/Users/flameberry/Developer/Dehazing/dataset/SS594_Multispectral_Dehazing/Haze1k/Haze1k/Haze1k_thick/dataset/test/input/357-inputs.png"
    input_image = Preprocess(Image.open(input_image_path).convert('RGB'))

    model = AODnet().to(device)
    checkpoint = torch.load('../saved_models/model_2023-08-27_15-34-05/AOD_9.pkl')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    for step, (haze_image, ori_image) in enumerate(testingDataLoader):
        haze_image, ori_image = haze_image.to(device), ori_image.to(device)

        output = model(haze_image)

        in_img = haze_image.cpu()[:1].detach().permute(2, 3, 1, 0).numpy().reshape(512, 512, 3)
        out = output.cpu()[:1].detach().permute(2, 3, 1, 0).numpy().reshape(512, 512, 3)
        clearImage = ori_image.cpu()[:1].detach().permute(2, 3, 1, 0).numpy().reshape(512, 512, 3)

        _, axes = plt.subplots(1, 3)
        axs = axes.flatten()
        axs[0].imshow(in_img)
        axs[1].imshow(out)
        axs[2].imshow(clearImage)
        plt.show()
