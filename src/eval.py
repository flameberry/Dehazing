import torch
import torch.utils.data as tu_data
import torchvision.transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure

from PIL import Image

import matplotlib.pyplot as plt

from DehazingDataset import DatasetType, DehazingDataset
from main import AODnet, Preprocess, GetProjectDir

def display(in_img, out_img, clear_img):
    _, axes = plt.subplots(1, 3)
    axs = axes.flatten()
    axs[0].imshow(in_img)
    axs[1].imshow(out_img)
    axs[2].imshow(clear_img)
    plt.show()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.has_mps else 'cpu')

    datasetPath = GetProjectDir() / 'dataset/SS594_Multispectral_Dehazing/Haze1k/Haze1k'
    testingDataset = DehazingDataset(dehazingDatasetPath=datasetPath, _type=DatasetType.Test, transformFn=Preprocess,
                                      verbose=False)
    testingDataLoader = tu_data.DataLoader(testingDataset, batch_size=32, shuffle=True, num_workers=3)

    # input_image_path = "/Users/flameberry/Developer/Dehazing/dataset/SS594_Multispectral_Dehazing/Haze1k/Haze1k/Haze1k_thick/dataset/test/input/357-inputs.png"
    # clear_image_path = "/Users/flameberry/Developer/Dehazing/dataset/SS594_Multispectral_Dehazing/Haze1k/Haze1k/Haze1k_thick/dataset/test/target/357-targets.png"
    # input_image = Preprocess(Image.open(input_image_path).convert('RGB'))
    # clear_image = Preprocess(Image.open(input_image_path).convert('RGB'))

    model = AODnet().to(device)
    checkpoint = torch.load('../saved_models/9_model_2023-08-27_17-52-39/AOD_9.pkl')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # output = model(input_image.to(device))
    # display(input_image, output, torchvision.transforms.PILToTensor()(clear_image))

    for step, (haze_image, ori_image) in enumerate(testingDataLoader):
        haze_image, ori_image = haze_image.to(device), ori_image.to(device)

        output = model(haze_image)

        ssim = StructuralSimilarityIndexMeasure().to(device)
        ssim_val = ssim(output, ori_image)
        ssim_fake_val = ssim(haze_image, ori_image)
        print(f'SSIM: {ssim_val}, SSIM_Fake: {ssim_fake_val}')
        perc = (ssim_val - ssim_fake_val) * 100.0 / (1.0 - ssim_fake_val)
        print(f'Percentage Improvement: {perc} %')

        in_img = haze_image.cpu()[:1].detach().permute(2, 3, 1, 0).numpy().reshape(512, 512, 3)
        out = output.cpu()[:1].detach().permute(2, 3, 1, 0).numpy().reshape(512, 512, 3)
        clearImage = ori_image.cpu()[:1].detach().permute(2, 3, 1, 0).numpy().reshape(512, 512, 3)

        display(in_img, out, clearImage)
