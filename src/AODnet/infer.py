import os
import pathlib
import PIL
from matplotlib import pyplot as plt
import torch, torchvision
import torch.optim
from AODnetModel import AODnet
import numpy as np
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from DehazingDataset import DatasetType, DehazingDataset
import torchvision.transforms as transforms
from PIL import Image

import cv2


def Preprocess(image: Image.Image) -> torch.Tensor:
    # Contrast Enhancement
    transform = transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.ColorJitter(
                brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05
            ),
            # transforms.functional.equalize
        ]
    )
    transformedImage = transform(image)

    # Gamma Correction
    gammaCorrectedImage = transforms.functional.adjust_gamma(transformedImage, 2.2)

    # Histogram Stretching
    min_val = gammaCorrectedImage.min()
    max_val = gammaCorrectedImage.max()
    stretchedImage = (gammaCorrectedImage - min_val) / (max_val - min_val)

    # for x in stretchedImage:
    #     for y in x:
    #         print(y)

    # Guided Filtering
    gFilter = cv2.ximgproc.createGuidedFilter(
        guide=stretchedImage.permute(1, 2, 0).numpy(), radius=3, eps=0.01
    )
    filteredImage = gFilter.filter(src=stretchedImage.permute(1, 2, 0).numpy())
    return torch.from_numpy(filteredImage).permute(2, 0, 1)


device = torch.device(
    "cpu"
    # "cuda"
    # if torch.cuda.is_available()
    # else "mps" if torch.backends.mps.is_built() else "cpu"
)


def image_haze_removal(input_image):
    hazy_image = np.asarray(input_image) / 255.0

    hazy_image = torch.from_numpy(hazy_image).float()
    hazy_image = hazy_image.permute(2, 0, 1)
    hazy_image = hazy_image.to(device).unsqueeze(0)

    # ld_net = AODnet().to(device)
    ld_net = torch.load("saved_models/saved_model_SSIM.pth", map_location=device)

    dehaze_image = ld_net(hazy_image)
    # return dehaze_image in a format that can be plotted using matplotlib
    return dehaze_image.cpu().detach().squeeze(0).permute(1, 2, 0).numpy()


def inference(args):
    datasetPath = (
        pathlib.Path(args["dataset"]) / "SS594_Multispectral_Dehazing/Haze1k/Haze1k"
    )
    test_dataset = DehazingDataset(
        dehazingDatasetPath=datasetPath,
        _type=DatasetType.Test,
        transformFn=Preprocess,
        verbose=False,
    )

    testing_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True
    )

    ld_net = torch.load("saved_models/saved_model_SSIM.pth", map_location=device)

    print("SSIM\tSSIM_Previous\tPSNR\tPSNR_Previous")

    print(len(testing_data_loader))

    ssim_old_values = []
    ssim_values = []
    psnr_old_values = []
    psnr_values = []

    ssim = StructuralSimilarityIndexMeasure().to(device)
    psnr = PeakSignalNoiseRatio().to(device)

    ld_net.eval()
    for iter_val, (hazy_image, hazefree_image) in enumerate(testing_data_loader):
        hazefree_image = hazefree_image.to(device)
        hazy_image = hazy_image.to(device)

        dehaze_image = ld_net(hazy_image)

        # Calculate and print the SSIM
        ssim_val = ssim(dehaze_image, hazefree_image)
        ssim_fake_val = ssim(hazy_image, hazefree_image)
        ssim_values.append(ssim_val)
        ssim_old_values.append(ssim_fake_val)

        # Calculate and print the PSNR
        psnr_val = psnr(dehaze_image, hazefree_image)
        psnr_fake_val = psnr(hazy_image, hazefree_image)
        psnr_values.append(psnr_val)
        psnr_old_values.append(psnr_fake_val)

        print(
            f"{ssim_val:.4f}\t|\t{ssim_fake_val:.4f}\t|\t{psnr_val:.4f}\t|\t{psnr_fake_val:.4f}"
        )

        save_path = f"visual_results/dehaze_img_{iter_val + 1}.jpg"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torchvision.utils.save_image(
            torch.cat((hazy_image, dehaze_image, hazefree_image), 0),
            save_path,
        )

    fig_, ax_ = ssim.plot(ssim_values)
    fig_, ax_ = ssim.plot(ssim_old_values, ax=ax_)
    plt.savefig("Inference_AODnet_SSIM.png")

    fig_, ax_1 = psnr.plot(psnr_values)
    fig_, ax_1 = psnr.plot(psnr_old_values, ax=ax_1)
    plt.savefig("Inference_AODnet_PSNR.png")

    # Plot the ssim and psnr
    plt.show()


if __name__ == "__main__":
    out = image_haze_removal(
        Image.open(
            "/Users/flameberry/Developer/Dehazing/dataset/SS594_Multispectral_Dehazing/Haze1k/Haze1k/Haze1k_moderate/dataset/test/input/1-inputs.png"
        )
    )
    plt.plot(out)
    plt.show()
    # inference({"dataset": "/Users/flameberry/Developer/Dehazing/dataset"})
