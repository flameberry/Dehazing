import matplotlib.pyplot as plt
from PIL import Image
from main import GetProjectDir, Preprocess

if __name__ == '__main__':
    img = Image.open(GetProjectDir() / 'dataset/SS594_Multispectral_Dehazing/Haze1k/Haze1k/Haze1k_thick/dataset/train/input/4-inputs.png').convert('RGB')
    orig_image = Image.open(GetProjectDir() / 'dataset/SS594_Multispectral_Dehazing/Haze1k/Haze1k/Haze1k_thick/dataset/train/target/4-targets.png').convert('RGB')
    processedImage = Preprocess(img)

    _, axes = plt.subplots(1, 3)
    axs = axes.flatten()
    axs[0].imshow(img)
    axs[0].set_xlabel('Hazy Image')
    axs[1].imshow(processedImage.permute(1, 2, 0).numpy())
    axs[1].set_xlabel('Preprocessed Image')
    axs[2].imshow(orig_image)
    axs[2].set_xlabel('Original Image')
    plt.show()