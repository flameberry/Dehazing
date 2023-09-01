import matplotlib.pyplot as plt
from PIL import Image
from main import GetProjectDir, Preprocess

if __name__ == '__main__':
    img = Image.open(GetProjectDir() / 'dataset/SS594_Multispectral_Dehazing/Haze1k/Haze1k/Haze1k_thick/dataset/train/input/4-inputs.png').convert('RGB')
    processedImage = Preprocess(img)

    _, axes = plt.subplots(1, 2)
    axs = axes.flatten()
    axs[0].imshow(img)
    axs[1].imshow(processedImage.permute(1, 2, 0).numpy())
    plt.show()