import os
import datetime

from PIL import Image
import pathlib
from DehazingDataset import DatasetType, DehazingDataset

# TODO: Abstract this in DehazingModel.py
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as tn_functional
import torch.utils.data as tu_data
from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as transforms
import torchvision.transforms.functional as tv_functional
from torchvision.utils import make_grid

import cv2.ximgproc
import matplotlib.pyplot as plt

def GetProjectDir() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent

def Preprocess(image: Image.Image) -> torch.Tensor:
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
    return torch.from_numpy(filteredImage).permute(2, 0, 1)

def save_model(epoch, path, net, optimizer, net_name):
    if not os.path.exists(os.path.join(path, net_name)):
        os.mkdir(os.path.join(path, net_name))
    torch.save({'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()},
               f=os.path.join(path, net_name, '{}_{}.pkl'.format('AOD', epoch)))

class AODnet(nn.Module):
    def __init__(self):
        super(AODnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=7, stride=1, padding=3)
        self.conv5 = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.b = 1

    def forward(self, x):
        x1 = tn_functional.relu(self.conv1(x))
        x2 = tn_functional.relu(self.conv2(x1))
        cat1 = torch.cat((x1, x2), 1)
        x3 = tn_functional.relu(self.conv3(cat1))
        cat2 = torch.cat((x2, x3), 1)
        x4 = tn_functional.relu(self.conv4(cat2))
        cat3 = torch.cat((x1, x2, x3, x4), 1)
        k = tn_functional.relu(self.conv5(cat3))

        if k.size() != x.size():
            raise Exception("k, haze image are different size!")

        output = k * x - k + self.b
        return tn_functional.relu(output)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.has_mps else 'cpu')

    datasetPath = GetProjectDir() / 'dataset/SS594_Multispectral_Dehazing/Haze1k/Haze1k'
    trainingDataset = DehazingDataset(dehazingDatasetPath=datasetPath, _type=DatasetType.Train, transformFn=Preprocess, verbose=False)
    validationDataset = DehazingDataset(dehazingDatasetPath=datasetPath, _type=DatasetType.Validation, transformFn=Preprocess, verbose=False)

    # tensor_image = dehazingDataset[0][0]
    # plt.imshow(tensor_image)
    # plt.title('Sample Preprocessed Image')
    # plt.show()
    print(trainingDataset[0][0].shape)

    # TODO: Abstract this in DehazingModel.py
    trainingDataLoader = tu_data.DataLoader(trainingDataset, batch_size=32, shuffle=True, num_workers=3)
    validationDataLoader = tu_data.DataLoader(validationDataset, batch_size=32, shuffle=True, num_workers=3)

    # Instantiate the AODNet model
    model = AODnet().to(device)
    print(model)

    criterion = nn.MSELoss().to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 10

    summary = SummaryWriter(log_dir=str(GetProjectDir() / ("summary/model_summary_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))), comment='')

    train_number = len(trainingDataset)

    print('Started Training...')
    model.train()
    for epoch in range(EPOCHS):
        for step, (haze_image, ori_image) in enumerate(trainingDataLoader):
            count = epoch * train_number + (step + 1)
            ori_image, haze_image = ori_image.to(device), haze_image.to(device)
            dehaze_image = model(haze_image)
            loss = criterion(dehaze_image, ori_image)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            summary.add_scalar('loss', loss.item(), count)
            if step % 100 == 0:
                summary.add_image('DeHaze_Images', make_grid(dehaze_image[:4].data, normalize=True, scale_each=True),
                                  count)
                summary.add_image('Haze_Images', make_grid(haze_image[:4].data, normalize=True, scale_each=True), count)
                summary.add_image('Origin_Images', make_grid(ori_image[:4].data, normalize=True, scale_each=True),
                                  count)

            print('Epoch: {}/{}  |  Step: {}/{}  |  lr: {:.6f}  | Loss: {:.6f}'
                  .format(epoch + 1, EPOCHS, step + 1, train_number,
                          optimizer.param_groups[0]['lr'], loss.item()))
        # -------------------------------------------------------------------
        # start validation
        print('Epoch: {}/{} | Validation Model Saving Images'.format(epoch + 1, EPOCHS))
        model.eval()
        for step, (haze_image, ori_image) in enumerate(validationDataLoader):
            if step > 10:  # only save image 10 times
                break
            ori_image, haze_image = ori_image.to(device), haze_image.to(device)
            dehaze_image = model(haze_image)
            torchvision.utils.save_image(
                torchvision.utils.make_grid(torch.cat((haze_image, dehaze_image, ori_image), 0),
                                            nrow=ori_image.shape[0]),
                os.path.join(GetProjectDir() / "output", '{}_{}.jpg'.format(epoch + 1, step)))

        model.train()
        # -------------------------------------------------------------------
        # save per epochs model
        save_model(epoch, GetProjectDir() / "saved_models", model, optimizer, str(epoch) + "_model_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # -------------------------------------------------------------------
    # train finish
    summary.close()
