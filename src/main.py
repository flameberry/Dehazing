import pathlib
from DehazingDataset import DatasetType, DehazingDataset

def GetProjectDir() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent


if __name__ == '__main__':
    datasetPath = GetProjectDir() / 'dataset/SS594_Multispectral_Dehazing/Haze1k/Haze1k'
    dataset = DehazingDataset(dehazingDatasetPath=datasetPath, _type=DatasetType.Train, verbose=True)
    print(len(dataset))
