import os
import pathlib
import numpy as np
from PIL import Image
import h5py

import cv2

from DehazingDataset import DatasetType
from DCP.dcp import DarkChannel, AtmLight, TransmissionEstimate, TransmissionRefine


def GenerateTransmissionMaps(datasetType: DatasetType, verbose=False):
    dehazingDatasetPath = pathlib.Path("dataset/SS594_Multispectral_Dehazing/Haze1k/Haze1k")

    hazyImagePaths = []
    clearImagePaths = []

    hazyImages = []
    clearImages = []
    estimateMaps = []
    refinedMaps = []

    for variant in ("Haze1k_thin", "Haze1k_moderate", "Haze1k_thick"):
        inputPath = dehazingDatasetPath / variant / "dataset" / datasetType.ToString() / "input"
        targetPath = dehazingDatasetPath / variant / "dataset" / datasetType.ToString() / "target"

        hazyImagePaths += [
            inputPath / filename
            for filename in sorted(os.listdir(inputPath))
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp"))
        ]
        clearImagePaths += [
            targetPath / filename
            for filename in sorted(os.listdir(targetPath))
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp"))
        ]

    print(len(hazyImagePaths), len(clearImagePaths))
    # Filtering the mismatching (input, target) image pair
    assert len(hazyImagePaths) == len(clearImagePaths)
    for hazyPath, clearPath in zip(hazyImagePaths, clearImagePaths):
        hazyImage = cv2.imread(str(hazyPath))
        clearImage = cv2.imread(str(clearPath))

        if hazyImage is None or clearImage is None:
            print(hazyPath, clearPath)

        valid = hazyImage is not None and clearImage is not None and hazyImage.shape == clearImage.shape
        if not valid:
            hazyImagePaths.remove(hazyPath)
            clearImagePaths.remove(clearPath)
            continue

        if verbose:
            print(hazyPath)
            print(clearPath)

        # Generate Transmission Maps
        I = hazyImage.astype("float64") / 255

        dark = DarkChannel(I, 15)
        A = AtmLight(I, dark)
        estimate = TransmissionEstimate(I, A, 15)
        refined = TransmissionRefine(hazyImage, estimate)

        hazyImages.append(hazyImage)
        clearImages.append(clearImage)
        estimateMaps.append(estimate)
        refinedMaps.append(refined)

        # cv2.imshow("Estimate", estimate)
        # cv2.imshow("Refined", refined)
        # cv2.waitKey()

    print(len(hazyImages), len(clearImages), len(estimateMaps), len(refinedMaps))
    return hazyImages, clearImages, estimateMaps, refinedMaps


if __name__ == "__main__":
    os.chdir(pathlib.Path(__file__).parent.parent)
    hazyImages, clearImages, estimateMaps, refinedMaps = GenerateTransmissionMaps(datasetType=DatasetType.Train)

    save_path = "dataset/dehaze.hdf5"

    hf = h5py.File(save_path, "a")
    dset = hf.create_dataset("hazy_image", data=hazyImages)
    dset = hf.create_dataset("clear_image", data=clearImages)
    dset = hf.create_dataset("transmission_map", data=estimateMaps)
    dset = hf.create_dataset("transmission_map_refined", data=refinedMaps)
    hf.close()  # close the hdf5 file
    print("hdf5 file size: %d bytes" % os.path.getsize(save_path))
