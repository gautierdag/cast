import glob
import json

from PIL import Image
from torch.utils.data import Dataset


class SimilarityPairDataset(Dataset):
    def __init__(self, data_dir="data", resize=(512, 512)):
        example_files = glob.glob(data_dir + "/*.json")
        self.data = []
        for file in example_files:
            with open(file, "r") as f:
                example = json.load(f)

            # load image pair
            path_id = file.split("/")[-1].split(".")[0]
            example["image_0"] = Image.open(f"{data_dir}/{path_id}_0.jpg").convert(
                "RGB"
            )
            example["image_1"] = Image.open(f"{data_dir}/{path_id}_1.jpg").convert(
                "RGB"
            )
            if resize is not None:
                example["image_0"] = example["image_0"].resize(resize)
                example["image_1"] = example["image_1"].resize(resize)

            self.data.append(example)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
