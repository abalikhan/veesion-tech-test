import os
from PIL import Image
from torch.utils.data import Dataset

class SSLDataset(Dataset):
    """
    Loads images from a folder, applies two separate augmentations per image.
    """
    def __init__(self, image_dir, transform1, transform2):
        self.image_dir = image_dir
        self.files = sorted(x for x in os.listdir(image_dir)
                            if x.lower().endswith('.jpg')) # I have .jpg files but could be modified to include other image types
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.files[idx])
        img = Image.open(img_path).convert("RGB")
        # Return two augmented views
        return self.transform1(img), self.transform2(img)
