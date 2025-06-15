from torchvision import transforms
import torch
import torch.nn.functional as F
import numpy as np
import random
# Two augment pipelines for contrastive learning
ssl_transform_1 = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
])

ssl_transform_2 = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(p=0.2)
])

# Custom Collate function for Dinov2 input compatibility
def ssl_collate_fn(batch):
    """
    batch is a list of tuples: [(img1, img2), (img1, img2), ...]
    We want to return two lists: [img1_0, img1_1, ...], [img2_0, img2_1, ...]
    """
    imgs1 = [item[0] for item in batch]
    imgs2 = [item[1] for item in batch]
    return imgs1, imgs2

def parse_layers(s):
    # Parses a string like "0,5,11" to [0,5,11]
    return [int(x) for x in s.split(",")]

def contrastive_loss(x1, x2, temperature=0.5):
    x1, x2 = F.normalize(x1, dim=1), F.normalize(x2, dim=1)
    logits = x1 @ x2.T / temperature
    labels = torch.arange(x1.size(0), device=x1.device)
    return F.cross_entropy(logits, labels)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)