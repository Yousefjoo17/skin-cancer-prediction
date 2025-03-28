
from collections import namedtuple
from PIL import Image
import torch
import os
import random 
import math
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.io import read_image

#benign =>False , malignant =>true
def get_images_data(is_train_folder,is_malignant):
    if is_train_folder:
        if is_malignant:
            folder_path = "data/train/malignant"
        else:
            folder_path = "data/train/benign"

    else:  # test
        if is_malignant:
            folder_path = "data/test/malignant"
        else:
            folder_path = "data/test/benign"

    transform = transforms.Compose([
        # transforms.Resize((224, 224)),  
        transforms.ToTensor()  
    ])

    def load_images(folder, label):
        imgs_list = []
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            try:
                img = Image.open(img_path).convert("RGB")  
                img_tensor = transform(img)
                img_id = os.path.splitext(filename)[0]  
                imgs_list.append((img_tensor,label, img_id))  
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        return imgs_list

    imgs_data = load_images(folder_path, is_malignant)

    return imgs_data    # list(image tensor, id, label)

augmentation_dict = {
    'flip': True, # Flips the image either horizontally or vertically.
    'offset': 0.1,# Moves the image by a fraction of its size (0.1 means 10% shift).
    'scale': 0.2, # Zoom In/Out
    'rotate': True, # Rotates the image by a random degree.
    'noise': 0.05  # Noise should be a small fraction (e.g., 0.05 instead of 25.0)
}

def get_augmented_img(img_tensor): 
    """
    Applies 2D augmentations (flip, scale, translate, rotate, and noise) to an image tensor.
    
    Args:
        img_tensor (torch.Tensor): Input image tensor of shape (1, C, H, W)

    Returns:
        torch.Tensor: Augmented image tensor of the same shape (1, C, H, W)
    """

    img_tensor = img_tensor.unsqueeze(0)
    device = img_tensor.device
    transform_t = torch.eye(3, dtype=torch.float32, device=device)  # 3×3 affine matrix

    # Horizontal and Vertical Flip
    if augmentation_dict.get('flip', False):
        if random.random() > 0.5:
            transform_t[0, 0] *= -1  # Flip horizontally

        if random.random() > 0.5:
            transform_t[1, 1] *= -1  # Flip vertically

    # Translation (Offset)
    if 'offset' in augmentation_dict:
        offset_val = augmentation_dict['offset']
        transform_t[0, 2] = (random.random() * 2 - 1) * offset_val  # X-axis translation
        transform_t[1, 2] = (random.random() * 2 - 1) * offset_val  # Y-axis translation

    # Scaling
    if 'scale' in augmentation_dict:
        scale_val = augmentation_dict['scale']
        scale_factor = 1.0 + (random.random() * 2 - 1) * scale_val
        transform_t[0, 0] *= scale_factor
        transform_t[1, 1] *= scale_factor

    # Rotation
    if 'rotate' in augmentation_dict:
        angle_rad = (random.random() * 2 - 1) * math.pi / 4  # Random angle between -45° and 45°
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

        rotation_t = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0,      0,     1]
        ], dtype=torch.float32, device=device)

        transform_t = rotation_t @ transform_t  # Apply rotation

    # Convert to (1,2,3) matrix for affine_grid
    affine_t = transform_t[:2].unsqueeze(0)  # Shape (1, 2, 3)

    # Generate grid for transformation
    grid = F.affine_grid(affine_t, img_tensor.size(), align_corners=False)

    # Apply the transformation
    augmented_img = F.grid_sample(img_tensor, grid, padding_mode='border', align_corners=False)

    # Add noise
    if 'noise' in augmentation_dict:
        noise_std = augmentation_dict['noise'] * torch.std(img_tensor)  # Scale noise based on image contrast
        noise = torch.randn_like(augmented_img, device=device) * noise_std
        augmented_img += noise

    return augmented_img.squeeze(0).clamp(0, 1)  # Keep pixel values in valid range
    # return the full tensor including batch diememsion (1,3,224,224)

class SkinCancerDataset(Dataset):
    def __init__(self, is_val=None, is_test=None, val_stride=0):
        self.is_test=is_test
        self.pos_list=None
        self.neg_list=None

        if (is_test):
            self.pos_list=get_images_data(is_train_folder=False,is_malignant=True) # get positive samples from test folder
            self.neg_list=get_images_data(is_train_folder=False,is_malignant=False) # get negative samples from test folder
            assert self.pos_list , self.neg_list
        
        else: # train & validation
            self.pos_list=get_images_data(is_train_folder=True,is_malignant=True) # get all positive samples from train folder
            self.neg_list=get_images_data(is_train_folder=True,is_malignant=False) # get all negative samples from train folder
            
            assert val_stride > 0, val_stride

            if(is_val): # validation set
                self.pos_list=self.pos_list[::val_stride]
                self.neg_list=self.neg_list[::val_stride]
                assert self.pos_list and self.neg_list
            else: # train set
                del self.pos_list[::val_stride]
                del self.neg_list[::val_stride]
                assert self.pos_list and self.neg_list
    

    def shuffleSamples(self): #  We will call this at the top of each epoch to randomize the order of samples being presented.
            random.shuffle(self.neg_list)
            random.shuffle(self.pos_list)

    def __len__(self):
        if(self.is_test):
            return len(self.neg_list)+ len(self.pos_list)
        return 3000  # if I use augmentation so I am not bounded by the actual number of samples on my disk, I can generate more and more
        
    def __getitem__(self, index):  # Balanced the data 1:1 pos:neg
        if(index % 2==0):
            index= index % len(self.neg_list)

            if (self.is_test):
                img=self.neg_list[index][0]
            else:
                img=get_augmented_img(self.neg_list[index][0])

            label=self.neg_list[index][1]
            img_id=self.neg_list[index][2]
                #augmented img tensor whose shape (3,224,224), img id , img label
        else:
            index= index % len(self.pos_list)

            if(self.is_test):
                img=self.pos_list[index][0]
            else:
                img=get_augmented_img(self.pos_list[index][0])
                
            label=self.pos_list[index][1]
            img_id=self.pos_list[index][2]

        img_label = torch.tensor([not label, label],dtype=torch.long)

        return img, img_label, img_id

