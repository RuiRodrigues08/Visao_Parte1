import albumentations as A
from albumentations.pytorch import ToTensorV2

# Default image size
IMAGE_SIZE = 48 

# Approximate mean and standard deviation of the GTSRB dataset
GTSRB_MEAN = [0.3403, 0.3121, 0.3214]
GTSRB_STD = [0.2724, 0.2608, 0.2669]

def get_train_transform_baseline():
    """
    Model 1 (Baseline) - no augmentation.
    Only resize, normalisation and conversion to a PyTorch tensor.
    """
    return A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Normalize(mean=GTSRB_MEAN, std=GTSRB_STD),
        ToTensorV2()
    ])

def get_train_transform_geo():
    """
    Model 2 (Geo) - CLAHE + geometric augmentation.
    """
    return A.Compose([
        # 1. Resize
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        # 2. CLAHE filter (adaptive histogram equalisation) - applied with 100% probability
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        # 3. Geometric augmentation (shift, scale, rotate)
        A.ShiftScaleRotate(
            shift_limit=0.1,   # Shift up to 10% in any direction
            scale_limit=0.2,   # Zoom in/out up to 20%
            rotate_limit=15,   # Rotate up to 15 degrees
            p=0.5              # Applied to only 50% of the images to keep variety
        ),
        # 4. Normalise and convert
        A.Normalize(mean=GTSRB_MEAN, std=GTSRB_STD),
        ToTensorV2()
    ])

def get_valid_transform():
    """
    Mandatory transform for validation and test.
    Never apply augmentation to evaluation data, only resize and normalisation.
    """
    return A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Normalize(mean=GTSRB_MEAN, std=GTSRB_STD),
        ToTensorV2()
    ])

def get_valid_transform_clahe():
    """
    Validation transform for models trained with CLAHE.
    Applies the contrast filter but never random rotations or zooms.
    """
    return A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0), 
        A.Normalize(mean=GTSRB_MEAN, std=GTSRB_STD),
        ToTensorV2()
    ])

def get_train_transform_color_geo():
    """Models 3 and 6 (Colour + Geo): CLAHE, rotation, zoom, colour, noise"""
    return A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.Normalize(mean=GTSRB_MEAN, std=GTSRB_STD),
        ToTensorV2()
    ])

def get_train_transform_occlusion():
    """Model 4 (Occlusion): Cutout / CoarseDropout"""
    return A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
        A.CoarseDropout(max_holes=2, max_height=12, max_width=12, min_holes=1, min_height=8, min_width=8, fill_value=0, p=0.5),
        A.Normalize(mean=GTSRB_MEAN, std=GTSRB_STD),
        ToTensorV2()
    ])

def get_train_transform_randaug():
    """Model 5 (simulated RandAugment): randomly picks heavy transforms"""
    return A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.SomeOf([
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3, rotate_limit=25, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.CoarseDropout(max_holes=3, max_height=16, max_width=16, fill_value=0, p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
        ], n=2, p=0.8),
        A.Normalize(mean=GTSRB_MEAN, std=GTSRB_STD),
        ToTensorV2()
    ])