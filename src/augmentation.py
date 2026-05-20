import albumentations as A
from albumentations.pytorch import ToTensorV2

# Tamanho padrão definido
IMAGE_SIZE = 48 

# Valores aproximados da média e desvio padrão do dataset GTSRB
GTSRB_MEAN = [0.3403, 0.3121, 0.3214]
GTSRB_STD = [0.2724, 0.2608, 0.2669]

def get_train_transform_baseline():
    """
    Modelo 1 (Baseline) - Sem augmentation.
    Apenas Resize, Normalização e conversão para Tensor PyTorch.
    """
    return A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Normalize(mean=GTSRB_MEAN, std=GTSRB_STD),
        ToTensorV2()
    ])

def get_train_transform_geo():
    """
    Modelo 2 (Geo) - CLAHE + Augmentation Geométrica.
    """
    return A.Compose([
        # 1. Redimensionar
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        # 2. Filtro CLAHE (Aplica a equalização de histograma adaptativa) - probabilidade de 100%
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        # 3. Augmentation Geométrica (Shift, Scale, Rotate)
        A.ShiftScaleRotate(
            shift_limit=0.1,   # Move 10% para os lados/cima/baixo
            scale_limit=0.2,   # Zoom in/out até 20%
            rotate_limit=15,   # Roda até 15 graus
            p=0.5              # Aplica apenas a 50% das imagens da batch para manter variedade
        ),
        # 4. Normalizar e converter
        A.Normalize(mean=GTSRB_MEAN, std=GTSRB_STD),
        ToTensorV2()
    ])

def get_valid_transform():
    """
    Transformação obrigatória para Validação e Teste.
    NUNCA se aplica augmentation a dados de teste, apenas Resize e Normalização!
    """
    return A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Normalize(mean=GTSRB_MEAN, std=GTSRB_STD),
        ToTensorV2()
    ])

def get_valid_transform_clahe():
    """
    Transformação de validação para modelos treinados com CLAHE.
    Aplica o filtro de luz, mas NUNCA aplica rotações ou zooms aleatórios.
    """
    return A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0), 
        A.Normalize(mean=GTSRB_MEAN, std=GTSRB_STD),
        ToTensorV2()
    ])

def get_train_transform_color_geo():
    """Modelo 3 e Modelo 6 (Cor + Geo): CLAHE, Rotação, Zoom, Cor, Ruído"""
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
    """Modelo 4 (Oclusão): Cutout / CoarseDropout"""
    return A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
        A.CoarseDropout(max_holes=2, max_height=12, max_width=12, min_holes=1, min_height=8, min_width=8, fill_value=0, p=0.5),
        A.Normalize(mean=GTSRB_MEAN, std=GTSRB_STD),
        ToTensorV2()
    ])

def get_train_transform_randaug():
    """Modelo 5 (RandAugment Simulado): Escolhe transformações pesadas aleatoriamente"""
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