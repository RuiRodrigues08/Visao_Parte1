import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import timm  # needed for Model 6 (Transformer)

def get_resnet18_modified(num_classes=43, pretrained=True):
    """
    Load ResNet18 and adapt the stem (initial layers) to small 48x48 images.
    """
    if pretrained:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    else:
        model = resnet18(weights=None)
        
    # 1. Modify the first convolution:
    # The original ResNet uses a 7x7 kernel with stride 2 (halves the image right here).
    # Changed to a 3x3 kernel with stride 1 to preserve the resolution of our 48x48 images.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # 2. Remove the initial MaxPool:
    # The original MaxPool also halves the image. Since the images are already small,
    # it is replaced by an Identity layer (passes the data through unchanged).
    model.maxpool = nn.Identity()
    
    # 3. Replace the last layer with the 43 GTSRB classes (instead of the 1000 ImageNet ones)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

def get_efficientnet_b0_modified(num_classes=43, pretrained=True):
    """
    Load EfficientNet-B0 and adapt the first layer to small images.
    (Used for Models 4 and 5.)
    """
    if pretrained:
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    else:
        model = efficientnet_b0(weights=None)
        
    # The first layer of EfficientNet-B0 is features[0][0].
    # It is originally a Conv2d with stride=2; changed here to stride=1.
    in_channels = model.features[0][0].in_channels
    out_channels = model.features[0][0].out_channels
    
    model.features[0][0] = nn.Conv2d(
        in_channels, out_channels, 
        kernel_size=3, stride=1, padding=1, bias=False
    )
    
    # Replace the final classifier
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    return model

def get_cct_model(num_classes=43, pretrained=True):
    """
    Model 6: Vision Transformer (ViT) Tiny (despite the function name, this is a ViT, not a CCT).
    img_size=48 makes timm adapt the position embeddings automatically.
    """
    import timm
    model = timm.create_model(
        'vit_tiny_patch16_224', 
        pretrained=pretrained, 
        num_classes=num_classes, 
        img_size=48 
    )
    return model

# Quick sanity check of the output shapes when the file is run directly
if __name__ == "__main__":
    dummy_input = torch.randn(2, 3, 48, 48) # Batch of 2 images, 48x48
    
    resnet = get_resnet18_modified()
    out_res = resnet(dummy_input)
    print(f"Output ResNet shape: {out_res.shape}") # Should return [2, 43]
    
    effnet = get_efficientnet_b0_modified()
    out_eff = effnet(dummy_input)
    print(f"Output EfficientNet shape: {out_eff.shape}") # Should return [2, 43]
    
    cct = get_cct_model()
    out_cct = cct(dummy_input)
    print(f"Output CCT shape: {out_cct.shape}") # Should return [2, 43]