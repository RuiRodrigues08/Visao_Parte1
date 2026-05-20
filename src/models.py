import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import timm  # <--- IMPORTANTE: Adicionado para o Modelo 6 (Transformer)

def get_resnet18_modified(num_classes=43, pretrained=True):
    """
    Carrega a ResNet18 e adapta o 'stem' (camadas iniciais) para imagens pequenas (48x48).
    """
    if pretrained:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    else:
        model = resnet18(weights=None)
        
    # 1. Modificar a primeira convolução:
    # A ResNet original usa um kernel 7x7 com stride 2 (reduz a imagem a metade logo aqui).
    # Mudei para um kernel 3x3 com stride 1 para preservar a resolução das nossas imagens de 48x48.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # 2. Remover o MaxPool inicial:
    # O MaxPool original também reduz a imagem a metade. Como as imagens já são pequenas,
    # substituí o MaxPool por uma camada Identity (que não faz nada, passa os dados intactos).
    model.maxpool = nn.Identity()
    
    # 3. Adaptar a última camada para as 43 classes do GTSRB (em vez das 1000 do ImageNet)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

def get_efficientnet_b0_modified(num_classes=43, pretrained=True):
    """
    Carrega a EfficientNetB0 e adapta a primeira camada para imagens pequenas.
    (Usado para os Modelos 4 e 5 no teu plano).
    """
    if pretrained:
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    else:
        model = efficientnet_b0(weights=None)
        
    # A primeira camada da EfficientNetB0 está no bloco features[0][0]
    # Originalmente é uma Conv2d com stride=2. No entanto mudei para stride=1.
    in_channels = model.features[0][0].in_channels
    out_channels = model.features[0][0].out_channels
    
    model.features[0][0] = nn.Conv2d(
        in_channels, out_channels, 
        kernel_size=3, stride=1, padding=1, bias=False
    )
    
    # Adaptar o classificador final
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    return model

def get_cct_model(num_classes=43, pretrained=True):
    """
    Substituto robusto: Vision Transformer (ViT) Tiny.
    Forçei o img_size=48 para o timm adaptar os Position Embeddings automaticamente.
    """
    import timm
    model = timm.create_model(
        'vit_tiny_patch16_224', 
        pretrained=pretrained, 
        num_classes=num_classes, 
        img_size=48 
    )
    return model

# Teste rápido para garantir que as dimensões batem certo se o ficheiro for executado
if __name__ == "__main__":
    dummy_input = torch.randn(2, 3, 48, 48) # Batch de 2 imagens 48x48
    
    resnet = get_resnet18_modified()
    out_res = resnet(dummy_input)
    print(f"Output ResNet shape: {out_res.shape}") # Deve devolver [2, 43]
    
    effnet = get_efficientnet_b0_modified()
    out_eff = effnet(dummy_input)
    print(f"Output EfficientNet shape: {out_eff.shape}") # Deve devolver [2, 43]
    
    cct = get_cct_model()
    out_cct = cct(dummy_input)
    print(f"Output CCT shape: {out_cct.shape}") # Deve devolver [2, 43]