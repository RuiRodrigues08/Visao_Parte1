import os
import torch
import numpy as np
from tqdm import tqdm

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for a single epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100.*correct/total:.2f}%"})
        
    return running_loss / len(dataloader), 100. * correct / total

def validate_one_epoch(model, dataloader, criterion, device):
    """Evaluate the model on the validation set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / len(dataloader), 100. * correct / total

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, model_name):
    """Main loop: runs the epochs and saves the best model."""
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    os.makedirs('../checkpoints', exist_ok=True)
    save_path = f"../checkpoints/{model_name}.pth"
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        
        if scheduler:
            scheduler.step()
            
        history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss); history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"-> Best model saved! (Accuracy: {best_acc:.2f}%)")
            
    return history

# ==========================================
# =   FUNCTIONS FOR MODEL 5 (MIXUP)        =
# ==========================================

def mixup_data(x, y, alpha=0.2, device='cuda'):
    """Generate the MixUp mixtures for a batch."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def train_mixup_epoch(model, dataloader, criterion, optimizer, device, alpha=0.2):
    """Epoch training loop specific to Model 5 (MixUp)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training (MixUp)", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        inputs, targets_a, targets_b, lam = mixup_data(images, labels, alpha, device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        
        if lam > 0.5:
            correct += predicted.eq(targets_a).sum().item()
        else:
            correct += predicted.eq(targets_b).sum().item()
            
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    return running_loss / len(dataloader), 100. * correct / total