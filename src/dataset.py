import os
import glob
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def get_data_splits(train_dir, test_size=0.15, random_state=42):
    """Read the class folders and return stratified train and validation DataFrames."""
    image_paths = glob.glob(os.path.join(train_dir, "*", "*.ppm")) 
    data = []
    for path in image_paths:
        class_id = int(os.path.basename(os.path.dirname(path)))
        data.append({"image_path": path, "class_id": class_id})
        
    df = pd.DataFrame(data)
    
    # Stratified split
    train_df, val_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=df["class_id"]
    )
    
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

class GTSRBDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["image_path"]
        label = self.df.iloc[idx]["class_id"]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
            
        return image, torch.tensor(label, dtype=torch.long)

def get_dataloaders(train_df, val_df, train_transform=None, val_transform=None, batch_size=32):
    """Utility to quickly build the train and validation dataloaders."""
    train_dataset = GTSRBDataset(train_df, transform=train_transform)
    val_dataset = GTSRBDataset(val_df, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader