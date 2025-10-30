import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import time
import argparse # For command-line arguments

# -----------------------------------------------------------------
# 1. FUNCTION TO VALIDATE DATA
# -----------------------------------------------------------------
def validate_data(df, df_name):
    """This function takes one of our dataframes and returns a clean Dataframe"""
    valid_rows = []
    invalid_count = 0
    print(f"\nValidating {df_name}...")
    
    # Use tqdm for a nice progress bar
    for index, row in tqdm(df.iterrows(), total=len(df)):
        image_path = Path(row['image_path'])
        mask_path = Path(row['mask_path'])
        
        try:
            # Check if files exist
            if not image_path.exists():
                print(f"Warning: Image file not found for row {index}. Skipping")
                invalid_count += 1
                continue
            if not mask_path.exists():
                print(f"Warning: Mask file not found for row {index}. Skipping")
                invalid_count += 1
                continue
            
            # Check if files are corrupted
            with Image.open(image_path) as img:
                img.load()
            with Image.open(mask_path) as mask:
                mask.load()
                
            # If all checks pass
            valid_rows.append(row)
            
        except (IOError, UnidentifiedImageError) as e:
            print(f"Warning: Corrupted image file at row {index}({image_path.name}). Error: {e}. Skipping.")
            invalid_count += 1
        except Exception as e:
            print(f"An unexpected error occurred for row {index}: {e}")
            invalid_count += 1
            
    print(f"Validation complete for {df_name}.")
    print(f"Original samples: {len(df)}")
    print(f"Removed samples: {invalid_count}")
    print(f"Valid samples: {len(valid_rows)}\n")
    
    # Return the new, clean DataFrame
    clean_df = pd.DataFrame(valid_rows)
    return clean_df

# -----------------------------------------------------------------
# 2. PYTORCH DATASET CLASS
# -----------------------------------------------------------------
class BreastCancerDataset(Dataset):
    """Custom Dataset for loading breast cancer images and masks."""
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = Path(row['image_path'])
        mask_path = Path(row['mask_path'])
        
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L') # 'L' for grayscale
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            mask = (mask > 0.5).float() # Binarize the mask to 0.0 or 1.0
            
        return image, mask

# -----------------------------------------------------------------
# 3. U-NET MODEL DEFINITION
# -----------------------------------------------------------------
class DoubleConv(nn.Module):
    """(Convolution => [Batch Norm] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), # Normalizes the output
            nn.ReLU(inplace=True),        # Activation function
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), # Normalizes the output
            nn.ReLU(inplace=True)         # Activation function
        )

    def forward(self, x):
        return self.double_conv(x)

class UNET(nn.Module):
    """U-Net Architecture for segmentation"""
    def __init__(self, in_channels, out_channels):
        super(UNET, self).__init__()
        
        # --- Encoder (Down-sampling Path) ---
        self.inc = DoubleConv(in_channels, 64)   # Input conv
        self.down1 = DoubleConv(64, 128)
        self.down2 = DoubleConv(128, 256)
        self.down3 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)              # Max pooling layer

        # --- Bottleneck ---
        self.bot = DoubleConv(512, 1024)

        # --- Decoder (Up-sampling Path) ---
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up1 = DoubleConv(1024, 512) # 512 (from upconv) + 512 (from skip)
        
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2 = DoubleConv(512, 256)  # 256 + 256
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up3 = DoubleConv(256, 128)  # 128 + 128
        
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up4 = DoubleConv(128, 64)   # 64 + 64

        # --- Output Layer ---
        # 1x1 conv to map to out_channels.
        # We output raw "logits" and use BCEWithLogitsLoss (no sigmoid here)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        p1 = self.pool(x1)
        x2 = self.down1(p1)
        p2 = self.pool(x2)
        x3 = self.down2(p2)
        p3 = self.pool(x3)
        x4 = self.down3(p3)
        p4 = self.pool(x4)
        
        # Bottleneck
        x5 = self.bot(p4)

        # Decoder (with Skip Connections)
        up1 = self.upconv1(x5)
        cat1 = torch.cat([up1, x4], dim=1) # Concat skip connection
        x6 = self.up1(cat1)

        up2 = self.upconv2(x6)
        cat2 = torch.cat([up2, x3], dim=1)
        x7 = self.up2(cat2)

        up3 = self.upconv3(x7)
        cat3 = torch.cat([up3, x2], dim=1)
        x8 = self.up3(cat3)

        up4 = self.upconv4(x8)
        cat4 = torch.cat([up4, x1], dim=1)
        x9 = self.up4(cat4)

        # Output
        logits = self.outc(x9) # Raw output (no sigmoid)
        return logits

# -----------------------------------------------------------------
# 4. MAIN TRAINING FUNCTION
# -----------------------------------------------------------------
def main(args):
    
    # --- 1. Load File Paths ---
    print("--- 1. Loading File Paths ---")
    
    base_dir = Path(args.data_path)
    
    # Check if the path is correct before continuing
    if not base_dir.exists():
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"ERROR: The path '{base_dir}' does not exist.")
        print("Please provide the correct path to the 'complete_set' folder")
        print(f"using the --data_path argument.")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        exit()
    else:
        print(f"Using base directory: {base_dir}")

    
    train_dir = base_dir / 'training_set'
    test_dir = base_dir / 'testing_set'

    # Build list of training images (excluding masks)
    all_train_files = list(train_dir.rglob('*.png'))
    train_image_paths = [path for path in all_train_files if '_mask' not in path.name]

    train_data = []
    for img_path in train_image_paths:
        label = img_path.parent.name
        mask_name = img_path.stem + '_mask.png'
        mask_path = img_path.with_name(mask_name)
        train_data.append({
            'label': label,
            'image_path': str(img_path),
            'mask_path': str(mask_path)
        })
    train_df = pd.DataFrame(train_data)
    print(f"Found {len(train_df)} training samples.")

    # Build list of test images (excluding masks)
    all_test_files = list(test_dir.rglob('*.png'))
    test_image_paths = [path for path in all_test_files if '_mask' not in path.name]
    
    test_data = []
    for img_path in test_image_paths:
        mask_name = img_path.stem + '_mask.png'
        mask_path = img_path.with_name(mask_name)
        test_data.append({
            'image_path': str(img_path),
            'mask_path': str(mask_path)
        })
    test_df = pd.DataFrame(test_data)
    print(f"Found {len(test_df)} test samples.")

    # --- 2. Clean and Validate Data ---
    print("\n--- 2. Validating Data ---")
    clean_train_df = validate_data(train_df, "Training Data")
    clean_test_df = validate_data(test_df, "Testing Data")

    # --- 3. Create Datasets and DataLoaders ---
    if len(clean_train_df) == 0:
        print("ERROR: No valid training data found. Cannot create DataLoader.")
        print("Please check your 'base_dir' path and data integrity.")
    else:
        print("\n--- 3. Creating Datasets and DataLoaders ---")
        IMG_SIZE = 256
        
        # Define transforms
        simple_transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor()
        ])
        
        # Create Datasets
        train_dataset = BreastCancerDataset(df=clean_train_df, transform=simple_transform)
        test_dataset = BreastCancerDataset(df=clean_test_df, transform=simple_transform)
        
        # Create DataLoaders
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 for Windows/cloud compatibility
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )
        print(f"Created train_loader with {len(train_loader)} batches.")
        print(f"Created test_loader with {len(test_loader)} batches.")

        # --- 4. Initialize Model, Loss, and Optimizer ---
        print("\n--- 4. Initializing Training Components ---")
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Initialize Model
        model = UNET(in_channels=3, out_channels=1).to(device)
        
        # Initialize Loss Function
        criterion = nn.BCEWithLogitsLoss()
        
        # Initialize Optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        
        print("Model, Loss, and Optimizer initialized successfully.")

        # --- 5. Start The Training Loop ---
        print("\n--- 5. Starting Model Training ---")
        
        for epoch in range(args.epochs):
            start_time = time.time()
            
            # --- Training Phase ---
            model.train() # Set model to training mode
            running_loss = 0.0
            
            for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
                images = images.to(device)
                masks = masks.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            avg_train_loss = running_loss / len(train_loader)
            
            # --- Validation Phase ---
            model.eval() # Set model to evaluation mode
            val_loss = 0.0
            
            with torch.no_grad(): # Disable gradient calculations
                for images, masks in tqdm(test_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]  "):
                    images = images.to(device)
                    masks = masks.to(device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / (len(test_loader) + 1e-6)
            end_time = time.time()
            
            # Print a summary for the epoch
            print(f"\nEpoch {epoch+1}/{args.epochs} | Time: {end_time - start_time:.2f}s")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Valid Loss: {avg_val_loss:.4f}\n")

        print("--- Training Finished ---")

        # Save the final model
        torch.save(model.state_dict(), args.output_model)
        print(f"Model saved successfully as '{args.output_model}'")

# -----------------------------------------------------------------
# 5. ARGUMENT PARSING AND SCRIPT EXECUTION
# -----------------------------------------------------------------
if __name__ == '__main__':
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="U-Net Training Script for Breast Cancer Segmentation")
    
    parser.add_argument('--data_path', type=str, required=True, 
                        help="Path to the 'complete_set' directory.")
    parser.add_argument('--epochs', type=int, default=5, 
                        help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=8, 
                        help="Batch size for training and validation.")
    parser.add_argument('--learning_rate', type=float, default=1e-4, 
                        help="Learning rate for the optimizer.")
    parser.add_argument('--output_model', type=str, default='unet_breast_cancer.pth', 
                        help="Path to save the trained model.")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Run the main training function
    main(args)


        




            



    
