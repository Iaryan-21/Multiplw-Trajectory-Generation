import torch
from torch.utils.data import DataLoader
from datasets import NuScenesDataset  # Ensure this points to your dataset module

def test_dataloader():
    trainset = NuScenesDataset('./dataset', 'mini_train')
    train_loader = DataLoader(trainset, shuffle=True, batch_size=2, num_workers=0)  # Set batch_size to a small value for testing

    for i, (img, state, gt) in enumerate(train_loader):
        print(f"Batch {i}: Image shape: {img.shape}, State shape: {state.shape}, Ground Truth shape: {gt.shape}")
        if i == 2:  # Only print first few batches for testing
            break

if __name__ == "__main__":
    test_dataloader()
