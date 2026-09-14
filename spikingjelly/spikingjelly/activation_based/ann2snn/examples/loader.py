import torch
from torch.utils.data import Dataset, DataLoader

class CustomLoader(Dataset):
    def __init__(self, base_dataset, cnn_model, device='cpu', detach=True):
        """
        base_dataset : any PyTorch Dataset (e.g., torchvision.datasets.MNIST)
        cnn_model    : a CNN (nn.Module) to apply as transform
        device       : device to run the CNN on ('cpu' or 'cuda')
        detach       : if True, return detached tensor (no gradient)
        """
        self.base_dataset = base_dataset
        self.cnn_model = cnn_model.to(device)
        self.device = device
        self.detach = detach
        self.cnn_model.eval()  # typically used in inference mode

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Get the original data and label
        x, y = self.base_dataset[idx]
        
        # Ensure input is tensor and batch dimension
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        if x.ndim == 3:  # (C, H, W)
            x = x.unsqueeze(0)  # add batch dim
        
        # Move to device and apply CNN
        with torch.no_grad():
            out = self.cnn_model(x.to(self.device))
        # Optionally detach and move back to CPU for DataLoader
        if self.detach:
            out = out.detach().squeeze(0)
        
        return out, y