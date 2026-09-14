import torch
import torchvision
import torch.nn as nn
import spikingjelly
from spikingjelly.activation_based import ann2snn
from tqdm import tqdm
from spikingjelly.activation_based.ann2snn.sample_models import mnist_cnn
import numpy as np
import matplotlib.pyplot as plt
try:
    from .loader import CustomLoader
except Exception as e:
    from loader import CustomLoader

global_seed = 68862025
device = 'cuda' # use 'cpu' if CUDA not available
download_dataset = False # downloads MNIST
download_model = False # downloads a 3 layer CNN classifier

# variables for stat hooks

class hyperparameters:
    T = 10
    batch_size = 100

def main(eval_fn,
        backbone=None, 
        head=None, 
        conversion = None, 
        download_dataset=False,
        device = 'cuda',
        dataset_dir=None):
    ''' Convert an ANN and evaluate it.
        ANN can be divided into a backbone that is frozen and a head that is converted'''

    torch.random.manual_seed(global_seed)
    torch.cuda.manual_seed(global_seed)
    device = device # use 'cpu' if CUDA not available
    dataset_dir = dataset_dir if dataset_dir is not None else './datasets/mnist' # folder to download dataset
    

    T = hyperparameters.T
    batch_size = hyperparameters.batch_size
    
    # Define data loaders

    train_data_dataset = torchvision.datasets.MNIST(
        root=dataset_dir,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=download_dataset)
    test_data_dataset = torchvision.datasets.MNIST(
        root=dataset_dir,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=download_dataset)

    if backbone:
        test_data_dataset = CustomLoader(test_data_dataset, backbone, device = device) 
        train_data_dataset = CustomLoader(train_data_dataset, backbone, device = device) 

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_data_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False)
    
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_data_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False)

    # if head is none, test the backbone directly
    if head is not None:
        val_model = conversion(head, train_data_loader)
    else:
        val_model = backbone

    print('Simulating...')
    return val_model, eval_fn(val_model, device, train_data_loader, test_data_loader, T=T if head is not None else None)

spike_counts = {}
def count_spikes_hook(module, input, output):
    name = module.__class__.__name__
    spike_counts[name] = spike_counts.get(name, 0) + output.sum().item()

def count_macs_hook(module, input, output):
    print(module)

class conversion_config:
    presets = {
        "max": "max",  # scale = the single largest activation seen (sensitive to outliers)
        "99.9%": "99.9%",  # scale = the 99.9th percentile activation (ignores rare outliers)
        "1/2 max": 1.0 / 2,  # scale = half the max activation (needs 2x the spikes of "max" per input)
        "1/4 max": 1.0 / 4,  # scale = a quarter of the max activation (needs 4x the spikes of "max" per input)
    }
    preset = "1/4 max"  # change this (e.g. conversion_config.preset = "max") to switch converters

def conversion_job(head, train_data_loader):
    mode = conversion_config.presets[conversion_config.preset]
    print('---------------------------------------------')
    print(f'Converting using preset={conversion_config.preset!r} (mode={mode!r})')

    model_converter = ann2snn.Converter(mode=mode, dataloader=train_data_loader)
    return model_converter(head)

def val(net, device, train_data_loader, test_data_loader, T=None):
    # This eval_fn tests an SNN (or an ANN if T=None) on the test loader 
    # And registers a hook to store spike counts of first spiking layer into spike_counts

    # Try to make modified versions of this evaluation function to collect stats.
    # Make sure that in your code you only use test_data_loader to evaluate performance.
    # Use only train_data_loader to calculate stats of layers using hooks on their appropriate nodes.
    
    net.eval().to(device)
    correct = 0.0
    total = 0.0
    if T is not None:
        corrects = np.zeros(T)
    with torch.no_grad():
        for batch, (img, label) in enumerate(tqdm(test_data_loader)):
            img = img.to(device)
            if T is None:
                # not a spiking model
                out = net(img)  
                correct += (out.argmax(dim=1) == label.to(device)).float().sum().item()
            else:

                for m in net.modules():
                    if hasattr(m, 'reset'):
                        m.reset()
                for t in range(T):
                        
                    hooks=[count_spikes_hook]
                    hook_nodes = [getattr(net, "").spiking0.if_node]

                    handles = []
                    for hook, hook_node in zip(hooks, hook_nodes):
                        handles.append(hook_node.register_forward_hook(hook))

                    #forward pass
                    x = net(img)

                    for handle in handles:
                        handle.remove()

                    if t == 0:
                        out = x
                    else:
                        out += x
                    corrects[t] += (out.argmax(dim=1) == label.to(device)).float().sum().item()
            total += out.shape[0]

    return correct / total if T is None else corrects / total


if __name__ == '__main__':

    # import model
    if download_model:
        print('Downloading SJ-mnist-cnn_model-sample.pth...')
        ann2snn.download_url("https://ndownloader.figshare.com/files/34960191", './SJ-mnist-cnn_model-sample.pth')

    model = mnist_cnn.CNN().to(device)
    model.load_state_dict(torch.load('SJ-mnist-cnn_model-sample.pth'))

    # divide the model into a backbone of the first #split_index modules of the CNN + a head comprising the remaining modules
    split_index = 4
    backbone = model.network[:split_index]
    head = model.network[split_index:]
    dataset = 'C:/python/sj/spikingjelly/datasets/mnist'
    print(main(eval_fn = val, 
        backbone = backbone, 
        head = head, 
        conversion =  conversion_job, 
        download_dataset = download_dataset,
        device = 'cuda',
        dataset_dir = dataset))
    print(spike_counts)

