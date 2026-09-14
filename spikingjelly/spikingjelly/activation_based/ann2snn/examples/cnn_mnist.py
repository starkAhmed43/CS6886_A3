import torch
import torchvision
import torch.nn as nn
import spikingjelly
from spikingjelly.activation_based import ann2snn
from spikingjelly.activation_based import neuron
from tqdm import tqdm
from spikingjelly.activation_based.ann2snn.sample_models import mnist_cnn
import numpy as np
import matplotlib.pyplot as plt
import threading
import concurrent.futures
import multiprocessing
import os
import warnings
import contextlib
try:
    from .loader import CustomLoader
except Exception as e:
    from loader import CustomLoader

global_seed = 68862025
device = 'cuda' # use 'cpu' if CUDA not available
download_dataset = False # downloads MNIST
download_model = False # downloads a 3 layer CNN classifier

# variables for stat hooks

class _Hyperparameters(threading.local):
    def __init__(self):
        self.T = 10
        self.batch_size = 100

hyperparameters = _Hyperparameters()

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

class _ThreadLocalDict(threading.local):
    def __init__(self):
        self.data = {}
    def __getitem__(self, key):
        return self.data[key]
    def __setitem__(self, key, value):
        self.data[key] = value
    def __contains__(self, key):
        return key in self.data
    def get(self, *args, **kwargs):
        return self.data.get(*args, **kwargs)
    def items(self):
        return self.data.items()
    def keys(self):
        return self.data.keys()
    def values(self):
        return self.data.values()
    def clear(self):
        self.data.clear()
    def __iter__(self):
        return iter(self.data)
    def __len__(self):
        return len(self.data)
    def __repr__(self):
        return repr(self.data)

spike_counts = _ThreadLocalDict()

def count_spikes_hook(module, input, output):
    name = module.__class__.__name__
    spike_counts[name] = spike_counts.get(name, 0) + output.sum().item()

def make_spike_count_hook(name):
    def hook(module, input, output):
        spike_counts[name] = spike_counts.get(name, 0) + output.sum().item()
    return hook

conv_macs = {}

def make_mac_count_hook(name):
    def hook(module, input, output):
        out_channels, out_h, out_w = output.shape[1], output.shape[2], output.shape[3]
        in_channels = module.in_channels
        kh, kw = module.kernel_size
        conv_macs[name] = out_channels * out_h * out_w * in_channels * kh * kw
    return hook

def compute_conv_macs(net, device='cuda', input_shape=(1, 1, 28, 28)):
    """Run one dummy forward pass through `net`, hooking every Conv2d layer to
    record its MAC count (based on actual runtime output shape) into `conv_macs`.
    Returns `conv_macs`."""
    conv_macs.clear()
    conv_layers = [
        (name, module)
        for name, module in net.named_modules()
        if isinstance(module, nn.Conv2d)
    ]
    handles = [
        module.register_forward_hook(make_mac_count_hook(f"conv{i + 1} (layer {name})"))
        for i, (name, module) in enumerate(conv_layers)
    ]
    dummy_input = torch.zeros(*input_shape, device=device)
    with torch.no_grad():
        net(dummy_input)
    for handle in handles:
        handle.remove()
    return conv_macs

def compute_fan_out(net):
    """Static (no forward pass needed) fan-out estimate per Conv2d layer: how many
    output positions (times out_channels) one input element feeds into, approximated
    as kernel_h * kernel_w * out_channels. Keyed by ordinal position (0, 1, 2, ...)
    matching conv layer order in `net`."""
    fan_out = {}
    conv_layers = [module for module in net.modules() if isinstance(module, nn.Conv2d)]
    for i, module in enumerate(conv_layers):
        kh, kw = module.kernel_size
        fan_out[i] = module.out_channels * kh * kw
    return fan_out

def compute_memory_energy(net, device='cuda', input_shape=(1, 1, 28, 28), write_energy_per_64B=25.0, dtype_bytes=4):
    """Estimate the memory-write energy of one forward pass through `net`.

    Since each [Conv2d, BatchNorm2d, ReLU, AvgPool2d] block is treated as fused for
    memory accounting, only the block's final output (the AvgPool2d output) is
    counted as a memory write and not the intermediate Conv/BatchNorm/ReLU
    activations within that block. Any trailing Linear layer's output is costed
    separately (Flatten is a reshape, not an actual memory write, so it's skipped).

    write_energy_per_64B: energy in nJ to write one 64-byte vector to memory.
    dtype_bytes: bytes per activation element (4 for float32).

    Returns a dict of {layer_name: energy_in_nJ}, one entry per fused block plus
    one for the final Linear layer.
    """
    memory_energy = {}

    def make_hook(name):
        def hook(module, input, output):
            num_bytes = output.numel() * dtype_bytes
            memory_energy[name] = (num_bytes / 64) * write_energy_per_64B
        return hook

    block_layers = [
        (name, module) for name, module in net.named_modules()
        if isinstance(module, nn.AvgPool2d)
    ]
    linear_layers = [
        (name, module) for name, module in net.named_modules()
        if isinstance(module, nn.Linear)
    ]

    handles = [
        module.register_forward_hook(make_hook(f"block{i + 1} (fused conv+bn+relu+avgpool, layer {name})"))
        for i, (name, module) in enumerate(block_layers)
    ]
    handles += [
        module.register_forward_hook(make_hook(f"linear (layer {name})"))
        for name, module in linear_layers
    ]

    dummy_input = torch.zeros(*input_shape, device=device)
    with torch.no_grad():
        net(dummy_input)

    for handle in handles:
        handle.remove()

    return memory_energy

class _ConversionConfig(threading.local):
    def __init__(self):
        self.presets = {
            "max": "max",  # scale = the single largest activation seen (sensitive to outliers)
            "99.9%": "99.9%",  # scale = the 99.9th percentile activation (ignores rare outliers)
            "1/2 max": 1.0 / 2,  # scale = half the max activation (needs 2x the spikes of "max" per input)
            "1/4 max": 1.0 / 4,  # scale = a quarter of the max activation (needs 4x the spikes of "max" per input)
        }
        self.preset = "1/4 max"  # change this (e.g. conversion_config.preset = "max") to switch converters

conversion_config = _ConversionConfig()

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
        spike_counts.clear()
    with torch.no_grad():
        for batch, (img, label) in enumerate(tqdm(test_data_loader, desc="Evaluating accuracy (test set)")):
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

                    #forward pass
                    x = net(img)

                    if t == 0:
                        out = x
                    else:
                        out += x
                    corrects[t] += (out.argmax(dim=1) == label.to(device)).float().sum().item()
            total += out.shape[0]

        # Spike-count stats: gathered on train_data_loader,
        if T is not None:
            for batch, (img, label) in enumerate(tqdm(train_data_loader, desc="Counting spikes (train set)")):
                img = img.to(device)

                for m in net.modules():
                    if hasattr(m, 'reset'):
                        m.reset()
                for t in range(T):

                    hook_nodes = {
                        name: module
                        for name, module in net.named_modules()
                        if isinstance(module, neuron.IFNode)
                    }
                    handles = [
                        module.register_forward_hook(make_spike_count_hook(name))
                        for name, module in hook_nodes.items()
                    ]

                    #forward pass
                    net(img)

                    for handle in handles:
                        handle.remove()

    return correct / total if T is None else corrects / total


def _search_worker(
    checkpoint_path, blocks_converted, T, preset,
    conv_macs, fan_out, ann_energy, conv_names, num_train_images,
    device, dataset_dir, download_dataset, mac_energy, ac_energy,
):
    warnings.filterwarnings("ignore")
    with open(os.devnull, "w") as devnull, \
            contextlib.redirect_stdout(devnull), \
            contextlib.redirect_stderr(devnull):
        model = mnist_cnn.CNN().to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        n = len(conv_names)
        split_index = (n - blocks_converted) * 4
        is_full = blocks_converted == n
        backbone = None if is_full else model.network[:split_index]
        head = model.network if is_full else model.network[split_index:]

        conversion_config.preset = preset
        hyperparameters.T = T

        _, accuracy = main(
            eval_fn=val,
            backbone=backbone,
            head=head,
            conversion=conversion_job,
            download_dataset=download_dataset,
            device=device,
            dataset_dir=dataset_dir,
        )
        counts = dict(spike_counts)

    backbone_names = conv_names[: n - blocks_converted]
    converted_names = conv_names[n - blocks_converted:]
    backbone_energy = sum(conv_macs[name] for name in backbone_names) * mac_energy

    converted_energy = 0.0
    for k, _ in enumerate(converted_names):
        spike_key = f"spiking{k}.if_node"
        avg_spikes = counts.get(spike_key, 0.0) / num_train_images
        orig_idx = n - blocks_converted + k
        converted_energy += avg_spikes * fan_out[orig_idx] * ac_energy

    snn_energy = backbone_energy + converted_energy
    energy_savings = 1 - snn_energy / ann_energy

    return {
        "blocks_converted": blocks_converted,
        "T": T,
        "preset": preset,
        "accuracy": accuracy[-1],
        "accuracy_curve": accuracy,
        "snn_energy_nJ": snn_energy,
        "energy_savings": energy_savings,
    }


def search_conversion_configs(
    checkpoint_path,
    conv_macs,
    fan_out,
    ann_energy,
    block_options=(1, 2, 3),
    T_options=(1, 2, 5, 10, 20),
    presets=None,
    device='cuda',
    dataset_dir=None,
    download_dataset=False,
    mac_energy=4.6,
    ac_energy=0.6,
    max_workers=32,
):
    """Search (blocks_converted, T, preset) configurations in parallel OS processes
    (not threads), since a small SNN forward pass underutilizes the GPU and threads
    previously deadlocked here under CUDA. Each worker process independently loads
    the model from `checkpoint_path` onto its own CUDA context, so this uses more
    GPU memory than a thread-based approach would (one CUDA context per worker) —
    reduce `max_workers` if you see out-of-memory errors.

    `conv_macs` and `fan_out` should come from `compute_conv_macs`/`compute_fan_out`
    run on the full, unconverted `model.network` (ordered conv1, conv2, conv3, ...).
    `ann_energy` is the unconverted CNN's total energy (e.g. sum(conv_macs.values()) * mac_energy).

    Returns a list of result dicts, one per configuration, each with:
    blocks_converted, T, preset, accuracy (final-timestep), accuracy_curve,
    snn_energy_nJ, energy_savings.
    """
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    if presets is None:
        presets = list(conversion_config.presets.keys())

    conv_names = list(conv_macs.keys())
    num_train_images = len(torchvision.datasets.MNIST(
        root=dataset_dir, train=True, download=download_dataset,
    ))

    configs = [
        (b, T, p) for b in block_options for T in T_options for p in presets
    ]

    ctx = multiprocessing.get_context('spawn')
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = [
            executor.submit(
                _search_worker,
                checkpoint_path, b, T, p,
                conv_macs, fan_out, ann_energy, conv_names, num_train_images,
                device, dataset_dir, download_dataset, mac_energy, ac_energy,
            )
            for b, T, p in configs
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results


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

