import os
import socket
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

class SimpleDataset(Dataset):
    def __init__(self):
        print("Initializing dataset...")
        self.data = datasets.MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
        print("Dataset initialized.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = self.fc(x)
        return x

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy}%")

def train(rank, world_size):
    device = torch.device(f"cuda:{rank % 2}")  # Assuming 2 GPUs per node
    torch.cuda.set_device(device)

    model = SimpleModel().to(device)
    ddp_model = DDP(model, device_ids=[device])

    dataset = SimpleDataset()
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    for epoch in range(10):
        sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = ddp_model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Rank {rank}: Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item()}")

    # Evaluation
    if rank % 2 == 0:  # Evaluate only on one GPU per node
        test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        evaluate(model, test_loader, device)

    # VRAM Usage
    current_vram = torch.cuda.memory_allocated(device) / (1024 ** 3)
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    print(f"Rank {rank}: Current VRAM Usage: {current_vram} GB, Peak: {peak_vram} GB")

def main():
    parser = argparse.ArgumentParser(description="PyTorch Distributed Training Example")
    parser.add_argument('--one_node', action='store_true', help='Run training on a single node')
    args = parser.parse_args()

    world_size = 4  # Total GPUs across all nodes
    if args.one_node:
        world_size = 1
        rank = 0
    else:
        hostnames = os.getenv('WORKER_HOSTNAMES').split(',')
        node_rank = hostnames.index(socket.gethostname())
        rank = node_rank * 2  # Assuming 2 GPUs per node

    setup(rank, world_size)
    train(rank, world_size)
    cleanup()

if __name__ == "__main__":
    main()
