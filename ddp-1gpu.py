import os
import socket
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# Define a simple dataset (for example, MNIST)
class SimpleDataset(Dataset):
    def __init__(self):
        print("Initializing dataset...")
        self.data = datasets.MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
        print("Dataset initialized.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Define a simple model
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
    print(f"Rank {rank}: Initializing process group...")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(f"Rank {rank}: Process group initialized.")

def cleanup():
    print("Cleaning up...")
    dist.destroy_process_group()
    print("Cleaned up.")
    
def evaluate(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient tracking
        for data, target in test_loader:
            data, target = data.to('cuda'), target.to('cuda')
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy}%")

def train(rank, world_size):
    print(f"Rank {rank}: Starting training...")
    setup(rank, world_size)

    # Create model and move it to GPU with id rank
    model = SimpleModel().to('cuda')
    print(f"Rank {rank}: Model created and moved to GPU.")
    
    # Only 1 GPU is used in this example
    device_id = 0
    ddp_model = DDP(model, device_ids=[device_id])

    dataset = SimpleDataset()
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler)

    criterion = nn.CrossEntropyLoss().to('cuda')
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    for epoch in range(10):
        print(f"Rank {rank}: Starting epoch {epoch + 1}...")
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to('cuda'), target.to('cuda')
            optimizer.zero_grad()
            outputs = ddp_model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f"Rank {rank}: Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item()}")
        print(f"Rank {rank}: Finished epoch {epoch + 1}.")

    cleanup()
    print(f"Rank {rank}: Training completed.")
    
    # Evaluation step
    test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    evaluate(model, test_loader)

    # Print VRAM usage
    current_vram = torch.cuda.memory_allocated(device_id) / (1024 ** 3)  # Convert to GB
    peak_vram = torch.cuda.max_memory_allocated(device_id) / (1024 ** 3)  # Convert to GB
    print(f"Rank {rank}: Current VRAM Usage: {current_vram} GB, Peak: {peak_vram} GB")

def main():
    parser = argparse.ArgumentParser(description="PyTorch Distributed Training Example")
    parser.add_argument('--one_node', action='store_true', help='Run training on a single node')
    args = parser.parse_args()

    if args.one_node:
        print("Running in single node mode.")
        world_size = 1
        rank = 0
    else:
        print("Running in distributed mode.")
        world_size = 2
        hostnames = os.getenv('WORKER_HOSTNAMES').split(',')
        rank = hostnames.index(socket.gethostname())

    train(rank, world_size)

if __name__ == "__main__":
    main()