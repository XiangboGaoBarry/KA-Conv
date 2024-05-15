
import torch
from torch import nn
from torchvision import datasets, transforms
from tqdm import tqdm
import random
import numpy as np
import argparse
import time

from kaconv.convkan import ConvKAN
from kaconv.kaconv import FastKANConvLayer
from torch.nn import Conv2d, BatchNorm2d

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

argparser = argparse.ArgumentParser()
argparser.add_argument("--model", type=str, default="kanconv", choices=["kanconv_small", "kanconv_tiny", "kanconv", "mlp", "convkan_efficient", "convkan_fast"], help="model name")
argparser.add_argument("--epochs", type=int, default=150, help="number of epochs")
argparser.add_argument("--seed", type=int, default=44, help="random seed")
argparser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
argparser.add_argument("--batch_size", type=int, default=128, help="batch size")
argparser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
argparser.add_argument("--momentum", type=float, default=0.9, help="momentum")
argparser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"], help="optimizer")
argparser.add_argument("--kan_type", type=str, default="RBF", choices=["RBF", "Poly", "Chebyshev", "Fourier", "BSpline"], help="kernel type")


args = argparser.parse_args()


set_seed(args.seed)

# Define the model
if args.model == "kanconv_small":
    model = nn.Sequential(
        FastKANConvLayer(3, 8, padding=1, kernel_size=3, stride=1, kan_type=args.kan_type),
        BatchNorm2d(8),
        FastKANConvLayer(8, 32, padding=1, kernel_size=3, stride=2, kan_type=args.kan_type),
        BatchNorm2d(32),
        FastKANConvLayer(32, 10, padding=1, kernel_size=3, stride=2, kan_type=args.kan_type),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    ).cuda()

elif args.model == "kanconv_tiny":
    model = nn.Sequential(
        FastKANConvLayer(3, 8, padding=1, kernel_size=3, stride=1, kan_type=args.kan_type),
        BatchNorm2d(8),
        FastKANConvLayer(8, 16, padding=1, kernel_size=3, stride=2, kan_type=args.kan_type),
        BatchNorm2d(16),
        FastKANConvLayer(16, 10, padding=1, kernel_size=3, stride=2, kan_type=args.kan_type),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    ).cuda()
    
elif args.model == "kanconv":
    model = nn.Sequential(
        FastKANConvLayer(3, 32, padding=1, kernel_size=3, stride=1, kan_type=args.kan_type),
        BatchNorm2d(32),
        FastKANConvLayer(32, 32, padding=1, kernel_size=3, stride=2, kan_type=args.kan_type),
        BatchNorm2d(32),
        FastKANConvLayer(32, 10, padding=1, kernel_size=3, stride=2, kan_type=args.kan_type),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    ).cuda()
elif args.model == "mlp":
    model = nn.Sequential(
        Conv2d(3, 32, padding=1, kernel_size=3, stride=1),
        nn.ReLU(),
        BatchNorm2d(32),
        Conv2d(32, 32, padding=1, kernel_size=3, stride=2),
        nn.ReLU(),
        BatchNorm2d(32),
        Conv2d(32, 10, padding=1, kernel_size=3, stride=2),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    ).cuda()
    
elif args.model == "convkan_efficient":
    model = nn.Sequential(
        ConvKAN(3, 32, padding=1, kernel_size=3, stride=1, kan_type = "EfficientKAN"),
        BatchNorm2d(32),
        FastKANConvLayer(32, 32, padding=1, kernel_size=3, stride=2, kan_type=args.kan_type),
        BatchNorm2d(32),
        FastKANConvLayer(32, 10, padding=1, kernel_size=3, stride=2, kan_type=args.kan_type),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    ).cuda()
    
elif args.model == "convkan_fast":
    model = nn.Sequential(
        ConvKAN(3, 32, padding=1, kernel_size=3, stride=1, kan_type = "FastKAN"),
        BatchNorm2d(32),
        FastKANConvLayer(32, 32, padding=1, kernel_size=3, stride=2, kan_type=args.kan_type),
        BatchNorm2d(32),
        FastKANConvLayer(32, 10, padding=1, kernel_size=3, stride=2, kan_type=args.kan_type),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    ).cuda()
    
else:
    raise ValueError(f"Unknown model: {args.model}")
    
    
    
# print number of parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_params:,} total parameters.')

# Define transformations and download the MNIST dataset
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=64)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=64)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
if args.optimizer == "adamw":
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.0001)


# print("mlp")
pbar = tqdm(range(args.epochs))
acc = 0
best_acc = 0
throughput = 0
for epoch in pbar:
    # Train the model
    model.train()
    
    for i, (x, y) in enumerate(train_loader):
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        pbar.set_description(f'Loss: {loss.item():.2e}, Acc {100 * acc:.2f}%, best Acc {100 * best_acc:.2f}%, Throughput {throughput:.2f} samples/s')

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        time_taken = 0
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            # time asynchronous
            start = time.time()
            y_hat = model(x)
            time_taken += time.time() - start
            _, predicted = torch.max(y_hat, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            acc = correct / total
            pbar.set_description(f'Loss: {loss.item():.2e}, Acc {100 * acc:.2f}%, best Acc {100 * best_acc:.2f}%, Throughput {throughput:.2f} samples/s')
        if acc > best_acc:
            best_acc = acc
        throughput = 1 / time_taken * total
    scheduler.step()

print(throughput)
    
