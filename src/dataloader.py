from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from augmentation import Augmentation
import os

from dataset import UnlabeledImageDataset

def get_cats_dogs_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def get_cifar10_transforms():
    base_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.RandomCrop(227, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    val_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Create augmentation wrapper
    train_transform = Augmentation(base_transform)

    return train_transform, val_transform

def get_cats_dogs_loaders(config, world_size, rank):
    train_transform, val_transform = get_cats_dogs_transforms()
    
    train_dataset = datasets.ImageFolder(config['data']['train_dir'], transform=train_transform)
    val_dataset = datasets.ImageFolder(config['data']['val_dir'], transform=val_transform)
    test_dataset = UnlabeledImageDataset(config['data']['test_dir'], transform=val_transform)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        sampler=train_sampler,
        num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        sampler=val_sampler,
        num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False, 
        num_workers=config['data']['num_workers']
    )
    
    return train_loader, val_loader, test_loader, train_dataset.classes

def get_cifar10_loaders(config, world_size, rank):
    train_transform, val_transform = get_cifar10_transforms()
    
    train_dataset = datasets.CIFAR10(
        root=config['data']['data_dir'], 
        train=True, 
        download=True, 
        transform=train_transform
    )
    
    val_dataset = datasets.CIFAR10(
        root=config['data']['data_dir'], 
        train=False, 
        download=True, 
        transform=val_transform
    )
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        sampler=train_sampler,
        num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        sampler=val_sampler,
        num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
        pin_memory=True
    )
    
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    return train_loader, val_loader, None, classes

def get_data_loaders(config, world_size, rank):
    dataset_name = config['data'].get('dataset', 'cats_dogs')
    
    if dataset_name.lower() == 'cifar10':
        return get_cifar10_loaders(config, world_size, rank)
    else:  # default to cats and dogs
        return get_cats_dogs_loaders(config, world_size, rank)