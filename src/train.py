from tqdm import tqdm
from socket import gethostname
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.utils.model_zoo as model_zoo

from torch.nn.parallel import DistributedDataParallel as DDP

from utils import load_config, ExperimentTracker
from model import AlexNet, VGG16, AlexNetPretrained, VGG16Pretrained
from dataloader import get_data_loaders

def set_random_seed(seed):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_model(model, train_loader, val_loader, config, device, tracker):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        patience=config['training']['scheduler_patience'], 
        factor=config['training']['scheduler_factor']
    )
    
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    
    for epoch in range(config['training']['num_epochs']):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["training"]["num_epochs"]}')
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            train_bar.set_postfix({'loss': running_loss/len(train_loader), 
                                 'acc': 100.*correct/total})
        
        train_loss = running_loss/len(train_loader)
        train_acc = 100.*correct/total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                val_predictions.extend(predicted.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        val_loss = running_loss/len(val_loader)
        val_acc = 100.*correct/total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1}/{config["training"]["num_epochs"]}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        scheduler.step(val_loss)
    
    # Save final confusion matrix
    tracker.save_confusion_matrix(val_targets, val_predictions, train_loader.dataset.classes)
    return train_losses, train_accuracies, val_losses, val_accuracies

def get_model(config):
    model_name = config['model'].get('name')
    is_pretrained = config['model'].get('is_pretrained')

    assert model_name in ['alexnet', 'vgg16'], f"Model should be either one {['alexnet', 'vgg16']}"
    if model_name == 'alexnet':
        if is_pretrained:
            model = AlexNetPretrained(num_classes=config['model'].get('num_classes'))
        else:
            model = AlexNet(num_classes=config['model'].get('num_classes'))

    if model_name == 'vgg16':
        if is_pretrained:
            model = VGG16(num_classes=config['model'].get('num_classes'))
        else:
            model = VGG16Pretrained(num_classes=config['model'].get('num_classes'))

    return model

def main():
    # Load config
    config = load_config('config.yaml')
    
    # Set random seed if specified in config
    if 'random_seed' in config:
        set_random_seed(config['random_seed'])
        print(f"Set random seed to {config['random_seed']} for reproducibility")

    # Setup distributed training on multinode & multi-GPU
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["SLURM_PROCID"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    assert gpus_per_node == torch.cuda.device_count()
    
    print(f"Hello from rank {rank} of {world_size} on {gethostname()} where there are" \
          f" {gpus_per_node} allocated GPUs per node.", flush=True)
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    if rank == 0: 
        print(f"Group initialized? {dist.is_initialized()}", flush=True)
    
    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)
    
    # Initialize experiment tracker
    tracker = ExperimentTracker(config)
    
    # Get data loaders and class names
    train_loader, val_loader, test_loader, classes = get_data_loaders(config, world_size, rank)
    
    # Update num_classes in config based on dataset
    config['model']['num_classes'] = len(classes)
    
    # Create model with config settings
    model = get_model(config).to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # Train model
    train_losses, train_accuracies, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, config, local_rank, tracker
    )
    
    # Only save result and run test when all resources are collected back to the first GPU
    if rank == 0:
        # Save training history and metrics
        tracker.save_metrics(train_losses, train_accuracies, val_losses, val_accuracies)
        tracker.plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies)
        tracker.save_model(model, classes)
        
        # Predict on test set
        model.eval()
        predictions = []
        filenames = []
        with torch.no_grad():
            for images, image_names in test_loader:
                images = images.to(local_rank)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                predictions.extend([classes[idx] for idx in predicted.cpu().numpy()])
                filenames.extend(image_names)
        
        # Save predictions using the tracker
        tracker.save_predictions(filenames, predictions)
    dist.destroy_process_group()

if __name__ == '__main__':
    main()