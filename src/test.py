import argparse
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from torch.nn.parallel import DistributedDataParallel as DDP
from socket import gethostname
import torch.distributed as dist
from model import AlexNet, VGG16, AlexNetPretrained, VGG16Pretrained
from dataloader import get_data_loaders
import yaml
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Inference a trained model')
    parser.add_argument('--config', type=str, required=True, help='Path to config.yaml')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model-type', type=str, required=True, 
                        choices=['alexnet', 'vgg16', 'alexnet_pretrained', 'vgg16_pretrained'],
                        help='Type of model to Inference')
    return parser.parse_args()

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

def load_model(config, model_path, local_rank):

    # Create model with config settings
    model = get_model(config).to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Load state dict
    model.load_state_dict(state_dict)
    return model

def plot_confusion_mat(conf_matrix, classes, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def val_model(model, val_loader, local_rank, classes):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(local_rank)
            labels = labels.to(local_rank)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    # # Calculate ROC-AUC (for multi-class, we use micro-averaging)
    # roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    
    # Create confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        # 'roc_auc': roc_auc,
        'confusion_matrix': conf_matrix
    }

def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

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

    
    # Get data loaders
    _, val_loader, _, classes = get_data_loaders(config, world_size, rank)
    
    # Load model
    model = load_model(config, args.model_path, local_rank)
    
    # val model
    metrics = val_model(model, val_loader, local_rank, classes)
    
    # Print metrics
    print(f"\nval Results:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    # print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Plot and save confusion matrix
    save_dir = Path(args.model_path).parent
    plot_confusion_mat(
        metrics['confusion_matrix'],
        classes,
        save_path=save_dir / 'confusion_matrix.png'
    )
    print(f"\nConfusion matrix has been saved to {save_dir / 'confusion_matrix.png'}")
    dist.destroy_process_group()

if __name__ == '__main__':
    main()