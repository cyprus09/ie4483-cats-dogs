import os
import yaml
import json
import torch
import numpy as np
from datetime import datetime
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import csv

class ExperimentTracker:
    def __init__(self, config):
        self.config = config
        self.base_dir = config['experiment']['output_dir']
        self.experiment_dir = self._create_experiment_dir()
        self._save_config()

    def _create_experiment_dir(self):
        label = self.config['experiment']['label']
        exp_dir = os.path.join(self.base_dir, label)
        os.makedirs(exp_dir, exist_ok=True)
        return exp_dir

    def _save_config(self):
        config_path = os.path.join(self.experiment_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)

    def save_metrics(self, train_losses, train_accuracies, val_losses, val_accuracies):
        metrics = {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        }
        metrics_path = os.path.join(self.experiment_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump({k: list(map(float, v)) for k, v in metrics.items()}, f)

    def save_model(self, model, classes):
        if self.config['experiment']['save_model']:
            model_path = os.path.join(self.experiment_dir, 'model.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'classes': classes
            }, model_path)

    def save_confusion_matrix(self, y_true, y_pred, classes):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(self.experiment_dir, 'confusion_matrix.png'))
        plt.close()

    def plot_training_history(self, train_losses, train_accuracies, val_losses, val_accuracies):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, 'training_history.png'))
        plt.close()

    def save_predictions(self, filenames, predictions):
        """
        Save model predictions to a CSV file.
        
        Args:
            filenames (list): List of image filenames
            predictions (list): List of predicted class labels
        """
        predictions_path = os.path.join(self.experiment_dir, 'test_predictions.csv')
        with open(predictions_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Image', 'Predicted Class'])
            for fname, pred in zip(filenames, predictions):
                writer.writerow([fname, pred])

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)