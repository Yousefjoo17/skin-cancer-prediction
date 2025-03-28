import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from dsets import SkinCancerDataset
from torch.utils.data import DataLoader
from model import SkinCancerModel
import torch.nn as nn
import sys

class ROCEvaluator:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--model-path',
            help='Path to trained model .state file',
            required=True,
        )
        self.cli_args = parser.parse_args(sys_argv)
      
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(self.cli_args.model_path)
        self.model.eval()

    def load_model(self, model_path):
        """Load a trained model from disk."""
        model = SkinCancerModel()
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.load_state_dict(torch.load(model_path)['model_state'])
        model = model.to(self.device)
        return model

    def get_predictions(self, dataset):
        """Run model on dataset and return labels/probabilities."""
        dataloader = DataLoader(dataset, batch_size=32, num_workers=4)
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in dataloader:
                inputs, labels, _ = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                logits, probs = self.model(inputs)
                all_labels.append(labels[:, 1].cpu().numpy())  # Positive class
                all_probs.append(probs[:, 1].cpu().numpy())    # Positive class

        return np.concatenate(all_labels), np.concatenate(all_probs)

    def plot_roc_curve(self, labels, probs):
        """Generate ROC curve, highlight the best threshold, and return metrics."""
        fpr, tpr, thresholds = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)

        # Find the best threshold using Youden's J statistic
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = thresholds[optimal_idx]
        best_fpr, best_tpr = fpr[optimal_idx], tpr[optimal_idx]

        # Plot ROC curve
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

        # Highlight the best threshold
        plt.scatter(best_fpr, best_tpr, color='red', marker='o', label=f'Best Threshold = {optimal_threshold:.3f}')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig('roc_curve.png')
        plt.close()

        return fpr, tpr, thresholds, roc_auc, optimal_threshold

    def main(self):
        """Main evaluation pipeline."""
        val_dataset = SkinCancerDataset(is_val=True, val_stride=10)
        labels, probs = self.get_predictions(val_dataset)

        # Generate ROC curve
        fpr, tpr, thresholds, roc_auc, optimal_thresh = self.plot_roc_curve(labels, probs)
        print(f"AUC: {roc_auc:.3f}")
        print(f"Optimal Threshold (Youden's J): {optimal_thresh:.3f}")

if __name__ == '__main__':
    ROCEvaluator().main()
