import argparse
import datetime
import hashlib
import shutil
import dsets  
import model  
import importlib
importlib.reload(dsets)
importlib.reload(model)
import numpy as np
from util.logconf import logging
import os
import sys

import torch
import torch.nn as nn

from dsets import SkinCancerDataset
from model import SkinCancerModel
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter # Run using -> tensorboard --logdir=runs

from util.util import enumerateWithEstimate

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class SkinCancerTestingApp():
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
            help='Batch size to use for testing',
            default=32,
            type=int,
        )
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=8,
            type=int,
        )    
        parser.add_argument('--model-path',
            help='Path to the trained model for evaluation',
            default="",
        )
        parser.add_argument('--threshold',
            help='Threshold for classification (default: 0.5)',
            default=0.5,
            type=float,
        )

        self.cli_args = parser.parse_args(sys_argv)
        self.threshold = self.cli_args.threshold  # Store threshold argument

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()

        self.model, self.optimizer = self.load_model(self.model, self.optimizer, self.cli_args.model_path)

    def initOptimizer(self):
        return Adam(self.model.parameters())
    
    def initModel(self):
        model = SkinCancerModel()
        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model
    
    def load_model(self, model, optimizer, model_path):
        checkpoint = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        print(f"Loaded best model from {model_path}")
        
        return model, optimizer

    def initTestDl(self):
        test_ds = SkinCancerDataset(
            is_test=True
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        test_dl = DataLoader(
            test_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return test_dl
    
    def main(self):
        log.info("Starting testing {}, {}".format(type(self).__name__, self.cli_args))

        test_dl = self.initTestDl()
        self.doTesting(test_dl)

    def doTesting(self, test_dl):
        with torch.no_grad():
            self.model.eval()

            all_labels = []
            all_predictions = []

            batch_iter = enumerateWithEstimate(
                test_dl,
                "testing ",
                start_ndx=test_dl.num_workers,
            )
            for _, batch_tup in batch_iter:
                input_t, label_t, _ = batch_tup

                input_g = input_t.to(self.device, non_blocking=True)
                label_g = label_t.to(self.device, non_blocking=True)

                _, probability_g = self.model(input_g)

                # Apply threshold for classification
                predictions_g = (probability_g[:, 1] >= self.threshold).long()

                all_labels.extend(label_g[:, 1].cpu().numpy())  # Positive class labels
                all_predictions.extend(predictions_g.cpu().numpy())

            self.compute_log_metrics(all_labels, all_predictions)

    def compute_log_metrics(self, all_labels, all_predictions):
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)

        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)

        positive_accuracy = accuracy_score(all_labels[all_labels == 1], all_predictions[all_labels == 1])
        negative_accuracy = accuracy_score(all_labels[all_labels == 0], all_predictions[all_labels == 0])

        log.info(f"Test Metrics:")
        log.info(f"Accuracy: {accuracy:.4f}")
        log.info(f"Positive Accuracy: {positive_accuracy:.4f}")
        log.info(f"Negative Accuracy: {negative_accuracy:.4f}")
        log.info(f"Precision: {precision:.4f}")
        log.info(f"Recall: {recall:.4f}")
        log.info(f"F1 Score: {f1:.4f}")

if __name__ == '__main__':
    SkinCancerTestingApp().main()
