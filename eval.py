"""
WirelessJEPA Downstream Task Evaluation Module
Implements linear probing and k-NN evaluation on downstream tasks
Based on: "WirelessJEPA: A Multi-Antenna Foundation Model using Spatio-temporal Wireless Latent Predictions"
Reference: Paper Section 4, Table I, II - Downstream task evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


class LinearProber(nn.Module):
    """Linear head for downstream task evaluation"""
    
    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class DownstreamEvaluator:
    """
    Downstream task evaluator with linear probing and k-NN methods.
    
    Paper Reference: Section 4, Table I, II
    Evaluates on RML2016.10a modulation classification with few-shot scenarios
    """
    
    def __init__(self, pretrained_model, config, device: str = 'cuda'):
        """
        Initialize evaluator.
        
        Args:
            pretrained_model: Pretrained WirelessJEPA model
            config: Configuration object
            device: Device to run on
        """
        self.pretrained_model = pretrained_model
        self.config = config
        self.device = device
        
        # Freeze pretrained encoder
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        
        self.results = {}
    
    @torch.no_grad()
    def extract_features(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features using pretrained encoder.
        
        Args:
            data_loader: DataLoader for data
        
        Returns:
            Features and labels arrays
        """
        self.pretrained_model.eval()
        
        all_features = []
        all_labels = []
        
        pbar = tqdm(data_loader, desc="Extracting features")
        
        for signals, labels in pbar:
            signals = signals.to(self.device)
            
            # Forward through the frozen student encoder and pool the spatial
            # latent map for downstream classifiers.
            if hasattr(self.pretrained_model, "extract_features"):
                features = self.pretrained_model.extract_features(signals)
            else:
                features, _ = self.pretrained_model.context_encoder(signals)
            
            # Convert to numpy
            features_np = features.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            all_features.append(features_np)
            all_labels.append(labels_np)
        
        features = np.concatenate(all_features, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        return features, labels
    
    def linear_probing(self, 
                      train_loader: DataLoader,
                      test_loader: DataLoader,
                      num_classes: int,
                      epochs: int = 100) -> Dict[str, float]:
        """
        Linear probing evaluation.
        
        Paper: Train linear classifier on frozen features
        
        Args:
            train_loader: Training DataLoader
            test_loader: Test DataLoader
            num_classes: Number of classes
            epochs: Training epochs for linear head
        
        Returns:
            Evaluation metrics
        """
        print("\n=== Linear Probing Evaluation ===")
        
        # Extract train features
        train_features, train_labels = self.extract_features(train_loader)
        test_features, test_labels = self.extract_features(test_loader)
        
        feature_dim = train_features.shape[1]
        
        # Create linear head
        head = LinearProber(feature_dim, num_classes).to(self.device)
        optimizer = Adam(head.parameters(), lr=self.config.evaluation.linear_probe_lr)
        criterion = nn.CrossEntropyLoss()
        
        # Convert to tensors
        train_features_tensor = torch.from_numpy(train_features).float().to(self.device)
        train_labels_tensor = torch.from_numpy(train_labels).long().to(self.device)
        test_features_tensor = torch.from_numpy(test_features).float().to(self.device)
        test_labels_tensor = torch.from_numpy(test_labels).long().to(self.device)
        
        # Training loop
        best_acc = 0.0
        for epoch in range(epochs):
            head.train()
            
            # Forward
            logits = head(train_features_tensor)
            loss = criterion(logits, train_labels_tensor)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Validation
            if (epoch + 1) % 10 == 0:
                head.eval()
                with torch.no_grad():
                    val_logits = head(test_features_tensor)
                    val_preds = val_logits.argmax(dim=1)
                    val_acc = (val_preds == test_labels_tensor).float().mean().item()
                    
                    if val_acc > best_acc:
                        best_acc = val_acc
                
                if (epoch + 1) % 50 == 0:
                    print(f"Epoch {epoch + 1}: Loss={loss.item():.4f}, Val Acc={val_acc:.4f}")
        
        # Final evaluation
        head.eval()
        with torch.no_grad():
            test_logits = head(test_features_tensor)
            test_preds = test_logits.argmax(dim=1).cpu().numpy()
        
        # Metrics
        accuracy = accuracy_score(test_labels, test_preds)
        f1 = f1_score(test_labels, test_preds, average='macro', zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'best_val_acc': best_acc
        }
        
        print(f"\nLinear Probing Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        return metrics, test_preds, test_labels
    
    def knn_evaluation(self,
                      train_loader: DataLoader,
                      test_loader: DataLoader,
                      k: int = 20) -> Dict[str, float]:
        """
        k-NN evaluation on features.
        
        Paper: k-NN classification on frozen features
        
        Args:
            train_loader: Training DataLoader
            test_loader: Test DataLoader
            k: Number of neighbors
        
        Returns:
            Evaluation metrics
        """
        print("\n=== k-NN Evaluation ===")
        
        # Extract features
        train_features, train_labels = self.extract_features(train_loader)
        test_features, test_labels = self.extract_features(test_loader)
        
        # Compute distances
        from sklearn.neighbors import KNeighborsClassifier
        
        knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
        knn.fit(train_features, train_labels)
        
        test_preds = knn.predict(test_features)
        
        # Metrics
        accuracy = accuracy_score(test_labels, test_preds)
        f1 = f1_score(test_labels, test_preds, average='macro', zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1
        }
        
        print(f"\nk-NN (k={k}) Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        return metrics, test_preds, test_labels
    
    def evaluate_few_shot(self,
                         train_loader: DataLoader,
                         test_loader: DataLoader,
                         num_classes: int,
                         few_shot_samples: int) -> Dict[str, float]:
        """
        Few-shot evaluation (1-shot, 100-shot, 500-shot).
        
        Paper Table I: Few-shot scenario evaluation
        """
        print(f"\n=== {few_shot_samples}-shot Evaluation ===")
        
        # This uses the same linear probing but with limited training data
        return self.linear_probing(train_loader, test_loader, num_classes, epochs=100)
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            modulation_list: List[str],
                            save_path: Optional[str] = None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=modulation_list,
                   yticklabels=modulation_list)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix - Modulation Classification')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Confusion matrix saved: {save_path}")
        
        plt.close()
    
    def plot_results_table(self, results: Dict, save_path: Optional[str] = None):
        """Generate results table visualization"""
        import pandas as pd
        
        df = pd.DataFrame(results).T
        
        plt.figure(figsize=(10, 6))
        plt.axis('tight')
        plt.axis('off')
        
        table = plt.table(cellText=df.values,
                        rowLabels=df.index,
                        colLabels=df.columns,
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        plt.title('WirelessJEPA Evaluation Results')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Results table saved: {save_path}")
        
        plt.close()
    
    def evaluate(self, train_loader: DataLoader, test_loader: DataLoader,
                num_classes: int, modulation_list: List[str]) -> Dict:
        """
        Complete evaluation pipeline.
        
        Paper Section 4: Full evaluation protocol
        """
        results = {}
        
        # Linear probing
        lp_metrics, lp_preds, lp_labels = self.linear_probing(
            train_loader, test_loader, num_classes
        )
        results['linear_probing'] = lp_metrics
        
        # k-NN evaluation
        knn_metrics, knn_preds, knn_labels = self.knn_evaluation(
            train_loader, test_loader, k=self.config.evaluation.knn_k
        )
        results['knn'] = knn_metrics
        
        # Save visualizations
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Confusion matrices
        self.plot_confusion_matrix(lp_labels, lp_preds, modulation_list,
                                  str(output_dir / 'confusion_matrix_lp.png'))
        self.plot_confusion_matrix(knn_labels, knn_preds, modulation_list,
                                  str(output_dir / 'confusion_matrix_knn.png'))
        
        # Results table
        self.plot_results_table(results, str(output_dir / 'results_table.png'))
        
        print(f"\n{'='*50}")
        print("FINAL EVALUATION RESULTS")
        print(f"{'='*50}")
        for method, metrics in results.items():
            print(f"\n{method}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        self.results = results
        return results


def evaluate_pretrained_model(model, config, train_loader, test_loader,
                             num_classes: int, modulation_list: List[str]):
    """
    Convenience function to evaluate a pretrained model.
    
    Args:
        model: Pretrained WirelessJEPA model
        config: Configuration object
        train_loader: Training DataLoader
        test_loader: Test DataLoader
        num_classes: Number of classes
        modulation_list: List of modulation names
    
    Returns:
        Evaluation results dictionary
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluator = DownstreamEvaluator(model, config, device=device)
    
    results = evaluator.evaluate(train_loader, test_loader, num_classes, modulation_list)
    
    return results


if __name__ == "__main__":
    print("Evaluation module initialized")
    print("Use this with pretrained models for downstream task evaluation")
