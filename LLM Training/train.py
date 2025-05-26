from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
import pandas as pd
import json
import numpy as np
import wandb
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc, precision_recall_curve, average_precision_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import io
from PIL import Image

# Set environment variable to disable dispatch_batches
os.environ["ACCELERATE_USE_DISPATCH_BATCHES"] = "false"

# Initialize wandb
wandb.login(key="YOUR_API")

# Load dataset
with open("dataset/synthetic_ti_rads_dataset.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)
label_map = {"TR1": 0, "TR2": 1, "TR3": 2, "TR4": 3, "TR5": 4}
df["label"] = df["ti_rads_score"].map(label_map)

# Print dataset distribution
print("Dataset distribution:")
distribution = df["ti_rads_score"].value_counts().sort_index()
for score, count in distribution.items():
    print(f"{score}: {count} samples")

# Split dataset
train_df, test_df = train_test_split(df, test_size=0.1, stratify=df["label"], random_state=42)
train_dataset = Dataset.from_pandas(train_df[["report_summary", "label"]])
test_dataset = Dataset.from_pandas(test_df[["report_summary", "label"]])

# Tokenizer and Model
checkpoint = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=5)

# Set max sequence length
max_length = 512

# Process datasets
def process_dataset(dataset):
    processed_data = []
    for example in dataset:
        encoded = tokenizer(
            example["report_summary"], 
            padding="max_length", 
            truncation=True, 
            max_length=max_length, 
            return_tensors="pt"
        )
        processed_data.append({
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "label": torch.tensor(example["label"]),
            "text": example["report_summary"]  # Store original text for later visualizations
        })
    return processed_data

train_data = process_dataset(train_dataset)
test_data = process_dataset(test_dataset)

# Create DataLoaders
batch_size = 16

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)

# Training configurations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Hyperparameters
epochs = 10
learning_rate = 2e-5
weight_decay = 0.01

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Learning rate scheduler
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),  # 10% warmup
    num_training_steps=total_steps
)

# Create class mappings
class_names = ["TR1", "TR2", "TR3", "TR4", "TR5"]
id2label = {idx: label for idx, label in enumerate(class_names)}
label2id = {label: idx for idx, label in enumerate(class_names)}

# Helper functions for visualizations
def plot_to_wandb_image(plt_figure):
    """Convert matplotlib figure to wandb Image"""
    buf = io.BytesIO()
    plt_figure.savefig(buf, format='png')
    buf.seek(0)
    # Convert BytesIO to PIL Image which wandb can handle
    image = Image.open(buf)
    return wandb.Image(image)

def create_class_metrics_chart(class_metrics, epoch):
    """Create a chart showing per-class metrics"""
    plt.figure(figsize=(12, 6))
    
    metrics = ['precision', 'recall', 'f1']
    x = np.arange(len(class_names))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [class_metrics[cls][metric] for cls in class_names]
        plt.bar(x + i*width, values, width, label=metric)
    
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title(f'Per-Class Metrics (Epoch {epoch+1})')
    plt.xticks(x + width, class_names)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.tight_layout()
    
    return plot_to_wandb_image(plt.figure())

def create_roc_curves(labels, preds_proba):
    """Create ROC curves for each class (one-vs-rest)"""
    plt.figure(figsize=(10, 8))
    
    # One-hot encode the labels for ROC calculation
    y_true = np.eye(len(class_names))[labels]
    
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true[:, i], preds_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-Rest)')
    plt.legend(loc="lower right")
    
    return plot_to_wandb_image(plt.figure())

def create_pr_curves(labels, preds_proba):
    """Create Precision-Recall curves for each class"""
    plt.figure(figsize=(10, 8))
    
    # One-hot encode the labels for PR calculation
    y_true = np.eye(len(class_names))[labels]
    
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_true[:, i], preds_proba[:, i])
        avg_precision = average_precision_score(y_true[:, i], preds_proba[:, i])
        plt.plot(recall, precision, lw=2, label=f'{class_name} (AP = {avg_precision:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="best")
    plt.grid(True)
    
    return plot_to_wandb_image(plt.figure())

def visualize_embeddings(model, data_loader, num_samples=100):
    """Extract embeddings and visualize with t-SNE"""
    model.eval()
    embeddings = []
    labels = []
    texts = []
    
    with torch.no_grad():
        for batch in data_loader:
            if len(embeddings) >= num_samples:
                break
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["label"].cpu().numpy()
            
            # Get the hidden states from the model
            outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)
            # Use CLS token embedding as sentence embedding
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            embeddings.extend(batch_embeddings)
            labels.extend(batch_labels)
            texts.extend(batch.get("text", [""] * len(batch_labels)))
            
            if len(embeddings) >= num_samples:
                embeddings = embeddings[:num_samples]
                labels = labels[:num_samples]
                texts = texts[:num_samples]
                break
    
    # Apply t-SNE to reduce dimensionality
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(np.array(embeddings))
    
    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        mask = np.array(labels) == i
        if np.any(mask):
            plt.scatter(
                embeddings_2d[mask, 0], 
                embeddings_2d[mask, 1], 
                label=class_name,
                alpha=0.7
            )
    
    plt.legend()
    plt.title('t-SNE Visualization of Report Embeddings')
    plt.tight_layout()
    
    return plot_to_wandb_image(plt.figure())

def create_confusion_matrix_chart(conf_matrix, title):
    """Create a more visually appealing confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix, 
        annot=True, 
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    
    return plot_to_wandb_image(plt.figure())

def calculate_per_class_metrics(y_true, y_pred):
    """Calculate precision, recall, and F1 for each class"""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(class_names))
    )
    
    class_metrics = {}
    for i, class_name in enumerate(class_names):
        class_metrics[class_name] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i]
        }
    
    return class_metrics

# Initialize wandb with more config details
run = wandb.init(
    project="tirads-classification",
    name="biomedbert-tirads-enhanced",
    config={
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "model_name": checkpoint,
        "max_length": max_length,
        "optimizer": "AdamW",
        "weight_decay": weight_decay,
        "scheduler": "linear_with_warmup",
        "warmup_steps": int(0.1 * total_steps),
        "num_classes": 5,
        "dataset_size": len(train_dataset),
        "class_mapping": label_map
    }
)

# Create custom WandB tables for tracking predictions
wandb.define_metric("global_step")
wandb.define_metric("train/*", step_metric="global_step")
wandb.define_metric("eval/*", step_metric="global_step")

# Track model parameter gradients and weights
wandb.watch(model, log="all", log_freq=10)

# Create a table to track example predictions
predictions_table = wandb.Table(columns=["epoch", "text", "true_label", "predicted_label", "confidence"])

# Training loop
global_step = 0
best_f1 = 0.0
best_model_state = None

# Log initial dataset distribution
train_label_counts = train_df["ti_rads_score"].value_counts().sort_index()
test_label_counts = test_df["ti_rads_score"].value_counts().sort_index()

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].bar(train_label_counts.index, train_label_counts.values)
ax[0].set_title("Training Set Distribution")
ax[0].set_xlabel("TI-RADS Score")
ax[0].set_ylabel("Count")

ax[1].bar(test_label_counts.index, test_label_counts.values)
ax[1].set_title("Test Set Distribution")
ax[1].set_xlabel("TI-RADS Score")
ax[1].set_ylabel("Count")
plt.tight_layout()

wandb.log({"dataset_distribution": plot_to_wandb_image(fig)})

for epoch in range(epochs):
    # Training
    model.train()
    train_loss = 0
    train_preds, train_labels = [], []
    train_proba = []
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
    for batch_idx, batch in enumerate(progress_bar):
        # Get batch data
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        
        # Calculate probabilities
        proba = torch.nn.functional.softmax(logits, dim=1)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Track gradient norms
        if batch_idx % 10 == 0:
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            wandb.log({"global_step": global_step, "train/gradient_norm": total_norm})
        
        optimizer.step()
        scheduler.step()
        
        # Update metrics
        train_loss += loss.item()
        batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
        batch_labels = labels.cpu().numpy()
        batch_proba = proba.detach().cpu().numpy()
        
        train_preds.extend(batch_preds)
        train_labels.extend(batch_labels)
        train_proba.extend(batch_proba)
        
        # Update progress bar
        progress_bar.set_postfix({"loss": loss.item()})
        
        # Log training metrics every 10 steps
        if batch_idx % 10 == 0:
            wandb.log({
                "global_step": global_step,
                "train/batch_loss": loss.item(),
                "train/learning_rate": scheduler.get_last_lr()[0]
            })
            
            # Log parameter histograms periodically
            if batch_idx % 50 == 0:
                for name, param in model.named_parameters():
                    if "layer" in name and ".0." in name:  # Log a subset of parameters
                        wandb.log({f"parameters/{name}": wandb.Histogram(param.detach().cpu().numpy())}, step=global_step)
        
        global_step += 1
    
    # Calculate training metrics
    train_loss = train_loss / len(train_loader)
    train_acc = accuracy_score(train_labels, train_preds)
    train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
        train_labels, train_preds, average='weighted'
    )
    
    # Calculate per-class metrics for training
    train_class_metrics = calculate_per_class_metrics(train_labels, train_preds)
    
    # Create training confusion matrix
    train_conf_matrix = wandb.plot.confusion_matrix(
        y_true=train_labels,
        preds=train_preds,
        class_names=class_names
    )
    
    # Create enhanced confusion matrix
    train_cm = np.zeros((len(class_names), len(class_names)), dtype=int)
    for true_label, pred_label in zip(train_labels, train_preds):
        train_cm[true_label][pred_label] += 1
    
    # Evaluation
    model.eval()
    eval_loss = 0
    eval_preds, eval_labels = [], []
    eval_proba = []
    eval_texts = []
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Eval]")
        for batch in progress_bar:
            # Get batch data
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            # Store texts for example tracking
            if "text" in batch:
                texts = batch["text"]
                eval_texts.extend(texts)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            # Calculate probabilities
            proba = torch.nn.functional.softmax(logits, dim=1)
            
            # Update metrics
            eval_loss += loss.item()
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            batch_labels = labels.cpu().numpy()
            batch_proba = proba.cpu().numpy()
            
            eval_preds.extend(batch_preds)
            eval_labels.extend(batch_labels)
            eval_proba.extend(batch_proba)
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})
    
    # Track some example predictions
    for i in range(min(10, len(eval_texts))):
        predictions_table.add_data(
            epoch + 1,
            eval_texts[i] if i < len(eval_texts) else "",
            class_names[eval_labels[i]],
            class_names[eval_preds[i]],
            float(eval_proba[i][eval_preds[i]])
        )
    
    # Calculate evaluation metrics
    eval_loss = eval_loss / len(test_loader)
    eval_acc = accuracy_score(eval_labels, eval_preds)
    eval_precision, eval_recall, eval_f1, _ = precision_recall_fscore_support(
        eval_labels, eval_preds, average='weighted'
    )
    
    # Calculate per-class metrics for evaluation
    eval_class_metrics = calculate_per_class_metrics(eval_labels, eval_preds)
    
    # Create evaluation confusion matrix
    eval_conf_matrix = wandb.plot.confusion_matrix(
        y_true=eval_labels,
        preds=eval_preds,
        class_names=class_names
    )
    
    # Create enhanced confusion matrix
    eval_cm = np.zeros((len(class_names), len(class_names)), dtype=int)
    for true_label, pred_label in zip(eval_labels, eval_preds):
        eval_cm[true_label][pred_label] += 1
    
    # Convert proba lists to numpy arrays for visualization
    train_proba_np = np.array(train_proba)
    eval_proba_np = np.array(eval_proba)
    
    # Create ROC and PR curves
    eval_roc_curves = create_roc_curves(eval_labels, eval_proba_np)
    eval_pr_curves = create_pr_curves(eval_labels, eval_proba_np)
    
    # Create class metrics charts
    train_class_chart = create_class_metrics_chart(train_class_metrics, epoch)
    eval_class_chart = create_class_metrics_chart(eval_class_metrics, epoch)
    
    # Create enhanced confusion matrices
    train_cm_chart = create_confusion_matrix_chart(train_cm, f"Training Confusion Matrix (Epoch {epoch+1})")
    eval_cm_chart = create_confusion_matrix_chart(eval_cm, f"Evaluation Confusion Matrix (Epoch {epoch+1})")
    
    # Create embedding visualization (every 2 epochs to save time)
    if epoch % 2 == 0 or epoch == epochs - 1:
        embedding_viz = visualize_embeddings(model, test_loader)
    
    # Create prediction distribution chart
    plt.figure(figsize=(10, 6))
    plt.hist([np.max(p) for p in eval_proba_np], bins=20, alpha=0.7)
    plt.title(f"Prediction Confidence Distribution (Epoch {epoch+1})")
    plt.xlabel("Confidence Score")
    plt.ylabel("Count")
    confidence_chart = plot_to_wandb_image(plt.figure())
    
    # Log all metrics to wandb
    wandb.log({
        "global_step": global_step,
        "epoch": epoch + 1,
        "train/loss": train_loss,
        "train/accuracy": train_acc,
        "train/f1": train_f1,
        "train/precision": train_precision,
        "train/recall": train_recall,
        "eval/loss": eval_loss,
        "eval/accuracy": eval_acc,
        "eval/f1": eval_f1,
        "eval/precision": eval_precision,
        "eval/recall": eval_recall,
        "visualizations/train_confusion_matrix": train_conf_matrix,
        "visualizations/eval_confusion_matrix": eval_conf_matrix, 
        "visualizations/enhanced_train_cm": train_cm_chart,
        "visualizations/enhanced_eval_cm": eval_cm_chart,
        "visualizations/roc_curves": eval_roc_curves,
        "visualizations/pr_curves": eval_pr_curves,
        "visualizations/train_class_metrics": train_class_chart,
        "visualizations/eval_class_metrics": eval_class_chart,
        "visualizations/confidence_distribution": confidence_chart,
    })
    
    # Log embedding visualization if available
    if epoch % 2 == 0 or epoch == epochs - 1:
        wandb.log({
            "global_step": global_step,
            "visualizations/embeddings": embedding_viz
        })
    
    # Save best model based on F1 score
    if eval_f1 > best_f1:
        best_f1 = eval_f1
        best_model_state = model.state_dict().copy()
        # Save the best model locally only - avoid symlink issues on Windows
        torch.save(best_model_state, "best_model.pt")
        # Log the F1 score as a summary metric
        wandb.run.summary["best_f1"] = best_f1
        wandb.run.summary["best_epoch"] = epoch + 1
    
    # Print metrics
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")
    print(f"Eval Loss: {eval_loss:.4f}, Accuracy: {eval_acc:.4f}, F1: {eval_f1:.4f}")

# Upload predictions table
wandb.log({"predictions_table": predictions_table})

# Load the best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)

# Save model
model.save_pretrained("./tirads_model")
tokenizer.save_pretrained("./tirads_model")

# Final evaluation metrics
print("\nFinal Evaluation Results:")
print(f"Accuracy: {eval_acc:.4f}")
print(f"F1 Score: {eval_f1:.4f}")
print(f"Precision: {eval_precision:.4f}")
print(f"Recall: {eval_recall:.4f}")

# End wandb run
wandb.finish()
