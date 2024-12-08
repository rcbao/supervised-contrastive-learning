import os
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt

# Hypothetical imports – replace these with your actual dataset and model code
from main_ce import CustomBrainDataset  # Your dataset class
from networks.resnet_big import SupConResNet         # Your model class
import torch
import torch.nn.functional as F

def train_one_epoch(model, train_loader, criterion, optimizer, epoch, device='cuda'):
    model.train()
    running_loss = 0.0
    total_batches = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_batches += 1

    avg_loss = running_loss / total_batches
    print(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}")
    return avg_loss

@torch.no_grad()
def validate(model, val_loader, criterion, device='cuda'):
    model.eval()
    running_loss = 0.0
    total_batches = 0

    # Example: Compute accuracy as a metric
    correct = 0
    total = 0

    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        total_batches += 1

        # Calculate accuracy
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total_batches
    accuracy = correct / total if total > 0 else 0.0
    print(f"Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy


# Define hyperparameter grids
batch_sizes = [8, 16, 32, 64, 128, 256]
learning_rates = [0.005, 0.01, 0.05, 0.1]
use_transform_options = [True, False]

# Output directory
os.makedirs("results", exist_ok=True)

# Sample transformations (replace with your actual transforms)
base_transform = transforms.Compose([
    # transforms.ToTensor(),
    transforms.Normalize(mean=[0.00143], std=[0.00149]),
])

aug_transform = transforms.Compose([
    transforms.RandomResizedCrop((80, 60)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    # transforms.ToTensor(),
    transforms.Normalize(mean=[0.00143], std=[0.00149]),
])

# Load your dataset (.npy arrays, replace with your paths)
train_images = np.load('Classification_AD_CN_MCI_datasets/brain_train_image_final.npy')
train_labels = np.load('Classification_AD_CN_MCI_datasets/brain_train_label.npy')
test_images = np.load('Classification_AD_CN_MCI_datasets/brain_test_image_final.npy')
test_labels = np.load('Classification_AD_CN_MCI_datasets/brain_test_label.npy')

# Modify images (discard first channel)
train_images_mod = train_images[:, 1, :, :]
test_images_mod = test_images[:, 1, :, :]

def run_experiment(batch_size, lr, use_transform):
    # Set up directory and filenames
    transform_name = "aug" if use_transform else "base"
    run_name = f"batchsize-{batch_size}-lr-{lr}-transform-{transform_name}"
    run_dir = os.path.join("results", run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Choose transforms
    transform = aug_transform if use_transform else base_transform

    # Create datasets and dataloaders
    train_dataset = CustomBrainDataset(train_images_mod, train_labels, transform=transform)
    test_dataset = CustomBrainDataset(test_images_mod, test_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize model
    model = SupConResNet(name='resnet50')
    model = nn.DataParallel(model).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Training parameters
    epochs = 200
    patience = 20
    best_val_loss = float('inf')
    no_improvement = 0
    train_losses = []
    val_losses = []

    for epoch in range(1, epochs+1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss, _ = validate(model, test_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement = 0
            # Save the best model
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
        else:
            no_improvement += 1

        # Early stopping
        if no_improvement >= patience:
            print(f"Early stopping at epoch {epoch} for run {run_name}")
            break

        # Print epoch summary
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save loss graph
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label='train_loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"Loss curves ({run_name})")
    plt.legend()
    plt.savefig(os.path.join(run_dir, f"loss-graph-{run_name}.png"))
    plt.close()

    # Load the best model for evaluation
    model.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pth")))
    model.eval()

    # Evaluate model on test set for confusion matrix & predictions
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.cuda()
            lbls = lbls.cuda()
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(lbls.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f"Confusion Matrix ({run_name})")
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(all_targets)))
    plt.xticks(tick_marks, np.unique(all_targets))
    plt.yticks(tick_marks, np.unique(all_targets))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(run_dir, f"confusion-matrix-{run_name}.png"))
    plt.close()

    # Classification report
    cls_report = classification_report(all_targets, all_preds, target_names=['Healthy', 'Disease', 'MCI'])
    with open(os.path.join(run_dir, f"classification-report-{run_name}.txt"), "w") as f:
        f.write(cls_report)

    # Save a few sample predictions images and their predicted labels
    sample_indices = [0, 1, 2, 3, 4]
    for idx in sample_indices:
        img, lbl = test_dataset[idx]
        img_np = img.squeeze().cpu().numpy()
        # forward pass
        model_input = img.unsqueeze(0).cuda()
        pred_label = torch.argmax(model(model_input)).item()

        plt.figure()
        plt.imshow(img_np, cmap='gray')
        plt.title(f"True: {['Healthy', 'Disease', 'MCI'][lbl]}, Pred: {['Healthy', 'Disease', 'MCI'][pred_label]}")
        plt.colorbar()
        plt.savefig(os.path.join(run_dir, f"sample-img-{idx}-{run_name}.png"))
        plt.close()

    # Save final metrics to a text file
    with open(os.path.join(run_dir, f"metrics-{run_name}.txt"), "w") as f:
        f.write(f"Training Losses: {train_losses}\n")
        f.write(f"Validation Losses: {val_losses}\n")
        f.write("Classification Report:\n")
        f.write(cls_report)

    print(f"Experiment {run_name} complete. Results saved to {run_dir}")

# Exhaustively run all combinations
for bs, lr, tf in itertools.product(batch_sizes, learning_rates, use_transform_options):
    run_experiment(bs, lr, tf)
