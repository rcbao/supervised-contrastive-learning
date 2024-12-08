import os
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F

# Hypothetical imports – replace with your actual dataset and model code
from main_ce import CustomBrainDataset  # Your dataset class
from networks.resnet_big import SupConResNet  # Your model class

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
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        total_batches += 1
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(labels.cpu().numpy())

    avg_loss = running_loss / total_batches
    accuracy = correct / total if total > 0 else 0.0
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    print(f"Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy, all_targets, all_preds

# Adjust hyperparameters as needed
batch_sizes = [16, 32, 64]
learning_rates = [0.01, 0.05]
use_transform_options = [True, False]

os.makedirs("results", exist_ok=True)

# Base transform only (no augmentation) for simplicity
base_transform = transforms.Compose([
    transforms.Normalize(mean=[0.00143], std=[0.00149]),
])

aug_transform = transforms.Compose([
    transforms.RandomResizedCrop((80, 60)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    # transforms.ToTensor(),
    transforms.Normalize(mean=[0.00143], std=[0.00149]),
])
# Load dataset
train_images = np.load('Classification_AD_CN_MCI_datasets/brain_train_image_final.npy')
train_labels = np.load('Classification_AD_CN_MCI_datasets/brain_train_label.npy')
test_images = np.load('Classification_AD_CN_MCI_datasets/brain_test_image_final.npy')
test_labels = np.load('Classification_AD_CN_MCI_datasets/brain_test_label.npy')

# Discard first channel
train_images_mod = train_images[:, 1, :, :]
test_images_mod = test_images[:, 1, :, :]

# Check class distribution
unique_train, counts_train = np.unique(train_labels, return_counts=True)
unique_test, counts_test = np.unique(test_labels, return_counts=True)

print("Class distribution in training set:", dict(zip(unique_train, counts_train)))
print("Class distribution in testing set:", dict(zip(unique_test, counts_test)))

# Optional: If class imbalance is severe, use WeightedRandomSampler
# Compute class weights for WeightedRandomSampler
class_counts = np.bincount(train_labels)
class_weights = 1.0 / class_counts
samples_weights = class_weights[train_labels]
sampler = WeightedRandomSampler(samples_weights, num_samples=len(samples_weights), replacement=True)

def run_experiment(batch_size, lr, use_transform): 
    transform_name = "aug" if use_transform else "base"
    run_name = f"batchsize-{batch_size}-lr-{lr}-transform-{transform_name}"
    run_dir = os.path.join("results", run_name)
    os.makedirs(run_dir, exist_ok=True)

    transform = aug_transform if use_transform else base_transform

    train_dataset = CustomBrainDataset(train_images_mod, train_labels, transform=transform)
    test_dataset = CustomBrainDataset(test_images_mod, test_labels, transform=transform)

    # Use WeightedRandomSampler to address class imbalance
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = SupConResNet(name='resnet50')
    model = nn.DataParallel(model).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Reduced epochs and patience for quick tests
    epochs = 150
    patience = 15
    best_val_loss = float('inf')
    no_improvement = 0
    train_losses = []
    val_losses = []

    for epoch in range(1, epochs+1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_acc, val_targets, val_preds = validate(model, test_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement = 0
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
        else:
            no_improvement += 1

        if no_improvement >= patience:
            print(f"Early stopping at epoch {epoch} for run {run_name}")
            break

        print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}")

    # Save loss curves
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label='train_loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f"Loss curves ({run_name})")
    plt.legend()
    plt.savefig(os.path.join(run_dir, f"loss-graph-{run_name}.png"), dpi=300)
    plt.close()

    # Load best model and evaluate again for confusion matrix
    model.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pth"), weights_only=True))
    model.eval()

    # Compute confusion matrix and classification report
    cm = confusion_matrix(val_targets, val_preds)
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f"Confusion Matrix ({run_name})")
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(val_targets)))
    plt.xticks(tick_marks, np.unique(val_targets))
    plt.yticks(tick_marks, np.unique(val_targets))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(run_dir, f"confusion-matrix-{run_name}.png"), dpi=300)
    plt.close()

    cls_report = classification_report(
        val_targets,
        val_preds,
        target_names=['Healthy', 'Disease', 'MCI'],
        zero_division=0  # Sets precision to 0.0 for undefined cases
    )

    # Save metrics and class distribution info
    with open(os.path.join(run_dir, f"metrics-{run_name}.txt"), "w") as f:
        f.write("Class distribution in training set:\n")
        f.write(str(dict(zip(unique_train, counts_train))) + "\n")
        f.write("Class distribution in testing set:\n")
        f.write(str(dict(zip(unique_test, counts_test))) + "\n\n")
        f.write(f"Training Losses: {train_losses}\n")
        f.write(f"Validation Losses: {val_losses}\n")
        f.write("Classification Report:\n")
        f.write(cls_report)

    print(f"Experiment {run_name} complete. Results saved to {run_dir}")

# Run experiments with adjusted parameters and simpler transform
for bs, lr, tf in itertools.product(batch_sizes, learning_rates, use_transform_options):
    run_experiment(bs, lr, tf)
