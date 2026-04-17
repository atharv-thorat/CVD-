import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

# Import your model
from model.model_architecture import ImprovedAttentionCNN


# -------------------------------
# CONFIGURATION
# -------------------------------
MODEL_PATH = "model/final_improved_model.pth"
TEST_DIR = "../CVI-img-datasets/imagedata"
BATCH_SIZE = 32
NUM_CLASSES = 5

# Clinical label mapping (folder 1→C0 etc.)
CLINICAL_LABELS = {
    "1": "C0",
    "2": "C1",
    "3": "C2_C3",
    "4": "C4",
    "5": "C5_C6"
}


# -------------------------------
# DEVICE
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")


# -------------------------------
# TRANSFORMS
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -------------------------------
# LOAD TEST DATA
# -------------------------------
test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"\nTotal Test Images: {len(test_dataset)}")
print("Folder to Index Mapping:", test_dataset.class_to_idx)

# Convert folder names to clinical names
sorted_folders = sorted(test_dataset.class_to_idx, key=test_dataset.class_to_idx.get)
CLASS_NAMES = [CLINICAL_LABELS[folder] for folder in sorted_folders]

print("Clinical Class Names:", CLASS_NAMES)


# -------------------------------
# LOAD MODEL
# -------------------------------
model = ImprovedAttentionCNN(num_classes=NUM_CLASSES, pretrained=False)

# Load checkpoint correctly
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

# If saved as dict
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

print("\nModel Loaded Successfully!")


# -------------------------------
# MODEL PARAMETERS
# -------------------------------
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\n================ MODEL PARAMETERS ================")
print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")


# -------------------------------
# EVALUATION
# -------------------------------
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(probs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Accuracy
accuracy = accuracy_score(all_labels, all_preds) * 100


# -------------------------------
# RESULTS
# -------------------------------
print("\n================ MODEL RESULTS ================")
print(f"\nTest Accuracy: {accuracy:.2f}%")

print("\nClassification Report:")
print(classification_report(
    all_labels,
    all_preds,
    target_names=CLASS_NAMES,
    digits=4
))

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Confusion Matrix Plot
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)

# 2. Per-class F1 bar chart
from sklearn.metrics import f1_score
f1s = f1_score(all_labels, all_preds, average=None)
plt.figure(figsize=(8, 5))
plt.bar(CLASS_NAMES, f1s, color='steelblue')
plt.title('Per-Class F1 Score')
plt.ylabel('F1 Score')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('f1_scores.png', dpi=150)

# 3. Confidence distribution
plt.figure(figsize=(8, 5))
plt.hist(all_confidences, bins=20, color='steelblue', edgecolor='black')
plt.title('Prediction Confidence Distribution')
plt.xlabel('Confidence')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('confidence_dist.png', dpi=150)

# -------------------------------
# SAMPLE PREDICTIONS
# -------------------------------
print("\n================ SAMPLE PREDICTIONS ================")

images, labels = next(iter(test_loader))
images = images.to(device)

with torch.no_grad():
    outputs = model(images)
    probs = torch.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probs, 1)

for i in range(min(5, len(images))):
    print(f"\nSample {i+1}")
    print(f"True Label : {CLASS_NAMES[labels[i].item()]}")
    print(f"Predicted  : {CLASS_NAMES[predicted[i].item()]}")
    print(f"Confidence : {confidence[i].item():.4f}")


print("\n================ EVALUATION COMPLETE ================\n")