import torch
import torchvision.transforms as transforms

# CEAP class names
class_names = ["C0", "C1", "C2_C3", "C4", "C5_C6"]

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_image(model, image):
    """
    Returns:
    ceap        -> CEAP class label
    severity    -> Gentle / Moderate / Severe
    confidence  -> model confidence
    class_idx   -> predicted class index
    input_tensor-> tensor required for GradCAM
    """

    device = next(model.parameters()).device

    # Apply transforms
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    class_idx = predicted.item()
    ceap = class_names[class_idx]

    # Severity mapping
    if ceap in ["C0", "C1"]:
        severity = "Gentle"
    elif ceap == "C2_C3":
        severity = "Moderate"
    else:
        severity = "Severe"

    return ceap, severity, confidence.item(), class_idx, input_tensor
