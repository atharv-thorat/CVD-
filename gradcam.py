import torch
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        def forward_hook(module, input, output):
            self.activations = output.detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Backward pass
        target = output[0][class_idx]
        target.backward()

        # Get gradients and activations
        gradients = self.gradients.cpu().numpy()[0]
        activations = self.activations.cpu().numpy()[0]

        # Weight activations by gradients
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Apply ReLU and normalize
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam