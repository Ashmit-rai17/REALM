import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# Import trained model and metadata
from model import model, class_names, suitability_map, device


# ---------------- GradCAM Class ----------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # use full backward hook (fixes warning)
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        grad_cam = (weights * self.activations).sum(dim=1, keepdim=True)
        grad_cam = F.relu(grad_cam)

        grad_cam = F.interpolate(grad_cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min())
        return grad_cam[0, 0].cpu().numpy()


# ---------------- Preprocessing ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ---------------- Grad-CAM Application ----------------
def apply_gradcam(image_path, save_path="gradcam_result.jpg"):
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 1)
    predicted_class = class_names[predicted.item()]
    suitability = suitability_map[predicted_class]

    # Grad-CAM on last ResNet block
    target_layer = model.layer4[-1].conv3 if hasattr(model.layer4[-1], "conv3") else model.layer4[-1].conv2
    gradcam = GradCAM(model, target_layer)
    mask = gradcam.generate(input_tensor, target_class=predicted.item())

    # Create Heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    img_np = np.array(img.resize((224, 224))) / 255.0
    overlay = heatmap * 0.5 + img_np * 0.5

    result = np.uint8(255 * overlay)
    cv2.imwrite(save_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

    # Plot results
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Heatmap")
    plt.imshow(heatmap)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title(f"Overlay\n{predicted_class} ({suitability})")
    plt.imshow(overlay)
    plt.axis("off")

    plt.show()

    print(f"\n✅ Prediction: {predicted_class}")
    print(f"🌱 Suitability: {suitability}")
    print(f"💾 Grad-CAM saved at: {os.path.abspath(save_path)}\n")

    return predicted_class, suitability


# ---------------- Example Run ----------------
if __name__ == "__main__":
    dataset_path = r"C:\Users\vinay\OneDrive\Desktop\REALM\EuroSAT"

    # Choose random class and random image
    class_folder = random.choice(class_names)
    class_path = os.path.join(dataset_path, class_folder)
    image_file = random.choice(os.listdir(class_path))
    image_path = os.path.join(class_path, image_file)

    print(f"\n🔍 Running Grad-CAM on: {image_path}")
    apply_gradcam(image_path, save_path="gradcam_output.jpg")
