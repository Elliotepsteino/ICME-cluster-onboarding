"""
inference_single_image.py

This script performs inference on a single input image 
using a pretrained neural network model. It loads the model, 
preprocesses the image, and predicts the class of the input image.

Usage:
    python inference_single_image.py

Requirements:
    - PyTorch
    - torchvision
    - PIL (Python Imaging Library)
"""
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from mnist_pytorch_example import Net  # Import your model class from the training script

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
model.load_state_dict(torch.load("mnist_cnn.pt")) 
model.eval()  # Set the model in evaluation mode

# Load the image
file_name = "mnist_image_rotated.png" # Replace with mnist_image.png to try the non rotated image.
image = Image.open(file_name).convert("L")
# Define the image transformation (same as in the training script)
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Apply the transformation to the image
image = preprocess(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image)
    probabilities = F.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

print(f"Predicted Class: {predicted_class}")
print(f"Probabilities: {probabilities}")


