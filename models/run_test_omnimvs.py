import torch
import cv2
import numpy as np
from omnimvs import OmniMVS  # Import the OmniMVS model
from spherical_sweep import SphericalSweeping  # Import the OmniMVS model

# Function to load the pretrained model
def load_model(weights_path, root_dir):
    """
    Load the OmniMVS model with the correct parameters.
    """
    sweep = SphericalSweeping(root_dir)  # Initialize sweeping
    ndisp = 64                # Set number of disparities (modify as needed)
    min_depth = 0.1           # Set minimum depth (modify as needed)
    w, h = 512, 512           # Input width & height

    model = OmniMVS(sweep, ndisp, min_depth, w, h)  # Initialize model
    
    # Load checkpoint
    checkpoint = torch.load(weights_path, map_location=torch.device("cpu"))

    # Check if checkpoint has 'net_state_dict'
    if "net_state_dict" in checkpoint:
        checkpoint = checkpoint["net_state_dict"]

    # Filter out mismatched keys
    model_state_dict = model.state_dict()
    filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in model_state_dict}

    # Load the filtered state dictionary
    model.load_state_dict(filtered_checkpoint, strict=False)

    model.eval()
    return model

def preprocess_image(image_path, resize_to=(640, 320), grayscale=True):
    """
    Load and preprocess an image for OmniMVS input.
    """
    # Load image in grayscale or color
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(image_path, flag)

    if image is None:
        raise FileNotFoundError(f"Error: Image not found at {image_path}")

    # Resize the image correctly
    image = cv2.resize(image, resize_to, interpolation=cv2.INTER_LINEAR)

    if grayscale:
        image = np.expand_dims(image, axis=-1)  # Add channel dimension (H, W) → (H, W, 1)

    image = image / 255.0  # Normalize to [0, 1]

    # Convert to tensor with correct shape (1, C, H, W)
    tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # Shape: (1, C, H, W)
    print(f"Preprocessed image shape: {tensor.shape}")  # Ensure (1, C, 320, 640)
    return tensor


def test_omnimvs(model, image_paths):
    """
    Prepare a batch of images from multiple cameras and pass it through the model.
    """
    cam_list = ['cam1', 'cam2', 'cam3', 'cam4']
    batch = {}

    # Ensure we have exactly 4 input images
    if len(image_paths) != 4:
        raise ValueError("Error: Expected 4 input images (one per camera)")

    for i, cam in enumerate(cam_list):
        batch[cam] = preprocess_image(image_paths[i])

    # Ensure all tensors are on the same device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for cam in cam_list:
        batch[cam] = batch[cam].to(device)  # Move tensors to GPU (if available)

    print("Final batch shapes before model execution:")
    for cam in batch:
        print(f"{cam}: {batch[cam].shape}")  # Expected: [1, C, 320, 640]

    with torch.no_grad():
        depth_map = model(batch)  # Ensure model expects a dictionary batch
        return depth_map.squeeze(0).cpu().numpy()



# Main testing function
if __name__ == "__main__":

    # Path to pretrained weights
    weights_path = r"C:\Users\MernaSherif\Desktop\SVMRepo\omnimvs_MyRepo\omnimvs-pytorch\weights\omnimvs_plus_ft.pt"  # Replace with your actual weights file path

    # Define the dataset root directory correctly
    root_dir = r"C:\Users\MernaSherif\Desktop\SVMRepo\omnimvs_dataset\urban_sunny\sunny"  

    # Call load_model with both required parameters
    model = load_model(weights_path, root_dir)

    # Paths to test fisheye images
    test_image_paths = [
        r"C:\Users\MernaSherif\Desktop\SVMRepo\omnimvs_dataset\urban_sunny\sunny\cam1\0001.png",  # cam_front
        r"C:\Users\MernaSherif\Desktop\SVMRepo\omnimvs_dataset\urban_sunny\sunny\cam2\0001.png",  # cam_right
        r"C:\Users\MernaSherif\Desktop\SVMRepo\omnimvs_dataset\urban_sunny\sunny\cam3\0001.png",  # cam_rear
        r"C:\Users\MernaSherif\Desktop\SVMRepo\omnimvs_dataset\urban_sunny\sunny\cam4\0001.png"   # cam_left
    ]

    # Run the model on the test images
    depth_map = test_omnimvs(model, test_image_paths)

    # Normalize depth values to the visible range
    depth_map = np.nan_to_num(depth_map)  # Remove NaN values if any
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)  # Normalize to [0,1]
    depth_map = (depth_map * 255).astype(np.uint8)  # Convert to [0,255]

    # Save and display
    cv2.imshow("Depth Map", depth_map)
    cv2.imwrite("depth_map_output.png", depth_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

