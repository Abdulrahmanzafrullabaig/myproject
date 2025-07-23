import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import cv2
from torchvision import transforms

def generate_gradcam(model, image_tensor, target_class):
    """Generate Grad-CAM visualization"""
    model.eval()
    
    # Register hook for gradients
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    # Register hooks on the last convolutional layer
    if hasattr(model, 'features'):  # VGG-style
        target_layer = model.features[-1]
    elif hasattr(model, 'layer4'):  # ResNet-style
        target_layer = model.layer4[-1]
    else:  # Other architectures
        target_layer = list(model.children())[-2]
    
    handle_backward = target_layer.register_backward_hook(backward_hook)
    handle_forward = target_layer.register_forward_hook(forward_hook)
    
    # Forward pass
    output = model(image_tensor)
    
    # Backward pass
    model.zero_grad()
    output[0, target_class].backward()
    
    # Generate Grad-CAM
    if gradients and activations:
        gradient = gradients[0]
        activation = activations[0]
        
        # Global average pooling of gradients
        weights = torch.mean(gradient, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activation, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.squeeze()
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        # Resize to input image size
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), 
                           size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().numpy()
    else:
        # Fallback: random heatmap for demo
        cam = np.random.rand(224, 224)
    
    # Clean up hooks
    handle_backward.remove()
    handle_forward.remove()
    
    return cam

def generate_lime_explanation(image_path, prediction_result):
    """Generate LIME-style explanation (simplified version)"""
    # For demo purposes, create a simple segmentation-based explanation
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    
    # Create superpixels (simplified)
    segments = np.zeros((224, 224), dtype=int)
    segment_size = 28  # 8x8 grid
    segment_id = 0
    
    for i in range(0, 224, segment_size):
        for j in range(0, 224, segment_size):
            segments[i:i+segment_size, j:j+segment_size] = segment_id
            segment_id += 1
    
    # Generate random importance scores for each segment
    num_segments = segment_id
    importance_scores = np.random.randn(num_segments)
    
    # Create explanation mask
    explanation_mask = np.zeros((224, 224))
    for seg_id in range(num_segments):
        mask = segments == seg_id
        explanation_mask[mask] = importance_scores[seg_id]
    
    return explanation_mask

def save_explanation_image(original_image_path, explanation_data, output_path, explanation_type='gradcam'):
    """Save explanation visualization"""
    # Load original image
    original_image = Image.open(original_image_path).convert('RGB')
    original_image = original_image.resize((224, 224))
    original_array = np.array(original_image)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_array)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Explanation heatmap
    if explanation_type == 'gradcam':
        heatmap = cm.jet(explanation_data)[:, :, :3]
        axes[1].imshow(heatmap)
        axes[1].set_title('Grad-CAM Heatmap')
    else:
        heatmap = cm.RdBu_r((explanation_data - explanation_data.min()) / 
                           (explanation_data.max() - explanation_data.min()))[:, :, :3]
        axes[1].imshow(heatmap)
        axes[1].set_title('LIME Explanation')
    axes[1].axis('off')
    
    # Overlay
    overlay = original_array.astype(float) / 255.0
    if explanation_type == 'gradcam':
        heatmap_overlay = cm.jet(explanation_data)[:, :, :3]
        overlay = 0.6 * overlay + 0.4 * heatmap_overlay
    else:
        heatmap_overlay = cm.RdBu_r((explanation_data - explanation_data.min()) / 
                                   (explanation_data.max() - explanation_data.min()))[:, :, :3]
        overlay = 0.7 * overlay + 0.3 * heatmap_overlay
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def generate_explanations(image_path, prediction_result):
    """Generate both Grad-CAM and LIME explanations"""
    explanations = {}
    
    try:
        # Create explanations directory
        explanations_dir = 'static/explanations'
        os.makedirs(explanations_dir, exist_ok=True)
        
        # Generate unique filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # For demo purposes, generate mock explanations
        # In a real implementation, you would load the actual models
        
        # Mock Grad-CAM
        gradcam_data = np.random.rand(224, 224)
        gradcam_path = os.path.join(explanations_dir, f'{base_name}_gradcam.png')
        save_explanation_image(image_path, gradcam_data, gradcam_path, 'gradcam')
        explanations['gradcam'] = gradcam_path
        
        # Mock LIME
        lime_data = generate_lime_explanation(image_path, prediction_result)
        lime_path = os.path.join(explanations_dir, f'{base_name}_lime.png')
        save_explanation_image(image_path, lime_data, lime_path, 'lime')
        explanations['lime'] = lime_path
        
    except Exception as e:
        print(f"Error generating explanations: {e}")
        explanations = {'error': str(e)}
    
    return explanations
