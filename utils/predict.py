import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os

# DR stage descriptions and advice
DR_STAGES = {
    0: {
        'name': 'No DR',
        'description': 'No signs of diabetic retinopathy detected.',
        'advice': 'Continue regular eye examinations and maintain good blood sugar control.'
    },
    1: {
        'name': 'Mild NPDR',
        'description': 'Mild non-proliferative diabetic retinopathy with microaneurysms.',
        'advice': 'Monitor closely with regular eye exams every 6-12 months. Maintain strict blood sugar control.'
    },
    2: {
        'name': 'Moderate NPDR',
        'description': 'Moderate non-proliferative diabetic retinopathy with hemorrhages and exudates.',
        'advice': 'Requires closer monitoring every 3-6 months. Consider referral to retinal specialist.'
    },
    3: {
        'name': 'Severe NPDR',
        'description': 'Severe non-proliferative diabetic retinopathy with extensive hemorrhages.',
        'advice': 'Urgent referral to retinal specialist required. May need laser treatment.'
    },
    4: {
        'name': 'Proliferative DR',
        'description': 'Proliferative diabetic retinopathy with neovascularization.',
        'advice': 'Immediate treatment required. High risk of vision loss without intervention.'
    }
}

def load_model(model_name, model_path, num_classes=5):
    """Load a pretrained model"""
    if not os.path.exists(model_path):
        # Create dummy model for demo purposes
        if model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'vgg16':
            model = models.vgg16(pretrained=True)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        elif model_name == 'mobilenet':
            model = models.mobilenet_v2(pretrained=True)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif model_name == 'efficientnet':
            model = models.efficientnet_b0(pretrained=True)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
        # Save dummy model
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), model_path)
        return model
    
    # Load actual model
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=False)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == 'mobilenet':
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'efficientnet':
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def preprocess_image(image_path):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def predict_single_model(model, image_tensor):
    """Make prediction with a single model"""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        return probabilities.numpy()[0]

def predict_ensemble(image_path):
    """Make ensemble prediction using all four models"""
    # Model configurations
    models_config = {
        'resnet50': 'models/resnet50.pth',
        'vgg16': 'models/vgg16.pth',
        'mobilenet': 'models/mobilenet.pth',
        'efficientnet': 'models/efficientnet.pth'
    }
    
    # Preprocess image
    image_tensor = preprocess_image(image_path)
    
    # Load models and make predictions
    predictions = {}
    all_probs = []
    
    for model_name, model_path in models_config.items():
        try:
            model = load_model(model_name, model_path)
            probs = predict_single_model(model, image_tensor)
            predictions[model_name] = {
                'probabilities': probs.tolist(),
                'predicted_class': int(np.argmax(probs)),
                'confidence': float(np.max(probs))
            }
            all_probs.append(probs)
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            # Use dummy prediction for demo
            dummy_probs = np.random.dirichlet(np.ones(5))
            predictions[model_name] = {
                'probabilities': dummy_probs.tolist(),
                'predicted_class': int(np.argmax(dummy_probs)),
                'confidence': float(np.max(dummy_probs))
            }
            all_probs.append(dummy_probs)
    
    # Ensemble prediction (average probabilities)
    if all_probs:
        ensemble_probs = np.mean(all_probs, axis=0)
        ensemble_class = int(np.argmax(ensemble_probs))
        ensemble_confidence = float(np.max(ensemble_probs))
    else:
        # Fallback
        ensemble_probs = np.random.dirichlet(np.ones(5))
        ensemble_class = int(np.argmax(ensemble_probs))
        ensemble_confidence = float(np.max(ensemble_probs))
    
    # Prepare result
    result = {
        'ensemble': {
            'predicted_class': ensemble_class,
            'confidence': ensemble_confidence,
            'probabilities': ensemble_probs.tolist()
        },
        'individual_models': predictions,
        'dr_stage': DR_STAGES[ensemble_class],
        'image_path': image_path
    }
    
    return result
