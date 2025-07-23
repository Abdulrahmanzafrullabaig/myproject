from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import torch
import torchvision.models as torchvision_models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
import sqlite3
import uuid
from functools import wraps
import json
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import lime
from lime import lime_image
import shap
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import timm  # For mobilenetv4_conv_large
from typing import Tuple, Optional, Dict, Any
import torch.nn as nn
import logging
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Gemini API Configuration
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'YOUR_GEMINI_API_KEY')
if GEMINI_API_KEY == 'YOUR_GEMINI_API_KEY':
    logger.warning("Gemini API key not found. Please set the GEMINI_API_KEY environment variable.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_DIR'] = 'models'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/reports', exist_ok=True)
os.makedirs(app.config['MODEL_DIR'], exist_ok=True)

# DR Classification stages
DR_STAGES = {
    0: "No DR - No visible abnormalities",
    1: "Mild DR - Microaneurysms only",
    2: "Moderate DR - More than microaneurysms",
    3: "Severe DR - Extensive abnormalities",
    4: "Proliferative DR - Neovascularization present"
}
CLASSES = list(DR_STAGES.values())
MODEL_ARCHITECTURES = ['resnet50', 'efficientnet', 'vgg16', 'mobilenetv4']

# Model names and file mappings
models_dict = {}  # Global dictionary to store models and conv layers
model_names = ['resnet50', 'efficientnet', 'vgg16', 'mobilenetv4']
model_files = {
    'resnet50': 'model1.pth',
    'efficientnet': 'model2.pth',
    'vgg16': 'model3.pth',
    'mobilenetv4': 'model4.pth'
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def get_transforms(arch: str) -> transforms.Compose:
    """Get image transforms based on architecture"""
    size = (384, 384) if arch == 'mobilenetv4' else (224, 224)
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def init_db():
    """Initialize the database"""
    try:
        conn = sqlite3.connect('diabetic_retinopathy.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      username TEXT UNIQUE NOT NULL,
                      email TEXT UNIQUE NOT NULL,
                      password_hash TEXT NOT NULL,
                      role TEXT NOT NULL,
                      full_name TEXT NOT NULL,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        c.execute('''CREATE TABLE IF NOT EXISTS reports
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      user_id INTEGER NOT NULL,
                      filename TEXT NOT NULL,
                      predictions TEXT NOT NULL,
                      final_prediction INTEGER NOT NULL,
                      confidence REAL NOT NULL,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      doctor_notes TEXT,
                      FOREIGN KEY (user_id) REFERENCES users (id))''')
        c.execute('''CREATE TABLE IF NOT EXISTS appointments
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      patient_id INTEGER NOT NULL,
                      doctor_id INTEGER NOT NULL,
                      appointment_date TIMESTAMP NOT NULL,
                      status TEXT DEFAULT 'pending',
                      notes TEXT,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      FOREIGN KEY (patient_id) REFERENCES users (id),
                      FOREIGN KEY (doctor_id) REFERENCES users (id))''')
        conn.commit()
        logger.info("Database initialized successfully")
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        raise
    finally:
        conn.close()

def initialize_model(arch: str) -> Tuple[nn.Module, nn.Module]:
    """Initialize an empty model with correct architecture"""
    logger.info(f"Initializing model: {arch}")
    try:
        if arch == 'resnet50':
            model = torchvision_models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
            final_conv_layer = model.layer4[-1].conv3
        elif arch == 'efficientnet':
            model = torchvision_models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
            final_conv_layer = model.features[-1][0]
        elif arch == 'vgg16':
            model = torchvision_models.vgg16(weights=None)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, len(CLASSES))
            final_conv_layer = model.features[-3]
        elif arch == 'mobilenetv4':
            try:
                model = timm.create_model('mobilenetv4_conv_large.e600_r384_in1k', 
                                         pretrained=False, 
                                         num_classes=len(CLASSES))
                final_conv_layer = model.conv_head
            except Exception as e:
                logger.warning(f"timm mobilenetv4_conv_large failed: {e}. Falling back to mobilenet_v3_large")
                model = torchvision_models.mobilenet_v3_large(weights=None)
                model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(CLASSES))
                final_conv_layer = model.features[-1][0]
        else:
            raise ValueError(f"Unsupported architecture: {arch}")
        
        return model, final_conv_layer
    except Exception as e:
        logger.error(f"Model initialization failed for {arch}: {e}")
        raise RuntimeError(f"Model initialization failed for {arch}: {str(e)}")

def load_model(arch: str, path: str) -> Tuple[Optional[nn.Module], Optional[nn.Module], Optional[str]]:
    """Load model weights from file with error handling"""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        logger.warning(f"Model file not found or empty at {path}. Initializing with random weights.")
        model, final_conv_layer = initialize_model(arch)
        model.to(device).eval()
        return model, final_conv_layer, None

    try:
        model, final_conv_layer = initialize_model(arch)
        state_dict = torch.load(path, map_location=device)
        if arch == 'mobilenetv4' and any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        model.to(device).eval()
        test_input = torch.rand(1, 3, 384, 384).to(device) if arch == 'mobilenetv4' else torch.rand(1, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(test_input)
            if output.shape[1] != len(CLASSES):
                logger.error(f"Model output shape mismatch for {arch}")
                return None, None, f"Model output shape mismatch for {arch}"
        logger.info(f"Model {arch} loaded successfully from {path}")
        return model, final_conv_layer, None
    except Exception as e:
        logger.error(f"Error loading {arch}: {e}")
        return None, None, f"Error loading {arch}: {str(e)}"

def load_all_models() -> Dict[str, Tuple[Any, Any, Optional[str]]]:
    """Load all models with error tracking"""
    model_paths = {
        'resnet50': os.path.join(app.config['MODEL_DIR'], 'model1.pth'),
        'efficientnet': os.path.join(app.config['MODEL_DIR'], 'model2.pth'),
        'vgg16': os.path.join(app.config['MODEL_DIR'], 'model3.pth'),
        'mobilenetv4': os.path.join(app.config['MODEL_DIR'], 'model4.pth')
    }
    
    loaded_models = {}
    errors = []
    
    for arch in MODEL_ARCHITECTURES:
        model, conv_layer, error = load_model(arch, model_paths.get(arch, ''))
        if error:
            errors.append(f"{arch}: {error}")
        else:
            loaded_models[arch] = (model, conv_layer)
    
    if errors:
        logger.error("Model loading errors:\n" + "\n".join(errors))
    
    return loaded_models

def login_required(f):
    """Decorator to require login"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def doctor_required(f):
    """Decorator to require doctor role"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or session.get('role') != 'doctor':
            flash('Access denied. Doctor privileges required.', 'error')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

def preprocess_image(image_path: str, arch: str) -> Tuple[torch.Tensor, Image.Image]:
    """Preprocess image for model prediction"""
    transform = get_transforms(arch)
    try:
        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0), image
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise

def predict_with_models(image_path: str) -> Dict[str, Dict]:
    """Get predictions from all models"""
    predictions = {}
    
    for arch in model_names:
        model_data = models_dict.get(arch)
        if not model_data:
            predictions[arch] = {'error': f'Model {arch} not available'}
            continue
        
        model, _ = model_data
        try:
            image_tensor, _ = preprocess_image(image_path, arch)
            image_tensor = image_tensor.to(device)
            
            with torch.no_grad():
                output = model(image_tensor)
                probabilities = torch.softmax(output, dim=1)[0]
                predicted_class = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_class].item()
                predictions[arch] = {
                    'probabilities': probabilities.cpu().numpy().tolist(),
                    'predicted_class': predicted_class,
                    'confidence': confidence
                }
        except Exception as e:
            logger.error(f"Prediction failed for {arch}: {e}")
            predictions[arch] = {'error': f'Prediction failed: {str(e)}'}
    
    return predictions

def majority_voting(predictions: Dict[str, Dict]) -> Tuple[int, float, Dict]:
    """Apply majority voting to ensemble predictions"""
    votes = []
    confidences = []
    
    for model_name, pred in predictions.items():
        if 'error' not in pred:
            votes.append(pred['predicted_class'])
            confidences.append(pred['confidence'])
    
    if not votes:
        logger.warning("No valid predictions for majority voting")
        return 0, 0.0, {}
    
    vote_counts = {}
    for vote in votes:
        vote_counts[vote] = vote_counts.get(vote, 0) + 1
    
    final_prediction = max(vote_counts.items(), key=lambda x: x[1])[0]
    ensemble_confidence = np.mean(confidences) if confidences else 0.0
    
    logger.info(f"Majority voting result: class {final_prediction}, confidence {ensemble_confidence:.2%}")
    return final_prediction, ensemble_confidence, vote_counts

def generate_grad_cam(image_path: str, model: nn.Module, final_conv: nn.Module, target_class: int, arch: str) -> Optional[np.ndarray]:
    """Generate Grad-CAM visualization"""
    size = (384, 384) if arch == 'mobilenetv4' else (224, 224)
    image = cv2.imread(image_path)
    image = cv2.resize(image, size)
    image_tensor, _ = preprocess_image(image_path, arch)
    
    try:
        cam = GradCAM(model=model, target_layers=[final_conv])
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = cam(input_tensor=image_tensor)[0, :]
        image_float = np.float32(image) / 255
        visualization = show_cam_on_image(image_float, grayscale_cam, use_rgb=True)
        logger.info(f"Grad-CAM generated successfully for {arch}")
        return visualization
    except Exception as e:
        logger.error(f"Grad-CAM failed for {arch}: {e}")
        return None

def generate_lime_explanation(image_path: str, arch: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Generate LIME explanation"""
    size = (384, 384) if arch == 'mobilenetv4' else (224, 224)
    try:
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image.resize(size))
        model_data = models_dict.get(arch)
        
        if not model_data:
            logger.error(f"No model data for {arch}")
            return None, None
        
        model, _ = model_data
        transform = get_transforms(arch)
        
        def predict_fn(images):
            tensors = [transform(Image.fromarray(img).convert('RGB')) for img in images]
            batch = torch.stack(tensors).to(device)
            with torch.no_grad():
                outputs = model(batch)
                probs = torch.softmax(outputs, dim=1)
            return probs.cpu().numpy()
        
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            image_array,
            predict_fn,
            top_labels=5,
            num_samples=1000
        )
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=5,
            hide_rest=False
        )
        logger.info(f"LIME explanation generated for {arch}")
        return temp, mask
    except Exception as e:
        logger.error(f"LIME failed for {arch}: {e}")
        return None, None

def generate_pdf_report(report_data: Dict, output_path: str):
    """Generate PDF report"""
    try:
        c = canvas.Canvas(output_path, pagesize=letter)
        width, height = letter
        c.setFont("Helvetica-Bold", 20)
        c.drawString(50, height - 50, "Diabetic Retinopathy Analysis Report")
        c.setFont("Helvetica", 12)
        y_position = height - 100
        c.drawString(50, y_position, f"Patient: {report_data['patient_name']}")
        y_position -= 20
        c.drawString(50, y_position, f"Date: {report_data['date']}")
        y_position -= 40
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_position, "Analysis Results:")
        y_position -= 30
        c.setFont("Helvetica", 12)
        c.drawString(50, y_position, f"Final Prediction: {DR_STAGES[report_data['final_prediction']]}")
        y_position -= 20
        c.drawString(50, y_position, f"Confidence: {report_data['confidence']:.2%}")
        y_position -= 30
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y_position, "Individual Model Predictions:")
        y_position -= 20
        c.setFont("Helvetica", 10)
        for model_name, pred in report_data['predictions'].items():
            if 'error' not in pred:
                c.drawString(70, y_position, f"{model_name}: {DR_STAGES[pred['predicted_class']]} ({pred['confidence']:.2%})")
            else:
                c.drawString(70, y_position, f"{model_name}: Error ({pred['error']})")
            y_position -= 15
        c.save()
        logger.info(f"PDF report generated at {output_path}")
    except Exception as e:
        logger.error(f"PDF report generation failed: {e}")
        raise

def get_gemini_explanation(stage: int) -> Dict[str, str]:
    """Get explanation from Gemini API"""
    if GEMINI_API_KEY == 'YOUR_GEMINI_API_KEY':
        return {
            "explanation": "Gemini API key not configured. Please contact the administrator.",
            "suggestions": "N/A",
            "precautions": "N/A"
        }
    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""
        Provide a clear, concise explanation for a patient who has been diagnosed with
        Diabetic Retinopathy Stage {stage} ({DR_STAGES[stage]}).
        The explanation should be easy to understand for a non-medical person.
        Also provide a list of suggestions and precautions.
        Format the output as a JSON object with three keys: "explanation", "suggestions", and "precautions".
        """
        response = model.generate_content(prompt)
        # Handle potential API errors or unexpected response formats
        if response and response.text:
            # Clean the response to ensure it's valid JSON
            cleaned_text = response.text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:-3].strip()
            return json.loads(cleaned_text)
        else:
            raise ValueError("Empty or invalid response from Gemini API")
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return {
            "explanation": f"Error generating explanation: {e}",
            "suggestions": "Please consult your doctor for recommendations.",
            "precautions": "Please consult your doctor for recommendations."
        }


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        role = request.form['role']
        full_name = request.form['full_name']
        if not all([username, email, password, role, full_name]):
            flash('All fields are required!', 'error')
            return render_template('register.html')
        password_hash = generate_password_hash(password)
        try:
            conn = sqlite3.connect('diabetic_retinopathy.db')
            c = conn.cursor()
            c.execute('INSERT INTO users (username, email, password_hash, role, full_name) VALUES (?, ?, ?, ?, ?)',
                     (username, email, password_hash, role, full_name))
            conn.commit()
            conn.close()
            flash('Registration successful! Please login.', 'success')
            logger.info(f"User {username} registered")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists!', 'error')
        except Exception as e:
            logger.error(f"Registration failed: {e}")
            flash('Registration failed. Please try again.', 'error')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        try:
            conn = sqlite3.connect('diabetic_retinopathy.db')
            c = conn.cursor()
            c.execute('SELECT * FROM users WHERE username = ?', (username,))
            user = c.fetchone()
            conn.close()
            if user and check_password_hash(user[3], password):
                session['user_id'] = user[0]
                session['username'] = user[1]
                session['role'] = user[4]
                session['full_name'] = user[5]
                flash('Login successful!', 'success')
                logger.info(f"User {username} logged in")
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid username or password!', 'error')
        except Exception as e:
            logger.error(f"Login failed: {e}")
            flash('Login failed. Please try again.', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    username = session.get('username', 'Unknown')
    session.clear()
    flash('You have been logged out.', 'info')
    logger.info(f"User {username} logged out")
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    try:
        conn = sqlite3.connect('diabetic_retinopathy.db')
        c = conn.cursor()
        c.execute('SELECT * FROM reports WHERE user_id = ? ORDER BY created_at DESC', (session['user_id'],))
        reports = c.fetchall()
        conn.close()
        return render_template('dashboard.html', reports=reports, dr_stages=DR_STAGES)
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        flash('Error loading dashboard. Please try again.', 'error')
        return redirect(url_for('index'))

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected!', 'error')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected!', 'error')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filename = f"{uuid.uuid4()}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(filepath)
                
                predictions = predict_with_models(filepath)
                final_prediction, ensemble_confidence, vote_counts = majority_voting(predictions)
                
                grad_cam_results = {}
                for arch in model_names:
                    model_data = models_dict.get(arch)
                    if model_data and 'error' not in predictions.get(arch, {}):
                        model, final_conv = model_data
                        grad_cam_img = generate_grad_cam(filepath, model, final_conv, final_prediction, arch)
                        if grad_cam_img is not None:
                            _, buffer = cv2.imencode('.png', grad_cam_img)
                            grad_cam_b64 = base64.b64encode(buffer).decode('utf-8')
                            grad_cam_results[arch] = grad_cam_b64
                
                lime_img, lime_segments = generate_lime_explanation(filepath, 'resnet50')
                if lime_img is not None:
                    _, buffer = cv2.imencode('.png', lime_img)
                    lime_b64 = base64.b64encode(buffer).decode('utf-8')
                else:
                    lime_b64 = None
                
                conn = sqlite3.connect('diabetic_retinopathy.db')
                c = conn.cursor()
                c.execute('''INSERT INTO reports (user_id, filename, predictions, final_prediction, confidence)
                            VALUES (?, ?, ?, ?, ?)''',
                         (session['user_id'], filename, json.dumps(predictions), final_prediction, ensemble_confidence))
                report_id = c.lastrowid
                conn.commit()
                conn.close()
                
                original_image = Image.open(filepath).convert('RGB')
                original_image.save('temp_original.png')
                with open('temp_original.png', 'rb') as img_file:
                    original_b64 = base64.b64encode(img_file.read()).decode('utf-8')
                os.remove('temp_original.png')
                
                chart_data = {
                    'labels': [arch for arch in predictions if 'error' not in predictions[arch]],
                    'data': [predictions[arch]['confidence'] for arch in predictions if 'error' not in predictions[arch]]
                }
                
                gemini_explanation = get_gemini_explanation(final_prediction)

                return render_template('results.html',
                                     predictions=predictions,
                                     final_prediction=final_prediction,
                                     ensemble_confidence=ensemble_confidence,
                                     vote_counts=vote_counts,
                                     dr_stages=DR_STAGES,
                                     grad_cam_results=grad_cam_results,
                                     lime_result=lime_b64,
                                     original_image=original_b64,
                                     report_id=report_id,
                                     chart_data=chart_data,
                                     gemini_explanation=gemini_explanation)
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                flash('Error processing image. Please try again.', 'error')
                return redirect(request.url)
        else:
            flash('Please upload a valid image file (PNG, JPG, JPEG)!', 'error')
    return render_template('predict.html')

@app.route('/share_report', methods=['POST'])
@login_required
def share_report():
    try:
        data = request.get_json()
        report_id = data['report_id']
        doctor_email = data['doctor_email']

        conn = sqlite3.connect('diabetic_retinopathy.db')
        c = conn.cursor()
        c.execute('SELECT * FROM reports WHERE id = ? AND user_id = ?', (report_id, session['user_id']))
        report = c.fetchone()
        conn.close()

        if not report:
            return jsonify({'success': False, 'error': 'Report not found'})

        # This is a placeholder for a real email sending implementation
        logger.info(f"Sharing report {report_id} with {doctor_email}")
        # In a real application, you would use a library like smtplib or a service like SendGrid
        # to send an email with a link to the report.
        # For this example, we'll just log the action.

        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error sharing report: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/download_report/<int:report_id>')
@login_required
def download_report(report_id):
    try:
        conn = sqlite3.connect('diabetic_retinopathy.db')
        c = conn.cursor()
        c.execute('SELECT * FROM reports WHERE id = ? AND user_id = ?', (report_id, session['user_id']))
        report = c.fetchone()
        conn.close()
        if not report:
            flash('Report not found!', 'error')
            return redirect(url_for('dashboard'))
        report_data = {
            'patient_name': session['full_name'],
            'date': report[7],
            'final_prediction': report[4],
            'confidence': report[5],
            'predictions': json.loads(report[3])
        }
        pdf_filename = f"report_{report_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf_path = os.path.join('static/reports', pdf_filename)
        generate_pdf_report(report_data, pdf_path)
        return send_file(pdf_path, as_attachment=True, download_name=pdf_filename)
    except Exception as e:
        logger.error(f"Report download error: {e}")
        flash('Error generating report. Please try again.', 'error')
        return redirect(url_for('dashboard'))

@app.route('/doctors')
@login_required
def doctors():
    if session.get('role') != 'patient':
        flash('Access denied. Patient privileges required.', 'error')
        return redirect(url_for('dashboard'))
    try:
        conn = sqlite3.connect('diabetic_retinopathy.db')
        c = conn.cursor()
        c.execute('SELECT id, full_name, email FROM users WHERE role = "doctor"')
        doctors = c.fetchall()
        conn.close()
        return render_template('doctors.html', doctors=doctors)
    except Exception as e:
        logger.error(f"Doctors page error: {e}")
        flash('Error loading doctors. Please try again.', 'error')
        return redirect(url_for('dashboard'))

@app.route('/schedule_appointment', methods=['POST'])
@login_required
def schedule_appointment():
    try:
        doctor_id = request.form['doctor_id']
        appointment_date = request.form['appointment_date']
        notes = request.form.get('notes', '')
        conn = sqlite3.connect('diabetic_retinopathy.db')
        c = conn.cursor()
        c.execute('''INSERT INTO appointments (patient_id, doctor_id, appointment_date, notes)
                    VALUES (?, ?, ?, ?)''',
                 (session['user_id'], doctor_id, appointment_date, notes))
        conn.commit()
        conn.close()
        flash('Appointment scheduled successfully!', 'success')
        logger.info(f"Appointment scheduled for user {session['user_id']} with doctor {doctor_id}")
        return redirect(url_for('dashboard'))
    except Exception as e:
        logger.error(f"Appointment scheduling error: {e}")
        flash('Error scheduling appointment. Please try again.', 'error')
        return redirect(url_for('dashboard'))

@app.route('/appointments')
@login_required
@doctor_required
def appointments():
    try:
        conn = sqlite3.connect('diabetic_retinopathy.db')
        c = conn.cursor()
        c.execute('''SELECT a.*, u.full_name as patient_name, u.email as patient_email
                    FROM appointments a
                    JOIN users u ON a.patient_id = u.id
                    WHERE a.doctor_id = ?
                    ORDER BY a.appointment_date''', (session['user_id'],))
        appointments = c.fetchall()
        conn.close()
        return render_template('appointments.html', appointments=appointments)
    except Exception as e:
        logger.error(f"Appointments page error: {e}")
        flash('Error loading appointments. Please try again.', 'error')
        return redirect(url_for('dashboard'))

@app.route('/update_appointment_status', methods=['POST'])
@login_required
@doctor_required
def update_appointment_status():
    try:
        data = request.get_json()
        appointment_id = data['appointment_id']
        status = data['status']
        
        conn = sqlite3.connect('diabetic_retinopathy.db')
        c = conn.cursor()
        c.execute('UPDATE appointments SET status = ? WHERE id = ? AND doctor_id = ?',
                 (status, appointment_id, session['user_id']))
        conn.commit()
        conn.close()
        
        logger.info(f"Appointment {appointment_id} status updated to {status} by doctor {session['user_id']}")
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error updating appointment status: {e}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    try:
        init_db()
        models_dict = load_all_models()  # Update global models_dict
        if not models_dict:
            logger.critical("No models loaded successfully")
            raise RuntimeError("Critical: No models loaded successfully")
        logger.info("Application started successfully")
        app.run(debug=False, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.critical(f"Application startup failed: {e}")
        raise
