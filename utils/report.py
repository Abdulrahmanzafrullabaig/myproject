from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import datetime
import os

def generate_pdf_report(user, prediction_result, prediction_id):
    """Generate PDF report for DR prediction"""
    
    # Create reports directory
    reports_dir = 'reports'
    os.makedirs(reports_dir, exist_ok=True)
    
    # Generate filename
    filename = f'DR_Report_{prediction_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
    filepath = os.path.join(reports_dir, filename)
    
    # Create PDF document
    doc = SimpleDocTemplate(filepath, pagesize=A4)
    story = []
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.darkblue,
        alignment=1  # Center alignment
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.darkblue
    )
    
    # Title
    story.append(Paragraph("Diabetic Retinopathy Analysis Report", title_style))
    story.append(Spacer(1, 20))
    
    # Patient Information
    story.append(Paragraph("Patient Information", heading_style))
    patient_data = [
        ['Name:', user.name],
        ['Email:', user.email],
        ['Age:', str(user.age) if user.age else 'N/A'],
        ['Gender:', user.gender if user.gender else 'N/A'],
        ['Phone:', user.phone if user.phone else 'N/A'],
        ['Report Date:', datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    ]
    
    patient_table = Table(patient_data, colWidths=[2*inch, 3*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (1, 0), (1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(patient_table)
    story.append(Spacer(1, 20))
    
    # Prediction Results
    story.append(Paragraph("Analysis Results", heading_style))
    
    ensemble_result = prediction_result['ensemble']
    dr_stage = prediction_result['dr_stage']
    
    # Main diagnosis
    diagnosis_text = f"""
    <b>Diagnosis:</b> {dr_stage['name']}<br/>
    <b>Confidence:</b> {ensemble_result['confidence']:.2%}<br/>
    <b>Description:</b> {dr_stage['description']}<br/>
    <b>Recommendations:</b> {dr_stage['advice']}
    """
    
    story.append(Paragraph(diagnosis_text, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Model Predictions Table
    story.append(Paragraph("Individual Model Predictions", heading_style))
    
    model_data = [['Model', 'Predicted Class', 'Confidence', 'Stage']]
    for model_name, model_result in prediction_result['individual_models'].items():
        predicted_class = model_result['predicted_class']
        confidence = model_result['confidence']
        stage_name = f"Stage {predicted_class}"
        
        model_data.append([
            model_name.upper(),
            str(predicted_class),
            f"{confidence:.2%}",
            stage_name
        ])
    
    # Add ensemble result
    model_data.append([
        'ENSEMBLE',
        str(ensemble_result['predicted_class']),
        f"{ensemble_result['confidence']:.2%}",
        f"Stage {ensemble_result['predicted_class']}"
    ])
    
    model_table = Table(model_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1.5*inch])
    model_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -2), colors.beige),
        ('BACKGROUND', (0, -1), (-1, -1), colors.lightblue),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(model_table)
    story.append(Spacer(1, 20))
    
    # DR Stages Information
    story.append(Paragraph("Diabetic Retinopathy Stages", heading_style))
    
    stages_info = """
    <b>Stage 0 - No DR:</b> No signs of diabetic retinopathy<br/>
    <b>Stage 1 - Mild NPDR:</b> Microaneurysms present<br/>
    <b>Stage 2 - Moderate NPDR:</b> Hemorrhages and exudates<br/>
    <b>Stage 3 - Severe NPDR:</b> Extensive hemorrhages<br/>
    <b>Stage 4 - Proliferative DR:</b> Neovascularization present<br/>
    """
    
    story.append(Paragraph(stages_info, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Disclaimer
    story.append(Paragraph("Important Disclaimer", heading_style))
    disclaimer_text = """
    This report is generated by an AI system for screening purposes only. 
    It should not be used as a substitute for professional medical diagnosis. 
    Please consult with a qualified ophthalmologist for proper medical evaluation 
    and treatment recommendations.
    """
    
    story.append(Paragraph(disclaimer_text, styles['Normal']))
    
    # Build PDF
    doc.build(story)
    
    return filepath
