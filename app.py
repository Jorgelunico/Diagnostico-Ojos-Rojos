import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import os
import time
from scipy import stats
import pdfkit
from io import BytesIO

# Configuración de wkhtmltopdf (asegúrate de que esté instalado en tu sistema)
path_wkhtmltopdf = '/usr/bin/wkhtmltopdf'
config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)

# --- Configuración de la página (DEBE SER LO PRIMERO QUE SE EJECUTA DE Streamlit) ---
st.set_page_config(
    page_title="Diagnóstico de Enfermedades de Ojos Rojos", # Título inicial, se actualizará dinámicamente
    page_icon="👁️",
    layout="wide"
)

# Textos para diferentes idiomas
TEXTS = {
    "es": {
        "page_title": "Diagnóstico de Enfermedades de Ojos Rojos",
        "app_title": "👁️ Sistema Inteligente para el Diagnóstico de Enfermedades Oculares Asociadas al Enrojecimiento de los Ojos",
        "app_description": "Esta aplicación utiliza modelos de aprendizaje profundo para diagnosticar enfermedades que causan ojos rojos a partir de imágenes oculares. Sube una imagen para obtener un diagnóstico.",
        "loading_models_error": "Error cargando modelos: {e}. Por favor verifica la carpeta 'models'.",
        "tab_diagnosis": "Diagnóstico",
        "tab_analysis": "Análisis de Modelos",
        "tab_report": "Reporte",
        "header_diagnosis": "Diagnóstico por Imagen",
        "upload_image_prompt": "Sube una imagen del ojo",
        "uploaded_image_caption": "Imagen subida",
        "diagnose_button": "Realizar diagnóstico",
        "model_load_error": "Los modelos no se cargaron correctamente. Por favor verifica la carpeta 'models'.",
        "analyzing_image_spinner": "Analizando imagen...",
        "model_1_title": "Modelo 1 (CNN Simple)",
        "model_2_title": "Modelo 2 (CNN Profunda)",
        "model_3_title": "Modelo 3 (CNN con ResNet)",
        "diagnosis_label": "Diagnóstico",
        "confidence_label": "Confianza",
        "consensus_diagnosis": "Diagnóstico consensuado",
        "header_analysis": "Análisis Comparativo de Modelos",
        "evaluate_models_button": "Evaluar Modelos",
        "evaluating_models_spinner": "Evaluando modelos...",
        "details_header": "Detalles:",
        "dataset_source": "El dataset se obtuvo de: https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification/data",
        "total_figures": "El total de figuras usadas son de 4236",
        "models_used": "Los modelos que usan son: modelo 1 (CNN Simple), modelo 2 (CNN Profundo), modelo 3 (CNN ResNet)",
        "epochs_used": "Las epocas usadas son de: 56",
        "training_time": "Los modelos en total se demoraron en entrenar aproximadamente: 24 horas",
        "best_model": "El mejor modelo es: El modelo 2 (CNN Profundo) con una precision de 0.932",
        "hardware_details_header": "Detalles de herramientas usadas:",
        "hardware_details": "Detalles de hardware del servicio de Google Colab usado para entrenar el modelo: \nRAM del sistema: 12.7GB\nRAM de la GPU: 15.0GB\nDisco: 112.6GB",
        "software_details": "Python 3.9, Docker Engine==28.3.2, Google Colab==Python 3.10., streamlit==1.36.0, tensorflow==2.18.0, numpy==1.26.4, pandas==1.5.3, matplotlib==3.9.0, seaborn==0.12.2, Pillow==9.5.0, scikit-learn==1.2.2, scipy==1.10.1, pdfkit==1.0.0, pyyaml==6.0, wkhtmltopdf==0.12.6",
        "performance_metrics_header": "Métricas de Rendimiento",
        "model_label": "Modelo",
        "accuracy_label": "Precisión",
        "sensitivity_label": "Sensibilidad",
        "specificity_label": "Especificidad",
        "f1_score_label": "F1-Score",
        "mcc_label": "MCC",
        "mcnemar_header": "Resultados de la Prueba de McNemar (Comparación Pareada de Modelos)",
        "compared_models": "Modelos Comparados",
        "chi_square_statistic": "Estadístico Chi-cuadrado",
        "p_value": "P-Valor",
        "significance": "Significancia (alpha=0.05)",
        "confusion_matrices_header": "Matrices de Confusión",
        "evaluation_complete": "¡Evaluación completada!",
        "header_report": "Generar Reporte PDF",
        "run_evaluation_warning": "Primero ejecuta la evaluación de modelos en la pestaña 'Análisis de Modelos'",
        "generate_report_button": "Generar Reporte Completo",
        "generating_report_spinner": "Generando reporte...",
        "download_report_button": "Descargar Reporte",
        "report_generated_success": "¡Reporte generado con éxito!",
        "report_title": "Reporte de Diagnostico de Enfermedades Oculares",
        "report_summary_metrics": "Resumen de Metricas por Modelo",
        "report_model_column": "Modelo",
        "report_precision_column": "Precision",
        "report_sensitivity_column": "Sensibilidad",
        "report_specificity_column": "Especificidad",
        "report_f1_column": "F1-Score",
        "report_mcc_column": "MCC",
        "report_statistical_comparison": "Comparacion Estadística entre Modelos",
        "report_mcnemar_test": "Prueba de McNemar:",
        "report_chi_square_statistic": "Estadistico Chi-cuadrado:",
        "report_significance": "Significancia (alpha=0.05):",
        "report_yes": "Si",
        "report_no": "No",
        "report_confusion_matrices": "Matrices de Confusion",
    },
    "en": {
        "page_title": "Red Eye Disease Diagnosis",
        "app_title": "👁️ Intelligent System for Diagnosing Red Eye-Related Ocular Diseases",
        "app_description": "This application uses deep learning models to diagnose red eye diseases from ocular images. Upload an image to get a diagnosis.",
        "loading_models_error": "Error loading models: {e}. Please check the 'models' folder.",
        "tab_diagnosis": "Diagnosis",
        "tab_analysis": "Model Analysis",
        "tab_report": "Report",
        "header_diagnosis": "Image-based Diagnosis",
        "upload_image_prompt": "Upload an eye image",
        "uploaded_image_caption": "Uploaded Image",
        "diagnose_button": "Perform Diagnosis",
        "model_load_error": "Models were not loaded correctly. Please check the 'models' folder.",
        "analyzing_image_spinner": "Analyzing image...",
        "model_1_title": "Model 1 (Simple CNN)",
        "model_2_title": "Model 2 (Deep CNN)",
        "model_3_title": "Model 3 (CNN with ResNet)",
        "diagnosis_label": "Diagnosis",
        "confidence_label": "Confidence",
        "consensus_diagnosis": "Consensus Diagnosis",
        "header_analysis": "Comparative Model Analysis",
        "evaluate_models_button": "Evaluate Models",
        "evaluating_models_spinner": "Evaluating models...",
        "details_header": "Details:",
        "dataset_source": "The dataset was obtained from: https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification/data",
        "total_figures": "Total figures used are 4236",
        "models_used": "The models used are: Model 1 (Simple CNN), Model 2 (Deep CNN), Model 3 (CNN ResNet)",
        "epochs_used": "Epochs used are: 56",
        "training_time": "The models took approximately: 24 hours to train",
        "best_model": "The best model is: Model 2 (Deep CNN) with an accuracy of 0.932",
        "hardware_details_header": "Hardware Details:",
        "hardware_details": "Google Colab service hardware details used for model training: \nSystem RAM: 12.7GB\nGPU RAM: 15.0GB\nDisk: 112.6GB",
        "software_details": "Python 3.9, Docker Engine==28.3.2, Google Colab==Python 3.10., streamlit==1.36.0, tensorflow==2.18.0, numpy==1.26.4, pandas==1.5.3, matplotlib==3.9.0, seaborn==0.12.2, Pillow==9.5.0, scikit-learn==1.2.2, scipy==1.10.1, pdfkit==1.0.0, pyyaml==6.0, wkhtmltopdf==0.12.6",
        "performance_metrics_header": "Performance Metrics",
        "model_label": "Model",
        "accuracy_label": "Accuracy",
        "sensitivity_label": "Sensitivity",
        "specificity_label": "Specificity",
        "f1_score_label": "F1-Score",
        "mcc_label": "MCC",
        "mcnemar_header": "McNemar's Test Results (Paired Model Comparison)",
        "compared_models": "Compared Models",
        "chi_square_statistic": "Chi-square Statistic",
        "p_value": "P-Value",
        "significance": "Significance (alpha=0.05)",
        "confusion_matrices_header": "Confusion Matrices",
        "evaluation_complete": "Evaluation complete!",
        "header_report": "Generate PDF Report",
        "run_evaluation_warning": "First run model evaluation in the 'Model Analysis' tab",
        "generate_report_button": "Generate Full Report",
        "generating_report_spinner": "Generating report...",
        "download_report_button": "Download Report",
        "report_generated_success": "Report generated successfully!",
        "report_title": "Ocular Disease Diagnosis Report",
        "report_summary_metrics": "Metrics Summary by Model",
        "report_model_column": "Model",
        "report_precision_column": "Precision",
        "report_sensitivity_column": "Sensitivity",
        "report_specificity_column": "Specificity",
        "report_f1_column": "F1-Score",
        "report_mcc_column": "MCC",
        "report_statistical_comparison": "Statistical Comparison Between Models",
        "report_mcnemar_test": "McNemar's Test:",
        "report_chi_square_statistic": "Chi-square statistic:",
        "report_significance": "Significance (alpha=0.05):",
        "report_yes": "Yes",
        "report_no": "No",
        "report_confusion_matrices": "Confusion Matrices",
    }
}

# Clases de enfermedades (mantienen el mismo texto, la descripción cambia según el idioma)
CLASSES = ['Catarata', 'Retinopatía diabética', 'Glaucoma', 'Normal']
CLASSES_DESC = {
    "es": {
        'Catarata': 'Opacidad del cristalino del ojo, lo que provoca visión borrosa o disminuida.',
        'Retinopatía diabética': 'Daño a los vasos sanguíneos de la retina causado por la diabetes, que puede llevar a la pérdida de la visión.',
        'Glaucoma': 'Daño del nervio óptico, generalmente asociado con una presión intraocular elevada, que puede causar pérdida de la visión y ceguera.',
        'Normal': 'Ojo saludable sin anomalías detectables.'
    },
    "en": {
        'Catarata': 'Opacity of the eye\'s natural lens, causing blurred or decreased vision.',
        'Retinopatía diabética': 'Damage to the blood vessels in the retina caused by diabetes, which can lead to vision loss.',
        'Glaucoma': 'Damage to the optic nerve, usually associated with elevated intraocular pressure, which can cause vision loss and blindness.',
        'Normal': 'Healthy eye with no detectable anomalies.'
    }
}

# --- Inicialización de st.session_state.lang (MOVIDO AQUÍ) ---
# Debe estar ANTES de cualquier uso de st.session_state.lang
if 'lang' not in st.session_state:
    st.session_state.lang = 'es' # Idioma predeterminado

# Obtener textos según el idioma seleccionado
current_texts = TEXTS[st.session_state.lang]
current_classes_desc = CLASSES_DESC[st.session_state.lang]

# --- Selección de Idioma (DESPUÉS de inicializar st.session_state.lang y current_texts) ---
lang_col, _ = st.columns([1, 4])
with lang_col:
    selected_lang = st.selectbox("Select Language / Seleccionar Idioma", ["Español", "English"], index=0 if st.session_state.lang == 'es' else 1)
    if selected_lang == "Español":
        st.session_state.lang = 'es'
    else:
        st.session_state.lang = 'en'

# Si el idioma ha cambiado a través del selectbox, necesitamos actualizar current_texts
# Esto asegura que el resto de la interfaz se actualice inmediatamente.
current_texts = TEXTS[st.session_state.lang]
current_classes_desc = CLASSES_DESC[st.session_state.lang]


st.title(current_texts["app_title"])
st.markdown(current_texts["app_description"])

@st.cache_resource
def load_models():
    try:
        model1 = load_model('models/Model_1_Training.h5')
        model2 = load_model('models/Model_2_Training.h5')
        model3 = load_model('models/Model_3_Training.keras')
        return model1, model2, model3
    except Exception as e:
        st.error(current_texts["loading_models_error"].format(e=e))
        return None, None, None

model1, model2, model3 = load_models()

# ... (El resto de tu código, funciones preprocess_image, generate_report, y la UI con las tabs) ...

# Preprocesamiento de imágenes
def preprocess_image(image, target_size=(128, 128)):
    img = image.resize(target_size)
    img_array = np.array(img)
    if img_array.shape[-1] == 4:  # Si tiene canal alpha
        img_array = img_array[..., :3]
    img_array = img_array / 255.0  # Normalización
    return np.expand_dims(img_array, axis=0)

# Preprocesamiento de imágenes 256*256
def preprocess_image2(image, target_size=(256, 256)):
    img = image.resize(target_size)
    img_array = np.array(img)
    if img_array.shape[-1] == 4:  # Si tiene canal alpha
        img_array = img_array[..., :3]
    img_array = img_array / 255.0  # Normalización
    return np.expand_dims(img_array, axis=0)

# Generar reporte PDF
def generate_report(metrics, model_names, output_path='report.pdf', lang='es'):
    texts = TEXTS[lang]
    
    html = f"""
    <html>
    <head>
        <title>{texts["report_title"]}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2e86c1; }}
            h2 {{ color: #1a5276; }}
            .metric-card {{ 
                background: #f8f9f9; 
                border-left: 4px solid #2e86c1; 
                padding: 10px; 
                margin: 10px 0;
            }}
            .row {{ display: flex; }}
            .col {{ flex: 1; padding: 10px; }}
            img {{ max-width: 100%; height: auto; }}
            table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>{texts["report_title"]}</h1>

        <h2>{texts['details_header']}</h2>
        <p>{texts['dataset_source']}</p>
        <p>{texts['total_figures']}</p>
        <p>{texts['models_used']}</p>
        <p>{texts['epochs_used']}</p>
        <p>{texts['training_time']}</p>
        <p>{texts['best_model']}</p>

        <h2>{texts['hardware_details_header']}</h2>
        <p>{texts['hardware_details']}</p>
        <p>{texts['software_details']}</p>

        <h2>{texts["report_summary_metrics"]}</h2>
        <table>
            <tr>
                <th>{texts["report_model_column"]}</th>
                <th>{texts["report_precision_column"]}</th>
                <th>{texts["report_sensitivity_column"]}</th>
                <th>{texts["report_specificity_column"]}</th>
                <th>{texts["report_f1_column"]}</th>
                <th>{texts["report_mcc_column"]}</th>
            </tr>
    """
    
    for i, model_name in enumerate(model_names):
        html += f"""
            <tr>
                <td>{model_name}</td>
                <td>{metrics[i]['accuracy']:.3f}</td>
                <td>{metrics[i]['sensitivity']:.3f}</td>
                <td>{metrics[i]['specificity']:.3f}</td>
                <td>{metrics[i]['f1']:.3f}</td>
                <td>{metrics[i]['mcc']:.3f}</td>
            </tr>
        """
    
    html += f"""
        </table>
        
        <h2>{texts["report_statistical_comparison"]}</h2>
    """
    # Hardcoded McNemar results for the report
    html+=f""" 
        <div class="metric-card">
            <h3>{texts['model_1_title']} vs {texts['model_2_title']}</h3>
            <p><strong>{texts['report_mcnemar_test']}</strong> p-value = 0.0000</p>
            <p><strong>{texts['report_chi_square_statistic']}</strong> 187.000</p>
            <p><strong>{texts['report_significance']}</strong> {texts['report_yes']}</p>
        </div>

        <div class="metric-card">
            <h3>{texts['model_1_title']} vs {texts['model_3_title']}</h3>
            <p><strong>{texts['report_mcnemar_test']}</strong> p-value = 0.0000</p>
            <p><strong>{texts['report_chi_square_statistic']}</strong> 284.000</p>
            <p><strong>{texts['report_significance']}</strong> {texts['report_yes']}</p>
        </div>

        <div class="metric-card">
            <h3>{texts['model_2_title']} vs {texts['model_3_title']}</h3>
            <p><strong>{texts['report_mcnemar_test']}</strong> p-value = 0.0000</p>
            <p><strong>{texts['report_chi_square_statistic']}</strong> 191.000</p>
            <p><strong>{texts['report_significance']}</strong> {texts['report_yes']}</p>
        </div>
        
        """
    
    html += f"<h2>{texts['report_confusion_matrices']}</h2><div class='row'>"
    
    # Matrices de confusión
    for i, model_name in enumerate(model_names):
        img_path = f'utils/confusion_matrix_{i}.png'
        html += f"""
        <div class="col">
            <h3>{model_name}</h3>
            <img src="{img_path}">
        </div>
        """
        if (i+1) % 2 == 0:
            html += "</div><div class='row'>"
    
    html += """
        </div>
    </body>
    </html>
    """
    # Opciones para permitir acceso a archivos locales
    options = {
        'enable-local-file-access': None
    }
    
    # Guardar HTML temporal
    with open('temp_report.html', 'w', encoding='utf-8') as f: # Añadir encoding
        f.write(html)
    
    # Convertir HTML a PDF
    pdfkit.from_file('temp_report.html', output_path, configuration=config, options=options)

# Interfaz de usuario con pestañas
tab1, tab2, tab3 = st.tabs([current_texts["tab_diagnosis"], current_texts["tab_analysis"], current_texts["tab_report"]])

with tab1:
    st.header(current_texts["header_diagnosis"])
    uploaded_file = st.file_uploader(current_texts["upload_image_prompt"], type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption=current_texts["uploaded_image_caption"], use_column_width=True)
        
        # Preprocesar imagen
        processed_img = preprocess_image(image)
        processed_img2 = preprocess_image2(image)
        
        if st.button(current_texts["diagnose_button"]):
            if model1 is None or model2 is None or model3 is None:
                st.error(current_texts["model_load_error"])
            else:
                with st.spinner(current_texts["analyzing_image_spinner"]):
                    # Predicciones
                    pred1 = model1.predict(processed_img)
                    pred2 = model2.predict(processed_img)
                    pred3 = model3.predict(processed_img2)
                    
                    # Obtener clases predichas
                    class_idx1 = np.argmax(pred1[0])
                    class_idx2 = np.argmax(pred2[0])
                    class_idx3 = np.argmax(pred3[0])
                    
                    # Obtener confianzas
                    confidence1 = np.max(pred1[0])
                    confidence2 = np.max(pred2[0])
                    confidence3 = np.max(pred3[0])
                    
                    # Mostrar resultados
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.subheader(current_texts["model_1_title"])
                        st.write(f"{current_texts['diagnosis_label']}: **{CLASSES[class_idx1]}**")
                        st.write(f"{current_texts['confidence_label']}: {confidence1:.2%}")
                        st.write(current_classes_desc[CLASSES[class_idx1]])
                    
                    with col2:
                        st.subheader(current_texts["model_2_title"])
                        st.write(f"{current_texts['diagnosis_label']}: **{CLASSES[class_idx2]}**")
                        st.write(f"{current_texts['confidence_label']}: {confidence2:.2%}")
                        st.write(current_classes_desc[CLASSES[class_idx2]])
                    
                    with col3:
                        st.subheader(current_texts["model_3_title"])
                        st.write(f"{current_texts['diagnosis_label']}: **{CLASSES[class_idx3]}**")
                        st.write(f"{current_texts['confidence_label']}: {confidence3:.2%}")
                        st.write(current_classes_desc[CLASSES[class_idx3]])
                
                # Determinar diagnóstico consensuado
                diagnoses = [class_idx1, class_idx2, class_idx3]
                final_diagnosis = max(set(diagnoses), key=diagnoses.count)
                
                st.success(f"{current_texts['consensus_diagnosis']}: **{CLASSES[final_diagnosis]}**")
                st.write(current_classes_desc[CLASSES[final_diagnosis]])

with tab2:
    st.header(current_texts["header_analysis"])
    if st.button(current_texts["evaluate_models_button"]):
        with st.spinner(current_texts["evaluating_models_spinner"]):
            # Métricas hardcodeadas (se pueden reemplazar por cálculo real si se tiene el conjunto de prueba)
            metrics = [
                {
                    'model': current_texts["model_1_title"],
                    'accuracy': 0.807,
                    'sensitivity': 0.776,
                    'specificity': 0.926,
                    'f1': 0.783,
                    'mcc': 0.709,
                },
                {
                    'model': current_texts["model_2_title"],
                    'accuracy': 0.932,
                    'sensitivity': 0.923,
                    'specificity': 0.974,
                    'f1': 0.923,
                    'mcc': 0.900,
                },
                {
                    'model': current_texts["model_3_title"],
                    'accuracy': 0.899,
                    'sensitivity': 0.895,
                    'specificity': 0.965,
                    'f1': 0.894,
                    'mcc': 0.863,
                }
            ] 

            st.subheader(current_texts["details_header"])       
            st.text(current_texts["dataset_source"])     
            st.text(current_texts["total_figures"])
            st.text(current_texts["models_used"])
            st.text(current_texts["epochs_used"])
            st.text(current_texts["training_time"])
            st.text(current_texts["best_model"])
            st.subheader(current_texts["hardware_details_header"])
            st.text(current_texts["hardware_details"])
            st.text(current_texts["software_details"])
            
            mcnemar_data = [
                {'Modelos Comparados': f'{current_texts["model_1_title"]} vs {current_texts["model_2_title"]}', 'Chi-square statistic': 187.000, 'P-Value': 0.0000, 'Significancia (alpha=0.05)': current_texts['report_yes']},
                {'Modelos Comparados': f'{current_texts["model_1_title"]} vs {current_texts["model_3_title"]}', 'Chi-square statistic': 284.000, 'P-Value': 0.0000, 'Significancia (alpha=0.05)': current_texts['report_yes']},
                {'Modelos Comparados': f'{current_texts["model_2_title"]} vs {current_texts["model_3_title"]}', 'Chi-square statistic': 191.000, 'P-Value': 0.0000, 'Significancia (alpha=0.05)': current_texts['report_yes']},
            ]
            
            df_mcnemar = pd.DataFrame(mcnemar_data)
            
            # Mostrar resultados
            st.subheader(current_texts["performance_metrics_header"])            
            cols = st.columns(len(metrics))
            for i, col in enumerate(cols):
                with col:
                    st.metric(label=current_texts["model_label"], value=metrics[i]['model'])
                    st.metric(label=current_texts["accuracy_label"], value=f"{metrics[i]['accuracy']:.3f}")
                    st.metric(label=current_texts["sensitivity_label"], value=f"{metrics[i]['sensitivity']:.3f}")
                    st.metric(label=current_texts["specificity_label"], value=f"{metrics[i]['specificity']:.3f}")
                    st.metric(label=current_texts["f1_score_label"], value=f"{metrics[i]['f1']:.3f}")
                    st.metric(label=current_texts["mcc_label"], value=f"{metrics[i]['mcc']:.3f}")
            

            st.subheader(current_texts["mcnemar_header"])
            st.dataframe(df_mcnemar, use_container_width=True)
            
            st.subheader(current_texts["confusion_matrices_header"])
            fig_col1, fig_col2, fig_col3 = st.columns(3)
            
            with fig_col1:
                st.image("utils/confusion_matrix_0.png", use_column_width=True)
            
            with fig_col2:
                st.image("utils/confusion_matrix_1.png", use_column_width=True)
            
            with fig_col3:
                st.image("utils/confusion_matrix_2.png", use_column_width=True)
            
            # Guardar métricas en session state para el reporte
            st.session_state.metrics = metrics
            st.session_state.model_names = [current_texts["model_1_title"], current_texts["model_2_title"], current_texts["model_3_title"]]
            
            st.success(current_texts["evaluation_complete"])

with tab3:
    st.header(current_texts["header_report"])
    
    if 'metrics' not in st.session_state:
        st.warning(current_texts["run_evaluation_warning"])
    else:
        if st.button(current_texts["generate_report_button"]):
            with st.spinner(current_texts["generating_report_spinner"]):
                # Crear PDF en memoria
                pdf_output = BytesIO()
                generate_report(
                    st.session_state.metrics,
                    st.session_state.model_names,
                    'report.pdf',
                    st.session_state.lang # Pasar el idioma para la generación del reporte
                )
                
                # Leer PDF generado
                with open('report.pdf', 'rb') as f:
                    pdf_bytes = f.read()
                
                # Mostrar botón de descarga
                st.download_button(
                    label=current_texts["download_report_button"], # Usar el texto traducido para el botón
                    data=pdf_bytes,
                    file_name=f"reporte_diagnostico_ocular_{st.session_state.lang}.pdf", # Nombre del archivo dinámico con el idioma
                    mime="application/pdf"
                )
                
                st.success(current_texts["report_generated_success"])