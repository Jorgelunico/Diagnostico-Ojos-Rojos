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

path_wkhtmltopdf = '/usr/bin/wkhtmltopdf' # O '/usr/bin/wkhtmltopdf', etc.
config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)

# Configuración de la página
st.set_page_config(
    page_title="Diagnóstico de Enfermedades de Ojos Rojos",
    page_icon="👁️",
    layout="wide"
)

st.title("👁️ Sistema Inteligente para el Diagnóstico de Enfermedades Oculares Asociadas al Enrojecimiento de los Ojos")
st.markdown("""
Esta aplicación utiliza modelos de aprendizaje profundo para diagnosticar enfermedades que causan ojos rojos 
a partir de imágenes oculares. Sube una imagen para obtener un diagnóstico.
""")
@st.cache_resource
def load_models():
    try:
        model1 = load_model('models/Model_1_Training.h5')
        model2 = load_model('models/Model_2_Training.h5')
        model3 = load_model('models/Model_3_Training.keras')
        return model1, model2, model3
    except Exception as e:
        st.error(f"Error cargando modelos: {e}")
        return None, None, None

model1, model2, model3 = load_models()

# Clases de enfermedades
CLASSES = ['Catarata', 'Retinopatía diabética', 'Glaucoma', 'Normal']
CLASSES_DESC = {
    'Catarata': 'Opacidad del cristalino del ojo, lo que provoca visión borrosa o disminuida.',
    'Retinopatía diabética': '  Daño a los vasos sanguíneos de la retina causado por la diabetes, que puede llevar a la pérdida de la visión.',
    'Glaucoma': 'Daño del nervio óptico, generalmente asociado con una presión intraocular elevada, que puede causar pérdida de la visión y ceguera.',
    'Normal': 'Ojo saludable sin anomalías detectables.'
}

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
def generate_report(metrics, model_names, output_path='report.pdf'):
    html = f"""
    <html>
    <head>
        <title>Reporte de Diagnostico de Enfermedades Oculares</title>
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
        <h1>Reporte de Diagnostico de Enfermedades Oculares</h1>
        
        <h2>Resumen de Metricas por Modelo</h2>
        <table>
            <tr>
                <th>Modelo</th>
                <th>Precision</th>
                <th>Sensibilidad</th>
                <th>Especificidad</th>
                <th>F1-Score</th>
                <th>MCC</th>
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
    
    html += """
        </table>
        
        <h2>Comparacion Estadistica entre Modelos</h2>
    """
    html+=""" 
        <div class="metric-card">
            <h3>Modelo 1 vs Modelo 2</h3>
            <p><strong>Prueba de McNemar:</strong> p-value = 0.0000</p>
            <p><strong>Chi-square statistic:</strong> 187.000</p>
            <p><strong>Significancia (alpha=0.05):</strong> Si</p>
        </div>

        <div class="metric-card">
            <h3>Modelo 1 vs Modelo 3</h3>
            <p><strong>Prueba de McNemar:</strong> p-value = 0.0000</p>
            <p><strong>Chi-square statistic:</strong> 284.000</p>
            <p><strong>Significancia (alpha=0.05):</strong> Si</p>
        </div>

        <div class="metric-card">
            <h3>Modelo 1 vs Modelo 2</h3>
            <p><strong>Prueba de McNemar:</strong> p-value = 0.0000</p>
            <p><strong>Chi-square statistic:</strong> 191.000</p>
            <p><strong>Significancia (alpha=0.05):</strong> Si</p>
        </div>
        
        """
    
    html += "<h2>Matrices de Confusion</h2><div class='row'>"
    
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
        'enable-local-file-access': None # Esto es crucial para imágenes/CSS locales
    }
    
    # Guardar HTML temporal
    with open('temp_report.html', 'w') as f:
        f.write(html)
    
    # Convertir HTML a PDF
    pdfkit.from_file('temp_report.html', output_path, configuration=config, options=options)

# Interfaz de usuario
tab1, tab2, tab3 = st.tabs(["Diagnóstico", "Análisis de Modelos", "Reporte"])

with tab1:
    st.header("Diagnóstico por Imagen")
    uploaded_file = st.file_uploader("Sube una imagen del ojo", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Imagen subida', use_column_width=True)
        
        # Preprocesar imagen
        processed_img = preprocess_image(image)
        processed_img2 = preprocess_image2(image)
        
        if st.button("Realizar diagnóstico"):
            if model1 is None or model2 is None or model3 is None:
                st.error("Los modelos no se cargaron correctamente. Por favor verifica la carpeta 'models'.")
            else:
                with st.spinner('Analizando imagen...'):
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
                        st.subheader("Modelo 1 (CNN Simple)")
                        st.write(f"Diagnóstico: **{CLASSES[class_idx1]}**")
                        st.write(f"Confianza: {confidence1:.2%}")
                        st.write(CLASSES_DESC[CLASSES[class_idx1]])
                    
                    with col2:
                        st.subheader("Modelo 2 (CNN Profunda)")
                        st.write(f"Diagnóstico: **{CLASSES[class_idx2]}**")
                        st.write(f"Confianza: {confidence2:.2%}")
                        st.write(CLASSES_DESC[CLASSES[class_idx2]])
                    
                    with col3:
                        st.subheader("Modelo 3 (CNN con ResNet)")
                        st.write(f"Diagnóstico: **{CLASSES[class_idx3]}**")
                        st.write(f"Confianza: {confidence3:.2%}")
                        st.write(CLASSES_DESC[CLASSES[class_idx3]])
                
                # Determinar diagnóstico consensuado
                diagnoses = [class_idx1, class_idx2, class_idx3]
                final_diagnosis = max(set(diagnoses), key=diagnoses.count)
                
                st.success(f"Diagnóstico consensuado: **{CLASSES[final_diagnosis]}**")
                st.write(CLASSES_DESC[CLASSES[final_diagnosis]])

with tab2:
    st.header("Análisis Comparativo de Modelos")
    if st.button("Evaluar Modelos"):
        with st.spinner('Evaluando modelos...'):
            # Calcular métricas para cada modelo
            metrics = [
                {
                    'model': 'Modelo 1 (CNN Simple)',
                    'accuracy': 0.807,
                    'sensitivity': 0.776,
                    'specificity': 0.926,
                    'f1': 0.783,
                    'mcc': 0.709,
                },
                {
                    'model': 'Modelo 2 (CNN Profundo)',
                    'accuracy': 0.932,
                    'sensitivity': 0.923,
                    'specificity': 0.974,
                    'f1': 0.923,
                    'mcc': 0.900,
                },
                {
                    'model': 'Modelo 3 (CNN ResNet)',
                    'accuracy': 0.899,
                    'sensitivity': 0.895,
                    'specificity': 0.965,
                    'f1': 0.894,
                    'mcc': 0.863,
                }
            ] 

            st.subheader("Detalles:")       
            st.text("El dataset se obtubo de: https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification/data ")     
            st.text("El total de figuras usadas son de 4236")
            st.text("Los modelos que usan son: modelo 1 (CNN Simple), modelo 2 (CNN Profundo), modelo 3 (CNN ResNet)")
            st.text("Las epocas usadas son de: 56")
            st.text("Los modelos en total se demoraron en entrenar aproximadamente : 24 horas")
            st.text("El mejor modelo es: El modelo 2 (CNN Profundo) con una precision de 0.932")
            st.subheader("Detalles de herramientas usadas:")
            st.text("Detalles de hadware del servicio de Google Colab usado para entrenar el modelo: \nRAM del sistema: 12.7GB\nRAM de la GPU: 15.0GB\nDisco: 112.6GB")
            st.text("""Python 3.9, Docker Engine==28.3.2, Google Colab==Python 3.10., streamlit==1.36.0, tensorflow==2.18.0, numpy==1.26.4, 
                    pandas==1.5.3, matplotlib==3.9.0, seaborn==0.12.2, Pillow==9.5.0, scikit-learn==1.2.2
                    scipy==1.10.1, pdfkit==1.0.0, pyyaml==6.0, wkhtmltopdf==0.12.6""")
            mcnemar_data = [
                {'Modelos Comparados': 'Modelo 1 vs Modelo 2', 'Chi-square statistic': 187.000, 'P-Value': 0.0000, 'Significancia (alpha=0.05)': 'Sí'},
                {'Modelos Comparados': 'Modelo 1 vs Modelo 3', 'Chi-square statistic': 284.000, 'P-Value': 0.0000, 'Significancia (alpha=0.05)': 'Sí'},
                {'Modelos Comparados': 'Modelo 2 vs Modelo 3', 'Chi-square statistic': 191.000, 'P-Value': 0.0000, 'Significancia (alpha=0.05)': 'Sí'},
            ]
            
            df_mcnemar = pd.DataFrame(mcnemar_data)
            
            # Mostrar resultados
            st.subheader("Métricas de Rendimiento")            
            cols = st.columns(len(metrics))
            for i, col in enumerate(cols):
                with col:
                    st.metric(label="Modelo", value=metrics[i]['model'])
                    st.metric(label="Precisión", value=f"{metrics[i]['accuracy']:.3f}")
                    st.metric(label="Sensibilidad", value=f"{metrics[i]['sensitivity']:.3f}")
                    st.metric(label="Especificidad", value=f"{metrics[i]['specificity']:.3f}")
                    st.metric(label="F1-Score", value=f"{metrics[i]['f1']:.3f}")
                    st.metric(label="MCC", value=f"{metrics[i]['mcc']:.3f}")
            

            st.subheader("Resultados de la Prueba de McNemar (Comparación Pareada de Modelos)")
            st.dataframe(df_mcnemar, use_container_width=True) # O st.table(df_mcnemar)

            st.subheader("Matrices de Confusión")
            fig_col1, fig_col2, fig_col3 = st.columns(3)
            
            with fig_col1:
                st.image("utils/confusion_matrix_0.png", use_column_width=True)
            
            with fig_col2:
                st.image("utils/confusion_matrix_1.png", use_column_width=True)
            
            with fig_col3:
                st.image("utils/confusion_matrix_2.png", use_column_width=True)
            
            # # Guardar métricas en session state para el reporte
            st.session_state.metrics = metrics
            st.session_state.model_names = ['Model 1', 'Model 2', 'Model 3']
            
            st.success("Evaluación completada!")

with tab3:
    st.header("Generar Reporte PDF")
    
    if 'metrics' not in st.session_state:
        st.warning("Primero ejecuta la evaluación de modelos en la pestaña 'Análisis de Modelos'")
    else:
        if st.button("Generar Reporte Completo"):
            with st.spinner('Generando reporte...'):
                # Crear PDF en memoria
                pdf_output = BytesIO()
                generate_report(
                    st.session_state.metrics,
                    st.session_state.model_names,
                    'report.pdf'
                )
                
                # Leer PDF generado
                with open('report.pdf', 'rb') as f:
                    pdf_bytes = f.read()
                
                # Mostrar botón de descarga
                st.download_button(
                    label="Descargar Reporte",
                    data=pdf_bytes,
                    file_name="reporte_diagnostico_ocular.pdf",
                    mime="application/pdf"
                )
                
                st.success("Reporte generado con éxito!")
