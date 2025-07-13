# Sistema Inteligente para el Diagnóstico de Enfermedades Oculares Asociadas al Enrojecimiento de los Ojos

## Descripción del Proyecto
Este proyecto es una aplicación web desarrollada con Streamlit que utiliza modelos de aprendizaje profundo para diagnosticar enfermedades oculares relacionadas con el enrojecimiento de los ojos a partir de imágenes oculares. La aplicación permite subir imágenes, realizar diagnósticos con tres modelos diferentes de redes neuronales convolucionales (CNN), comparar el rendimiento de los modelos y generar reportes en formato PDF con los resultados.

## Características Principales
- Diagnóstico automático de enfermedades oculares: Catarata, Retinopatía diabética, Glaucoma y ojo Normal.
- Uso de tres modelos de aprendizaje profundo para mejorar la precisión del diagnóstico.
- Visualización de resultados y confianza de cada modelo.
- Análisis comparativo de modelos con métricas de rendimiento y pruebas estadísticas.
- Generación de reportes PDF con resumen de métricas, comparación estadística y matrices de confusión.

## Estructura del Proyecto
- `app.py`: Código principal de la aplicación Streamlit.
- `models/`: Carpeta que contiene los modelos entrenados (`Model_1_Training.h5`, `Model_2_Training.h5`, `Model_3_Training.keras`).
- `utils/`: Contiene imágenes de matrices de confusión usadas en la aplicación.
- `training/`: Notebooks y scripts para el entrenamiento de los modelos.
- `evaluation/`: Notebooks para la evaluación y análisis de los modelos.
- `reports/`: Carpeta para almacenar reportes generados.
- `requirements.txt`: Dependencias de Python necesarias para ejecutar la aplicación.
- `Dockerfile`: Archivo para construir una imagen Docker con la aplicación.

## Datos de Muestra
Los datos de muestra no están incluidos en este repositorio. Puede descargar los datos desde el siguiente enlace de Kaggle:
https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification/data

## Instalación y Ejecución

### Requisitos Previos
- Python 3.9 o superior
- pip instalado
- (Opcional) Docker para ejecutar en contenedor

### Instalación Local
1. Clonar el repositorio:
   ```bash
   git clone <url-del-repositorio>
   cd Diagnostico-Ojos-Rojos
   ```
2. Crear y activar un entorno virtual (opcional pero recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```
3. Instalar las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. Ejecutar la aplicación:
   ```bash
   streamlit run app.py
   ```
5. Abrir en el navegador la URL que Streamlit indique (por defecto http://localhost:8501).

### Ejecución con Docker
1. Construir la imagen Docker:
   ```bash
   docker build -t diagnostico-ojos-rojos .
   ```
2. Ejecutar el contenedor:
   ```bash
   docker run -p 8501:8501 diagnostico-ojos-rojos
   ```
3. Acceder a la aplicación en http://localhost:8501.

## Uso de la Aplicación
- En la pestaña **Diagnóstico**, subir una imagen del ojo para obtener el diagnóstico de las enfermedades.
- En la pestaña **Análisis de Modelos**, evaluar y comparar el rendimiento de los tres modelos con métricas y matrices de confusión.
- En la pestaña **Reporte**, generar y descargar un reporte PDF con los resultados del diagnóstico y análisis.

## Modelos de Aprendizaje Profundo
- **Modelo 1 (CNN Simple):** Red neuronal convolucional básica para diagnóstico.
- **Modelo 2 (CNN Profunda):** Red convolucional más profunda para mayor precisión.
- **Modelo 3 (CNN con ResNet):** Modelo avanzado basado en arquitectura ResNet para mejorar el rendimiento.
---

Este proyecto facilita el diagnóstico automatizado de enfermedades oculares, ayudando a profesionales de la salud a tomar decisiones informadas basadas en imágenes médicas.
