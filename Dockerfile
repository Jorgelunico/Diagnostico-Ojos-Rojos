# Imagen base con Python y TensorFlow
FROM python:3.9-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y 
    #\
    #libgl1-mesa-glx \
    #libglib2.0-0 \
    #&& rm -rf /var/lib/apt/lists/*

# Directorio de trabajo
WORKDIR /app

# Copiar archivos necesarios
COPY requirements.txt .
#COPY app.py .
#COPY models/ ./models/
#COPY reports/ ./reports/

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Puerto para Streamlit
EXPOSE 8501

# Comando de ejecución
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
