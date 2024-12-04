# Version Time Series
# Base, son librerias ya creadas (imagenes), ver en el dockerhub python
FROM python:3.11-slim

# Determinamos nuestro directorio de trabajo
WORKDIR /app

# Dependencias y librerias. Hacer un instalador de dependencias y librerias a utilizar
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    git \
    && apt-get clean
    
# Agregamos nuestros archivos # todos los archivos de la carpeta agregarlos a /app
COPY . /app

#RUN apk add --no-cache git
# Instalar requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
#RUN pip install git+https://github.com/amazon-science/chronos-forecasting.git
#RUN pip install "tsfm_public[notebooks] @ git+https://github.com/ibm-granite/granite-tsfm.git@v0.2.14"
#RUN pip install nixtla

# Como se ejecuta. Al darle docker run que ejecutara el contenedor
ENTRYPOINT ["python", "predict.py"]
