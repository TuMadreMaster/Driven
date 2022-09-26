# Seleccionamos la imagen base para el dockerfile de python
FROM python:3.9.12-bullseye

# Instalamos las dependencias 
ADD ./requirements.txt .
RUN pip install -r requirements.txt

# seteamos el working directory del contenedor
WORKDIR /code

# Copiamos todos los datos necesarios
COPY ./app /code/app

#CMD [ "python", "test.py"]