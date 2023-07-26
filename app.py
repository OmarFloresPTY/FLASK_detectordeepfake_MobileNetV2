from flask import Flask, render_template, request, send_from_directory, url_for
from PIL import Image
import cv2
import requests as rq
import numpy as np
import base64
from io import BytesIO
import os
import tensorflow as tf

app = Flask(__name__)

# Carga el modelo de clasificación fuera del alcance de la aplicación
modelo = tf.keras.models.load_model('./Modelo_Guardado')

def categorizar_imagen(imagen):
    img = Image.open(imagen)
    img = np.array(img).astype(float) / 255
    img = cv2.resize(img, (224, 224))
    prediccion = modelo.predict(img.reshape(-1, 224, 224, 3))
    return np.argmax(prediccion[0], axis=-1)

def categorizar_url(url):
    respuesta = rq.get(url)
    img = Image.open(BytesIO(respuesta.content))
    img = np.array(img).astype(float) / 255.0
    img = cv2.resize(img, (224, 224))
    prediccion = modelo.predict(img.reshape(-1, 224, 224, 3))
    return np.argmax(prediccion[0], axis=-1)

@app.route('/', methods=['GET', 'POST'])
def clasificacion():
    if request.method == 'POST':
        # Obtiene el archivo de imagen enviado desde el formulario
        imagen = request.files['imagen']
        # Obtiene el url de la imagen enviado desde el formulario
        url = request.form['image_url']
        
        if imagen:
            # Renderiza la plantilla con el resultado de la clasificación y la imagen
            with imagen as img:
                # Realiza la clasificación
                prediccion = categorizar_imagen(img)
                
                # Codifica la imagen como base64 para mostrarla en el HTML
                imagen_base64 = base64.b64encode(img.read()).decode("utf-8")

                # Renderiza la plantilla con el resultado de la clasificación y la imagen
                resultado = "Detectado como Real" if prediccion == 0 else "Detectado como DeepFake"
                return render_template('result_IMG.html', resultado=resultado, imagen=imagen_base64)
        
        if url:
            prediccion = categorizar_url(url)
            resultado = "Detectado como Real" if prediccion == 0 else "Detectado como DeepFake"
            return render_template('result_URL.html', imagen_url=url, resultado=resultado)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run()
