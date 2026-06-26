# ==================================================
# IMPORTACIONES
# ==================================================

import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image


# ==================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==================================================

st.set_page_config(
    page_title="Modelo Predictivo de Perros y Gatos",
    page_icon="🐶",
    layout="centered"
)


# ==================================================
# ENCABEZADO
# ==================================================

st.markdown("""
<style>

.titulo{
background: linear-gradient(90deg,#2563eb,#0ea5e9);
padding:25px;
border-radius:18px;
text-align:center;
color:white;
box-shadow:0px 6px 18px rgba(0,0,0,.25);
margin-bottom:25px;
}

.titulo h1{
margin:0;
font-size:38px;
}

.titulo h3{
margin:8px;
font-weight:400;
}

.info{
background:#F5F7FA;
padding:18px;
border-radius:15px;
border-left:6px solid #2563eb;
margin-top:20px;
margin-bottom:20px;
color:#1f2937;
font-size:17px;
line-height:1.7;
}

.info b{
color:#111827;
}

.footer{
text-align:center;
color:gray;
font-size:14px;
margin-top:30px;
}

</style>

<div class="titulo">

<h1>🐶🐱 Modelo Predictivo de Perros y Gatos</h1>

<h3>Universidad Nacional Autónoma de Honduras</h3>

<h3>Ingeniería en Sistemas</h3>

<h3>Inteligencia Artificial</h3>

</div>

<div class="info">

<b>Desarrollado por:</b><br>

👨‍💻 Alberto Daniel Lobo Chavarría

<br><br>

Este sistema utiliza un modelo de Deep Learning basado en <b>MobileNetV2</b> para clasificar imágenes de perros y gatos mediante Inteligencia Artificial.

</div>

""", unsafe_allow_html=True)

# ==================================================
# RUTAS
# ==================================================

IMG_SIZE = (224, 224)

MODEL_DIR = Path("Modelo_perro_gato")

MODEL_PATH = MODEL_DIR / "modelo_perro_gato.keras"

CLASS_PATH = MODEL_DIR / "class_names.json"


# ==================================================
# CARGAR MODELO
# ==================================================

@st.cache_resource
def cargar_modelo():

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

@st.cache_data
def cargar_clases():

    with open(CLASS_PATH, "r", encoding="utf-8") as f:

        return json.load(f)

modelo = cargar_modelo()

clases = cargar_clases()



# ==================================================
# PREPARAR IMAGEN
# ==================================================

def preparar_imagen(imagen):

    imagen = imagen.convert("RGB")
    imagen = imagen.resize(IMG_SIZE)

    img = np.array(imagen, dtype=np.float32)

    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    return img


# ==================================================
# PREDICCIÓN
# ==================================================

def predecir(imagen):

    img = preparar_imagen(imagen)

    pred = modelo.predict(img, verbose=0)[0]

    indice = np.argmax(pred)

    confianza = float(pred[indice]) * 100

    clase = clases[indice]

    return clase, confianza


# ==================================================
# SUBIR IMAGEN
# ==================================================

st.divider()

st.subheader("📤 Suba una imagen")

archivo = st.file_uploader(

    "Seleccione una imagen",

    type=["jpg","jpeg","png"]

)


# ==================================================
# MOSTRAR IMAGEN
# ==================================================

if archivo is not None:

    imagen = Image.open(archivo)

    st.image(
        imagen,
        caption="Imagen cargada",
        use_container_width=True
    )

    clase, confianza = predecir(imagen)

    st.divider()

    st.subheader("Resultado")

    if clase == "Perros":

        st.markdown("""
        <div style="
        background:#d1fae5;
        padding:20px;
        border-radius:15px;
        text-align:center;
        font-size:30px;
        font-weight:bold;
        color:#065f46;">
        🐶 Esto es un Perro
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div style="
        background:#fee2e2;
        padding:20px;
        border-radius:15px;
        text-align:center;
        font-size:30px;
        font-weight:bold;
        color:#991b1b;">
        🐱 Esto es un Gato
        </div>
        """, unsafe_allow_html=True)

    st.progress(confianza / 100)

    st.markdown(
        f"<h2 style='text-align:center;color:#2563eb;'>Confianza: {confianza:.2f}%</h2>",
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="footer">

    <hr>

    Modelo desarrollado con TensorFlow • MobileNetV2 • Streamlit

    </div>
    """, unsafe_allow_html=True)

