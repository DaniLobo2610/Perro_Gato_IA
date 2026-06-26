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
# 🐶🐱 Modelo Predictivo de Perros y Gatos

### Universidad Nacional Autónoma de Honduras

**Ingeniería en Sistemas**

**Inteligencia Artificial**

---

### 👨‍💻 Alberto Daniel Lobo Chavarría

Este sistema utiliza un modelo de Inteligencia Artificial basado en **MobileNetV2**
para identificar automáticamente si una imagen corresponde a un **Perro** o un **Gato**.
""")


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

    # Realizar la predicción
    clase, confianza = predecir(imagen)

    st.divider()
    st.subheader("Resultado")

    if clase == "Perros":
        st.success("🐶 Esto es un Perro")
    else:
        st.success("🐱 Esto es un Gato")

    st.progress(min(confianza / 100, 1.0))

    st.write(f"**Confianza del modelo:** {confianza:.2f}%")