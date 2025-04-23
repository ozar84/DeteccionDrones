import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from ultralytics import YOLO
import base64
from io import BytesIO

# Configuración de la página
st.set_page_config(page_title="Detección de Drones", page_icon="🛸", layout="wide")

# Ruta al modelo y a los logos
MODEL_PATH = "C:/Users/alber/OneDrive/Escritorio/best.pt"
SENSIA_LOGO = "C:/Users/alber/OneDrive/Escritorio/Sensia.png"
MBIT_LOGO = "C:/Users/alber/OneDrive/Escritorio/mbit.jpg"

# Función para convertir imagen a base64
def image_to_base64(path, width=140, height=60):
    img = Image.open(path).convert("RGBA").resize((width, height))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

# Convertir imágenes a base64
sensia_b64 = image_to_base64(SENSIA_LOGO)
mbit_b64 = image_to_base64(MBIT_LOGO)

# Estilos y cabecera HTML
st.markdown(f"""
    <style>
    .header-container {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        background-color: #ffffff;
        padding: 0.5rem 2rem;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }}
    .header-logo {{
        height: 40px;
    }}
    .header-title {{
        font-size: 1.8rem;
        font-weight: bold;
        color: #222;
        text-align: center;
        flex-grow: 1;
    }}
    </style>

    <div class="header-container">
        <img src="data:image/png;base64,{sensia_b64}" class="header-logo">
        <div class="header-title">🛸 Detección de Drones</div>
        <img src="data:image/png;base64,{mbit_b64}" class="header-logo">
    </div>
""", unsafe_allow_html=True)

# Cargar modelo
@st.cache_resource
def load_model():
    try:
        return YOLO(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Error cargando el modelo: {e}")
        return None

# Carga de imagen
st.markdown("### 📷 Sube una imagen para analizar si hay presencia de drones")
uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    with st.spinner("🔍 Analizando imagen..."):
        model = load_model()
        if model:
            results = model(image_np, imgsz=640)
            result = results[0]
            boxes = result.boxes

            xyxy = boxes.xyxy.numpy()
            conf = boxes.conf.numpy()
            cls = boxes.cls.numpy()
            class_names = [result.names[int(c)] for c in cls]

            df = pd.DataFrame({
                'Clase': class_names,
                'Confianza': np.round(conf, 2),
                'Xmin': xyxy[:, 0].astype(int),
                'Ymin': xyxy[:, 1].astype(int),
                'Xmax': xyxy[:, 2].astype(int),
                'Ymax': xyxy[:, 3].astype(int)
            })

            st.markdown("## 📊 Resultados de la Detección")
            if len(df) > 0:
                st.success("✅ Se han detectado objetos en la imagen.")
                st.dataframe(df, use_container_width=True)

                annotated_img = result.plot()

                st.markdown("### 🖼 Comparativa: Original vs. Detectado")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Imagen Original", use_container_width=True)
                with col2:
                    st.image(annotated_img, caption="Detección YOLOv8", use_container_width=True)
            else:
                st.warning("⚠️ No se han detectado drones en la imagen.")
                
                # Mostrar la imagen original incluso si no se detectaron drones
                st.markdown("### 🖼 Comparativa: Original vs. Detectado")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Imagen Original", use_container_width=True)
                with col2:
                    st.image(image, caption="Imagen sin Detección", use_container_width=True)
