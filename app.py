import streamlit as st
import cv2
import numpy as np
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image

st.set_page_config(page_title="Lector PDF - Customer Service", layout="wide")

st.title("📄 Procesador de Documentos - Customer Service")
st.markdown("Arrastra el PDF del cliente. El sistema optimizará la imagen y extraerá el texto automáticamente.")

archivo_pdf = st.file_uploader("Sube el PDF aquí", type=["pdf"])

if archivo_pdf is not None:
    bytes_pdf = archivo_pdf.read()
    st.info("Procesando documento... Un momento por favor.")
    
    # Convertir la primera página del PDF a imagen de alta calidad
    imagenes = convert_from_bytes(bytes_pdf, dpi=300)
    imagen_original = imagenes[0]
    
    # Preprocesamiento con OpenCV (Mejora drástica para escaneos y fotos)
    img_cv = cv2.cvtColor(np.array(imagen_original), cv2.COLOR_RGB2BGR)
    gris = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Contraste adaptativo (CLAHE) y binarización (OTSU) para limpiar fondos oscuros o sucios
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contraste = clahe.apply(gris)
    _, binaria = cv2.threshold(contraste, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Ejecutar Tesseract OCR en español
    with st.spinner('Extrayendo texto con Tesseract OCR...'):
        texto_extraido = pytesseract.image_to_string(binaria, lang='spa')
        
        if not texto_extraido.strip():
            texto_extraido = "No se pudo reconocer texto claro en el documento. Revisa la imagen original."

    # Interfaz de Vista Dividida (Split View)
    st.divider()
    col_izq, col_der = st.columns(2)

    with col_izq:
        st.subheader("🖼️ Documento Original")
        st.image(imagen_original, use_container_width=True)

    with col_der:
        st.subheader("📝 Texto Extraído (Listo para copiar)")
        texto_final = st.text_area("Selecciona, copia y pega en el sistema interno:", 
                                   value=texto_extraido, 
                                   height=500)
        
        st.download_button(
            label="Descargar Texto (.txt)",
            data=texto_final,
            file_name="texto_extraido.txt",
            mime="text/plain"
        )
