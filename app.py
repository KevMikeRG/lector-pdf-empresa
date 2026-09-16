import streamlit as st
import cv2
import numpy as np
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image

# Configuración de la página limpia y enfocada
st.set_page_config(page_title="Lector de Texto PDF - Customer Service", layout="centered")

st.title("📄 Extractor Directo de Texto")
st.markdown("Arrastra el PDF del cliente. El sistema extraerá y pulirá el texto de forma inmediata, sin previsualizaciones innecesarias.")

# Widget de carga limpio
archivo_pdf = st.file_uploader("Sube o arrastra tu PDF aquí", type=["pdf"])

if archivo_pdf is not None:
    bytes_pdf = archivo_pdf.read()
    
    with st.spinner('Procesando documento y optimizando texto...'):
        # Convertir todas las páginas del PDF a alta resolución
        imagenes = convert_from_bytes(bytes_pdf, dpi=300)
        texto_completo = ""
        
        for index, imagen_pil in enumerate(imagenes):
            img_cv = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
            gris = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            denoised = cv2.fastNlMeansDenoising(gris, h=30)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            contraste = clahe.apply(denoised)
            _, binaria = cv2.threshold(contraste, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            custom_config = r'--oem 1 --psm 3'
            texto_pagina = pytesseract.image_to_string(binaria, lang='spa', config=custom_config)
            texto_completo += f"\n--- PÁGINA {index + 1} ---\n{texto_pagina}\n"

    if not texto_completo.strip():
        texto_completo = "No se pudo extraer texto claro. El documento podría ser ilegible."

    st.success(f"¡Texto extraído con éxito de {len(imagenes)} página(s)!")
    
    # MOSTRAR DIRECTAMENTE EL TEXTO EN GRANDE (Sin mostrar PDFs ni imágenes)
    st.subheader("📝 Texto Extraído y Pulido")
    texto_final = st.text_area("Haz clic dentro, selecciona y copia (Ctrl+C):", 
                               value=texto_completo, 
                               height=400)
    
    # Botón de descarga opcional
    st.download_button(
        label="📥 Descargar como Archivo .txt",
        data=texto_final,
        file_name="texto_extraido.txt",
        mime="text/plain"
    )
