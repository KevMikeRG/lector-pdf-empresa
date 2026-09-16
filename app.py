import streamlit as st
import cv2
import numpy as np
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image

st.set_page_config(page_title="Lector de Texto PDF - Customer Service", layout="centered")

# CSS para ampliar la zona de arrastre y evitar que el navegador abra el PDF
st.markdown("""
    <style>
    /* Ocultar elementos innecesarios de Streamlit para que sea minimalistra */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Ampliar la caja de carga para que ocupe todo el ancho y sea fácil de atinar */
    [data-testid="stFileUploader"] {
        width: 100%;
    }
    [data-testid="stFileUploader"] section {
        background-color: #f8fafc;
        border: 3px dashed #3b82f6;
        border-radius: 12px;
        padding: 40px 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📄 Extractor Directo de Texto")
st.markdown("Arrastra tu PDF en el recuadro de abajo:")

# Widget de carga limpio
archivo_pdf = st.file_uploader("", type=["pdf"])

if archivo_pdf is not None:
    bytes_pdf = archivo_pdf.read()
    
    with st.spinner('Procesando documento...'):
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
    
    # MOSTRAR DIRECTAMENTE EL TEXTO EN GRANDE
    texto_final = st.text_area("Haz clic dentro, selecciona y copia (Ctrl+C):", 
                               value=texto_completo, 
                               height=320)
    
    st.download_button(
        label="📥 Descargar como .txt",
        data=texto_final,
        file_name="texto_extraido.txt",
        mime="text/plain"
    )
