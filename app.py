import streamlit as st
import cv2
import numpy as np
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image

# Configuración de la página
st.set_page_config(page_title="Lector Inteligente PDF - Customer Service", layout="wide")

st.title("📄 Procesador Avanzado de Documentos - Customer Service")
st.markdown("Sube el PDF del cliente (soporta múltiples páginas). El sistema aplicará filtros ópticos avanzados para maximizar la precisión de lectura.")

archivo_pdf = st.file_uploader("Arrastra o selecciona el PDF aquí", type=["pdf"])

if archivo_pdf is not None:
    bytes_pdf = archivo_pdf.read()
    
    with st.spinner('Optimizando imágenes y extrayendo texto con máxima precisión...'):
        # Convertir todas las páginas del PDF a alta resolución (300 DPI)
        imagenes = convert_from_bytes(bytes_pdf, dpi=300)
        texto_completo = ""
        
        # Iterar página por página (ideal para documentos de 2 a 10 páginas)
        for index, imagen_pil in enumerate(imagenes):
            # 1. Convertir imagen PIL a formato OpenCV (BGR)
            img_cv = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
            gris = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # 2. Reducción de ruido avanzada para eliminar manchas de escáner y puntos negros
            denoised = cv2.fastNlMeansDenoising(gris, h=30)
            
            # 3. Aumento de contraste inteligente (CLAHE) para letras tenues o borrosas
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            contraste = clahe.apply(denoised)
            
            # 4. Binarización adaptativa Otsu para separar perfectamente el texto del fondo
            _, binaria = cv2.threshold(contraste, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 5. Configuración avanzada de Tesseract:
            # --oem 1: Usa el motor de red neuronal (LSTM) para mayor precisión
            # --psm 3: Segmentación automática de página completa
            custom_config = r'--oem 1 --psm 3'
            texto_pagina = pytesseract.image_to_string(binaria, lang='spa', config=custom_config)
            
            # Acumular el texto ordenado por páginas
            texto_completo += f"\n--- PÁGINA {index + 1} ---\n{texto_pagina}\n"

    # Validación si el documento está totalmente en blanco o ilegible
    if not texto_completo.strip():
        texto_completo = "No se pudo extraer texto claro. El documento podría ser una imagen completamente ilegible o un gráfico sin texto reconocible."

    st.success(f"¡Proceso completado con éxito! Se analizaron {len(imagenes)} página(s).")

    # Interfaz de Vista Dividida (Split View)
    st.divider()
    col_izq, col_der = st.columns(2)

    with col_izq:
        st.subheader("🖼️ Documento Original (Página 1)")
        st.image(imagenes[0], use_container_width=True)
        if len(imagenes) > 1:
            st.info(f"Nota informativa: El documento cargado contiene {len(imagenes)} páginas en total.")

    with col_der:
        st.subheader("📝 Texto Extraído y Pulido")
        texto_final = st.text_area("Revisa, selecciona y copia al sistema interno:", 
                                   value=texto_completo, 
                                   height=500)
        
        st.download_button(
            label="Descargar Texto Completo (.txt)",
            data=texto_final,
            file_name="texto_extraido_customer_service.txt",
            mime="text/plain"
        )
