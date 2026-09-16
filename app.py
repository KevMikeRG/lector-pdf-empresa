import streamlit as st
import cv2
import numpy as np
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image

# Configuración de la página
st.set_page_config(page_title="Lector Inteligente PDF - Customer Service", layout="wide")

# Detectar si estamos en "Modo Widget" mediante la URL (ej: ?view=compact)
query_params = st.query_params
modo_compacto = query_params.get("view") == "compact"

if not modo_compacto:
    # VISTA COMPLETA (La que ya conoces)
    st.title("📄 Procesador Avanzado de Documentos - Customer Service")
    st.markdown("Sube el PDF del cliente (soporta múltiples páginas) para extraer y optimizar el texto.")
else:
    # VISTA COMPACTA (Solo la caja para incrustar en tu programa)
    st.markdown("### 📥 Carga Rápida de PDF")

# Widget de carga de archivos común para ambos modos
archivo_pdf = st.file_uploader("Arrastra o selecciona el PDF aquí", type=["pdf"], key="uploader_principal")

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

    if modo_compacto:
        # Si se usó desde el widget compacto, mostramos un botón para abrir el análisis completo en grande
        st.success("¡PDF procesado con éxito!")
        # Guardamos el texto temporalmente en session_state o abrimos la vista
        st.text_area("Texto Extraído Rápido:", value=texto_completo, height=150)
        st.markdown(f"[🔍 Abrir Analizador Completo en Pantalla Grande]({st.context.headers.get('Host', '')}/?view=full)", unsafe_allow_html=True)
    else:
        # VISTA COMPLETA DE ANÁLISIS (Split View)
        st.success(f"¡Proceso completado! Se analizaron {len(imagenes)} página(s).")
        st.divider()
        col_izq, col_der = st.columns(2)

        with col_izq:
            st.subheader("🖼️ Documento Original (Página 1)")
            st.image(imagenes[0], use_container_width=True)

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
