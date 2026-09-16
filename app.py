import streamlit as st
import cv2
import numpy as np
from pdf2image import convert_from_bytes
from paddleocr import PaddleOCR
from PIL import Image
import io

# 1. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(page_title="Lector Inteligente PDF - Customer Service", layout="wide")

# 2. CARGA DEL MOTOR DE IA (Caché para que sea instantáneo tras el primer uso)
@st.cache_resource
def iniciar_motor_ocr():
    # use_angle_cls=True detecta y corrige automáticamente si el texto está al revés
    return PaddleOCR(use_angle_cls=True, lang='es', use_gpu=False, enable_mkldnn=False)

ocr_engine = iniciar_motor_ocr()

# 3. FUNCIONES DE PREPROCESAMIENTO (Visión por Computadora)
def mejorar_imagen_para_ocr(imagen_pil):
    """Aplica filtros para limpiar fondos borrosos y resaltar la tinta."""
    # Convertir a formato OpenCV
    img_cv = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
    
    # Convertir a escala de grises
    gris = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Aumentar contraste (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contraste_mejorado = clahe.apply(gris)
    
    # Suavizado ligero para eliminar ruido (polvo del escáner)
    suavizado = cv2.GaussianBlur(contraste_mejorado, (3, 3), 0)
    
    # Binarización adaptativa para separar texto del fondo
    binaria = cv2.adaptiveThreshold(suavizado, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)
                                    
    # LA SOLUCIÓN: Convertir la imagen de 1 canal de vuelta a 3 canales 
    # para que el motor de Inteligencia Artificial la acepte sin errores.
    imagen_lista_para_ia = cv2.cvtColor(binaria, cv2.COLOR_GRAY2BGR)
    
    return imagen_lista_para_ia

# 4. INTERFAZ DE USUARIO (Frontend)
st.title("📄 Procesador Inteligente de Documentos")
st.markdown("Arrastra un PDF borroso, invertido o con texto manuscrito. El sistema lo limpiará y digitalizará automáticamente.")

archivo_pdf = st.file_uploader("Sube el PDF del cliente aquí", type=["pdf"])

if archivo_pdf is not None:
    bytes_pdf = archivo_pdf.read()
    
    st.info("Procesando documento... Por favor espera unos segundos.")
    
    # Convertir la primera página del PDF a imagen (Alta resolución 300 DPI)
    # En producción se iteraría sobre todas las páginas
    imagenes = convert_from_bytes(bytes_pdf, dpi=300)
    imagen_original = imagenes[0]
    
    # Procesar imagen con IA
    with st.spinner('Ejecutando limpieza y reconocimiento de texto (OCR/HTR)...'):
        # Limpieza visual
        img_limpia_cv = mejorar_imagen_para_ocr(imagen_original)
        
        # Ejecutar Motor OCR de Paddle (Extrae coordenadas, texto y nivel de confianza)
        resultados = ocr_engine.ocr(img_limpia_cv)
        
        texto_extraido = ""
        if resultados != [None]:
            for linea in resultados[0]:
                texto_extraido += linea[1][0] + "\n"
        else:
            texto_extraido = "No se detectó texto en el documento."

    # 5. VISTA DIVIDIDA (SPLIT VIEW)
    st.divider()
    col_izq, col_der = st.columns(2)

    with col_izq:
        st.subheader("🖼️ Documento Original")
        st.image(imagen_original, use_column_width=True)

    with col_der:
        st.subheader("📝 Texto Transcrito (Digitalizado)")
        # Caja de texto editable por si el empleado necesita hacer una corrección menor
        texto_final = st.text_area("Selecciona, copia y pega en el sistema interno:", 
                                   value=texto_extraido, 
                                   height=500)
        
        # Generar un archivo .txt descargable como alternativa rápida
        st.download_button(
            label="Descargar Texto (.txt)",
            data=texto_final,
            file_name="texto_digitalizado.txt",
            mime="text/plain"
        )
