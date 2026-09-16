import streamlit as st
import cv2
import numpy as np
from pdf2image import convert_from_bytes
from paddleocr import PaddleOCR
from PIL import Image
import io

st.set_page_config(page_title="Lector Inteligente PDF - Customer Service", layout="wide")

@st.cache_resource
def iniciar_motor_ocr():
  # Usando la versión estable 2.7.0.3
  return PaddleOCR(use_angle_cls=True, lang='es', show_log=False)

ocr_engine = iniciar_motor_ocr()

def mejorar_imagen_para_ocr(imagen_pil):
  img_cv = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
  gris = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  contraste_mejorado = clahe.apply(gris)
  suavizado = cv2.GaussianBlur(contraste_mejorado, (3, 3), 0)
  binaria = cv2.adaptiveThreshold(suavizado, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
  # Reconvertir a 3 canales para evitar el TypeError
  imagen_lista_para_ia = cv2.cvtColor(binaria, cv2.COLOR_GRAY2BGR)
  return imagen_lista_para_ia

st.title("📄 Procesador Inteligente de Documentos")
st.markdown("Arrastra un PDF borroso, invertido o con texto manuscrito. El sistema lo limpiará y digitalizará automáticamente.")

archivo_pdf = st.file_uploader("Sube el PDF del cliente aquí", type=["pdf"])

if archivo_pdf is not None:
  bytes_pdf = archivo_pdf.read()
  st.info("Procesando documento... Por favor espera unos segundos.")
  
  imagenes = convert_from_bytes(bytes_pdf, dpi=300)
  imagen_original = imagenes[0]
  
  with st.spinner('Ejecutando limpieza y reconocimiento de texto (OCR/HTR)...'):
      img_limpia_cv = mejorar_imagen_para_ocr(imagen_original)
      resultados = ocr_engine.ocr(img_limpia_cv, cls=True)
      
      texto_extraido = ""
      # Validación de seguridad para evitar errores si la página está en blanco
      if resultados is not None and len(resultados) > 0 and resultados[0] is not None:
          for linea in resultados[0]:
              texto_extraido += linea[1][0] + "\n"
      else:
          texto_extraido = "No se detectó texto en el documento."

  st.divider()
  col_izq, col_der = st.columns(2)

  with col_izq:
      st.subheader("🖼️ Documento Original")
      st.image(imagen_original, use_column_width=True)

  with col_der:
      st.subheader("📝 Texto Transcrito (Digitalizado)")
      texto_final = st.text_area("Selecciona, copia y pega en el sistema interno:", value=texto_extraido, height=500)
      st.download_button(label="Descargar Texto (.txt)", data=texto_final, file_name="texto_digitalizado.txt", mime="text/plain")
