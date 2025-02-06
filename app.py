
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
# Cargar el modelo 
model = load_model('fashion_mnist.keras')


# Crear intefaz de usuario

st.title('Clasificador de ropa')
st.write('Sube una imagen para clasificaral como una categoría de ropa.')
#subir imagen
uploaded_file = st.file_uploader("Sube una imagen en escala de grises de 28x28 píxeles.", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
  #Vamos a procesar la imagen
  image=Image.open(uploaded_file).convert('L') #con convert('L') se convierte de rgb a blanco y negro
  image=image.resize((28,28)) # con esto redimencionamos la imagen
  img_array=np.array(image) # con  numpy convertimos la iamgen en un array para posterirormente normalizarla
  img_array=img_array/255.0 # normalizamos la imagen

  # la primera posici indiaca que solo hay una imagen los siguietes dos las dimesciones de la imagen y 
  # el ultimo digito que solo hay un canal de color (blanco y negro)
  img_array = img_array.reshape(1,28,28,1)

  # Mostrar la imagen subida

  st.image(image,caption='Imagen subida',use_column_width=True)

  # prediccion
  prediction=model.predict(img_array)

  classes = ['Camiseta/top', 'Pantalón', 'Jersey', 'Vestido', 'Abrigo', 'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Bota']

  st.write('Predicción:', classes[np.argmax(prediction)])

  # Mostrar probabilidades
  for i, prob in enumerate(prediction[0]):
        st.write(f"{classes[i]}: {prob:.2%}")

    # Clase con mayor probabilidad
   st.write("Predicción:", classes[np.argmax(prediction)])
