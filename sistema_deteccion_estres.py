import tensorflow as tf
import numpy as np
import os

# Forzar a TensorFlow a usar la CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Cargar el modelo guardado
modelo_cargado = tf.keras.models.load_model(os.path.join('modelos', 'modelo_final_5.h5'))

# Ejemplo de predicción con una nueva imagen
def cargar_y_preprocesar_imagen(file_path):
    image = np.load(file_path, allow_pickle=True)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)  # Añadir una dimensión para el batch
    return image

# Lista de rutas de archivos y sus etiquetas verdaderas
imagenes_y_etiquetas = [
    ("DATASET/39/Native_Language_Interview/20191209_143012_901.npy", "BAJO ESTRES"),
    ("DATASET/40/Native_Language_Script_Reading/20191209_161228_302.npy", "ESTRES NEUTRAL"),
    ("DATASET/41/Non-native_Language_Interview/20191209_185806_246.npy", "ALTO ESTRES"),
    ("DATASET/42/Non-native_Language_Script_Reading/20191209_203019_446.npy", "ESTRES NEUTRAL"),
    ("DATASET/43/Native_Language_Interview/20191210_102915_896.npy", "BAJO ESTRES"),
    ("DATASET/44/Native_Language_Script_Reading/20191210_122051_855.npy", "ESTRES NEUTRAL"),
    ("DATASET/45/Non-native_Language_Interview/20191210_145907_014.npy", "ALTO ESTRES"),
    ("DATASET/46/Non-native_Language_Script_Reading/20191210_164850_630.npy", "ESTRES NEUTRAL"),
    ("DATASET/47/Native_Language_Interview/20191210_182957_173.npy", "BAJO ESTRES"),
    ("DATASET/48/Non-native_Language_Script_Reading/20191210_204953_026.npy", "ESTRES NEUTRAL"),
    ("DATASET/01/Native_Language_Interview/20191129_105442_175.npy", "BAJO ESTRES"),
    ("DATASET/02/Native_Language_Script_Reading/20191129_123515_606.npy", "ESTRES NEUTRAL"),
    ("DATASET/03/Non-native_Language_Interview/20191129_153203_981.npy", "ALTO ESTRES"),
    ("DATASET/04/Non-native_Language_Script_Reading/20191129_170929_253.npy", "ESTRES NEUTRAL"),
    ("DATASET/05/Native_Language_Interview/20191129_183247_629.npy", "BAJO ESTRES"),
    ("DATASET/05/Native_Language_Interview/20191129_183247_713.npy", "BAJO ESTRES"),
    ("DATASET/05/Non-native_Language_Interview/20191129_190241_149.npy", "ALTO ESTRES"),
    ("DATASET/07/Non-native_Language_Interview/20191202_111233_377.npy", "ALTO ESTRES"),
    ("DATASET/07/Non-native_Language_Interview/20191202_111233_334.npy", "ALTO ESTRES"),
    ("DATASET/07/Non-native_Language_Script_Reading/20191202_110134_634.npy", "ESTRES NEUTRAL"),    
]
# Diccionario de etiquetas
labels = ['ESTRES NEUTRAL', 'BAJO ESTRES', 'ALTO ESTRES']

# Verificar las predicciones para cada imagen
for file_path, true_label in imagenes_y_etiquetas:
    # Cargar y preprocesar la imagen
    image = cargar_y_preprocesar_imagen(file_path)
    
    # Realizar la predicción
    prediction = modelo_cargado.predict(image)
    predicted_label = labels[np.argmax(prediction)]
    
    # Comparar la predicción con la etiqueta verdadera
    if predicted_label == true_label:
        print(f"Imagen: {file_path} - Predicción Correcta: {predicted_label}")
    else:
        print(f"Imagen: {file_path} - Predicción Incorrecta: {predicted_label} (Verdadera: {true_label})")