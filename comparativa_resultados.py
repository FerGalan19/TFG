import matplotlib.pyplot as plt
import numpy as np

# Datos proporcionados
modelos = ['MODELO 1', 'MODELO 2', 'MODELO 3', 'MODELO 4', 'MODELO 5']
entrenamiento = [99.52, 99.35, 99.39, 99.71, 99.56]
validacion = [99.48, 99.42, 99.39, 99.60, 99.38]
test = [97.37, 97.80, 97.53, 96.77, 97.39]
test_error = [0.26, 0.17, 0.69, 0.48, 0.45]

x = np.arange(len(modelos))  # el rango de modelos
width = 0.3  # el ancho de las barras

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# Color verde claro para entrenamiento y azul claro para validación
color_entrenamiento = 'lightgreen'
color_validacion = 'lightblue'

# Gráfica de Entrenamiento y Validación
rects1 = ax1.bar(x - width/2, entrenamiento, width, label='Entrenamiento', color=color_entrenamiento)
rects2 = ax1.bar(x + width/2, validacion, width, label='Validación', color=color_validacion)

# Establecer la escala del eje y entre 90 y 100
ax1.set_ylim(98, 100)

# Añadir texto para las etiquetas, título y leyenda
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Accuracy de Entrenamiento y Validación por Modelo')
ax1.set_xticks(x)
ax1.set_xticklabels(modelos)
ax1.legend(loc='lower right')

# Añadir las etiquetas encima de las barras
def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 puntos de desplazamiento vertical
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1, ax1)
autolabel(rects2, ax1)

# Gráfica de Test
rects3 = ax2.bar(x, test, width, yerr=test_error, label='Test', capsize=5, color=color_entrenamiento)

# Establecer la escala del eje y entre 90 y 100
ax2.set_ylim(94, 100)

# Añadir texto para las etiquetas, título y leyenda
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Accuracy de Test por Modelo')
ax2.set_xticks(x)
ax2.set_xticklabels(modelos)
ax2.legend(loc='lower right')

# Ajuste para que las etiquetas de las barras de test estén justo por encima de las barras
def autolabel_above(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height + test_error[rects.index(rect)]),
                    xytext=(0, 5),  # 5 puntos de desplazamiento vertical para que quede arriba del error bar
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10)

autolabel_above(rects3, ax2)

fig.tight_layout()

# Guardar la gráfica en un archivo
plt.savefig('grafica_resultados.png')

plt.show()
