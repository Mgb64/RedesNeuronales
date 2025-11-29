# Transfer Learning para la Clasificación de Flores con EfficientNet

**TL;DR:** Se entrenó un modelo para clasificar **104 especies de flores** utilizando
**EfficientNetB0 pre-entrenada en ImageNet**. El dataset se proporcionó en formato
**TFRecords**, aplicando *aumentación de datos* y la estrategia de *feature extraction*
(congelando la base convolucional). El modelo alcanzó una **accuracy de validación de
`[Faltante]`**, con un entrenamiento estable y convergente en solo 10 épocas.

---

## 1. Problema y contexto

Este proyecto aborda un problema de **clasificación de imágenes de flores** (*fine-grained
image classification*), en el que se busca distinguir entre **104 especies
distintas**. Muchas de ellas presentan diferencias visuales mínimas, lo cual dificulta la
tarea. Entrenar un modelo desde cero implicaría un alto costo computacional y un riesgo
considerable de sobreajuste, por lo que se emplea **Transfer Learning** para aprovechar
las representaciones visuales previamente aprendidas por EfficientNet en ImageNet (bordes,
texturas, patrones de color, etc.).

**¿Por qué Transfer Learning en este caso?**  Porque reduce drásticamente el tiempo de
entrenamiento, requiere menos datos y ofrece una mejora significativa en tareas con clases
altamente similares entre sí.

---

## 2. Datos

Se utilizó el dataset [**Flower Classification with
TPUs**](https://www.kaggle.com/competitions/flower-classification-with-tpus) de Kaggle, el
cual ya incluye las imágenes divididas en subconjuntos de `Train`, `Validation` y `Test`,
almacenadas en formato **TFRecords** y con resolución homogénea de 224×224 píxeles.

### Preprocesamiento

Las imágenes se decodifican desde TFRecords a tensores `float32` y se normalizan siguiendo
la convención utilizada por EfficientNet:

```
x_norm = x / 127.5 - 1
```

### Aumentación de Datos

Con el objetivo de mejorar la capacidad de generalización del modelo, se aplicó *data
augmentation* exclusivamente al conjunto de entrenamiento. Las transformaciones empleadas
fueron:

* **`RandomFlip("horizontal")`**: inversión horizontal de la imagen.
* **`RandomRotation(0.2")`**: rotación aleatoria de hasta un 20%.
* **`RandomZoom(0.2")`**: zoom aleatorio del 20%.
* **`RandomContrast(0.1")`**: variación del contraste en un 10%.

Estas técnicas permiten aumentar la variabilidad del conjunto de datos sin necesidad de
recolectar más imágenes.

---

## 3. Modelo y estrategia de Transfer Learning

Se empleó **EfficientNetB0**, cargando sus pesos pre-entrenados en **ImageNet**,
únicamente como extractor de características. Para ello, se congelaron sus pesos y se
añadió una cabeza clasificadora adaptada a las **104 clases**, utilizando una función de
activación `softmax`.  Además, se incorporaron capas de **dropout** para reducir el riesgo
de sobreajuste.

Código del modelo:

```python
from tensorflow.keras import layers, models
import tensorflow as tf

NUM_CLASSES = 104

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax')
])
```

---

## 4. Entrenamiento

El entrenamiento del modelo se realizó en **dos fases**: una fase inicial de *transfer
learning* y una posterior de *fine-tuning* descogelando las ultimas 20 capas del modelo
efficientnet, con el objetivo de que el modelo aprendiera progresivamente las
características más relevantes de las imágenes.

Los **hiperparámetros** utilizados en la fase de transferencia fueron:

* **Optimizador:** Adam
* **Función de pérdida:** Sparse Categorical Crossentropy
* **Learning rate:** 0.001
* **Batch size:** 32
* **Número de épocas:** 20

Para garantizar la **reproducibilidad**, se fijó la semilla de la siguiente manera:

```python
import numpy as np
import tensorflow as tf

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

Además, se implementó **early stopping** para evitar sobreajuste.

En la fase de *fine-tuning*, se emplearon los mismos parámetros, con la excepción de:

* **Learning rate:** 0.0001
* **Número de épocas:** 10

Esto permitió ajustar ligeramente los pesos del modelo preentrenado sin perder las
características aprendidas de ImageNet.

Debido al **desbalance de clases**, se aplicaron **pesos balanceados** durante el
entrenamiento. En general, el modelo mostró un comportamiento estable y consistente en
términos de *loss* y *accuracy*, gracias al uso de **regularización** y **aumentación de
datos**.

![Gráficas de entrenamiento](imagenes/Graficas.png)

Se observa que la **accuracy de validación supera a la de entrenamiento**, mientras que la
**pérdida en validación es ligeramente menor** que la de entrenamiento. Esto es esperable
debido a las técnicas de regularización aplicadas. Lo importante es que tanto la pérdida
como la accuracy muestran una evolución coherente en ambos conjuntos: la pérdida disminuye
y la accuracy aumenta de manera consistente.

---

## 5. Resultados y evaluación

El modelo se evaluó únicamente en el conjunto de validación, ya que los datos de prueba no
incluyen etiquetas. Los resultados fueron:

* **Accuracy:** 83.94%
* **Precisión:** 85.04%
* **Recall:** 83.94%
* **F1-score:** 83.87%

Cada métrica se calculó teniendo en cuenta la proporción de cada clase respecto al total,
lo que permite una evaluación más equitativa en un conjunto con múltiples categorías.

Pero el modelo al intentar predecir en el conjunto de test se veía sesgado por una 
clase, prediciendo siempre la misma.

---

## 6. Probando el modelo con una aplicación de celular.

Para utilizar el modelo, aprovecharemos el uso de una página web con acceso a la camara
para clasificar diferentes flores. La idea es activar un puerto local en nuestra
computadora con python y junto con ngrok crear un enlace web para enviarlo a nuestro
celular. Las instrucciones específicas para usar la aplicación estarán en el `README` del
repositorio.

---

## 7. Lecciones aprendidas y limitaciones

**Aspectos positivos:**

* EfficientNet ofrece un excelente rendimiento con bajo costo computacional.
* El uso de TFRecords y AUTOTUNE mejora notablemente la eficiencia en la carga de datos.
* Las técnicas de regularización y aumentación de datos reducen el riesgo de sobreajuste.
* El finetuning hace que el modelo no se estanque y aprenda aun más características del
  dataset.

**Limitaciones:**

* El dataset presenta cierto desbalance entre clases, lo cual sesgo al modelo a una sola clase.
* El modelo se limita a aprender patrones visuales y no integra información contextual.
* Es importante revisar la licencia del dataset y considerar posibles sesgos, como
  iluminación, especies sobre-representadas o fondos característicos.

---

## 8. Reproducibilidad

Para reproducir estos resultados clona el siguiente repositorio con este comando `git
clone https://github.com/Mgb64/RedesNeuronales.git`. Dentro de este repositorio en la
carpeta de `Medium`
encontraras lo siguiente:  
- La libreta con el cógigo que crea al modelo: `Modelo.ipynb`.
- El ambiente de conda `enviroment.yml` que contiene los paquetes necesarios para que la
  libreta funcione.
- Al final, se generara el archivo model.onxx que puede ser utilizado para inferencia.
---

# 9. Uso del Modelo

Una forma interesante de usar el modelo es creando una página web en la cual podamos
activar la cámara del celular, tomar fotos con él a diferentes flores y que el modelo en
la misma página nos diga la especie de la flor. En el repositorio encontrarás el archivo
index.html en el cual esta definida esta página. Para ejecutarlo en el celular deberás en
la misma carpeta en donde se envuentra index.html los comandos.
- `python -m http 8000`: Utiliza el puerto 8000 de tu computadora para la página
- `ngrok http 8000`: Vincula tu puerto 8000 con una dirección en el servidor de ellos

Cuando ejecutes el último comando te deberá de proporcionar un link. Con este puedes abrir
la página en el celular y usar la cámara trasera para tomar fotos a diferentes flores y
ver que predice el modelo.

##  Referencias

* Tan & Le (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural
  Networks*.
* Documentación oficial de TensorFlow / Keras.
* Dataset *Flower Classification with TPUs*, disponible en Kaggle.  * Scikit-learn:
métricas de evaluación. 

---
1. Reproducibilidad
   - [x] Notebook/Colab enlazado y ejecuta de inicio a fin sin errores.
   - [x] Seeds fijadas (`random`, `numpy`, `torch/tf`) y aviso del runtime (CPU/GPU,
         T4/V100/A100).
   - [x] `requirements.txt` / `environment.yml` o celda `pip install ...` clara.
   - [x] Script/notebook descarga y organiza datos (o explica cómo obtenerlos).

2. Exactitud técnica
   - [x] Indicas modelo pre-entrenado (paper, checkpoint) y qué capas
         congelas/descongelas.
   - [x] Justificas hiperparámetros clave (lr, batch, epochs).
   - [x] Métricas correctas para la tarea (accuracy, precision, recall, F1) y curvas.
   - [x] Incluyes sanity checks (formas, batch pequeño, overfit a pocas muestras).

3. Ética y licencias
   - [x] Citas dataset y modelo con enlaces y licencias.
   - [x] Declaras posibles sesgos y límites del modelo.
   - [ ] Código con licencia clara (MIT/Apache-2.0/BSD-3-Clause).

4. Presentación
   - [x] TL;DR inicial (3–4 líneas) y conclusiones/limitaciones claras.
   - [x] Imágenes con texto alternativo y gráficos legibles (títulos/leyendas).
   - [x] Ortografía y formato correcto.
   - [ ] Enlaces a repo/weights/demo funcionando.

5. Seguridad y privacidad
   - [x] No publicas claves/tokens/credenciales en el código o logs.
   - [x] No hay datos sensibles.


