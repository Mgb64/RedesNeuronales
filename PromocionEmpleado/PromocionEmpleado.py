#!/usr/bin/env python
# coding: utf-8

# # Predecir la promoción de un empleado

# 
# 
# * Limpieza/preprocesamiento  
# * Clases desbalanceadas
# *   Keras-tuner
# 
# 
# 

# Dataset
# 
# https://www.kaggle.com/code/muhammadimran112233/employee-promotion-end-to-end-solution/input

# 
# incluir, al menos, lo siguiente
# 
# 
# * Manejo de datos faltantes
# * datos repetidos
# * columnas constantes
# * ¿hay outliers?
# * ¿selección de atributos?
# * estandarización de datos
# * Manejo de datos desbalanceados
# * Keras-tuner, usar conjunto de validación
#   * construir modelo con los mejores parámetros
#   * graficar loss y acc de train y validación
#   * evaluar con X_test
#   * obtener matriz de confusión

# # Importando Dataset y Librerias

# In[213]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split


# In[214]:


df = pd.read_csv('./employee_promotion.csv')


# # Análisis Exploratorio de datos
# 
# Tenemos las siguientes columnas
# 
# - **department**: Departamento del empleado  
# - **region**: Región de empleo (no ordenada)  
# - **education**: Nivel educativo  
# - **gender**: Género del empleado  
# - **recruitment_channel**: Canal de reclutamiento del empleado  
# - **no_of_trainings**: Número de capacitaciones adicionales completadas el año anterior (habilidades blandas, técnicas, etc.)  
# - **age**: Edad del empleado  
# - **previous_year_rating**: Calificación del empleado en el año anterior  
# - **length_of_service**: Antigüedad en años de servicio  
# - **awards_won**: Si ganó premios el año anterior → 1, de lo contrario 0  
# - **avg_training_score**: Puntaje promedio en evaluaciones de capacitaciones actuales  
# - **is_promoted**: **(Variable objetivo)** Si fue recomendado para promoción  
# 
# Como vemos hay varios preguntas que tenemos que hacernos de primera mano, ¿Necesitamos el número de empleado?, ¿Tener el genero es necesario, habra un cesgo en los datos hacia los hombres para hacer las promociones?

# In[215]:


df.shape


# In[216]:


df.info()


# Al parecer solo las columnas avg_training_score, education, previous_year_rating son las que tienen valores nulos, el porcentaje son minimo entonces tal vez sea buena idea imputarlas

# In[217]:


df.describe()


# In[218]:


df.head(10)


# Tenemos como columnas categóricas a:
# - department
# - region
# - educacion
# - genero
# - recruiment_chanel
# - awards_won
# 
# El resto son columnas numericas.

# In[219]:


# Convertir todas las columnas object a category
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype('category')

# Convertir awards_won a category
df['awards_won'] = df['awards_won'].astype('category')


# Vemos si hay filas repetidas

# In[220]:


df.duplicated().sum()


# Vamos a ver filas con valores nulos

# In[221]:


df.isnull().sum() / df.shape[0] * 100


# Vamos a ver si hay sesgo al momento de promocionar a alguien por el genero.

# In[222]:


values_counts = df[['gender']].value_counts()
number_men = values_counts['m']
number_women = values_counts['f']

labels = ['Mujeres', 'Hombres']

plt.pie([number_women, number_men], labels=labels, autopct='%1.1f%%', startangle=90)
plt.title('Distribución entre hombres y mujeres')
plt.show()





# Vemos que aproximadamente el 70% de las personas son hombres y 30% mujeres, vamos a ver si se conserva esta proporcion al momento de promocionar a alguien

# In[223]:


df_promoted = df[df['is_promoted'] == 1]
df_promoted.groupby('is_promoted')['gender'].value_counts()

number_men_promoted = len(df_promoted[df_promoted['gender'] == 'm'])
number_women_promoted = len(df_promoted[df_promoted['gender'] == 'f'])

labels = ['Mujeres', 'Hombres']

plt.pie([number_women_promoted, number_men_promoted], labels=labels, autopct='%1.1f%%', startangle=90)
plt.title('Distribución entre hombres y mujeres promovidos')
plt.show()


# Vemos que no hay un sesgo al momento de promover a un empleado por su genero

# Ahora vamos a ver la distribucion de nuestras variables

# In[224]:


for col in df.columns:
    plt.figure(figsize=(6,4))
    if df[col].dtype == 'object':
        sns.countplot(data=df, x=col)
        plt.title(f"Distribución de {col}")
    else:  # numéricas
        sns.histplot(data=df, x=col, kde=True, bins=20)
        plt.title(f"Distribución de {col}")
    plt.xticks(rotation=90)
    plt.show()


# Vemos que tenemos clases desbalanceadas, si nos fijamos en la variable "is_promoted"

# In[225]:


labels = ['Promovidos', 'No Promovidos']

promoted = len(df[df['is_promoted'] == 1])
no_promoted = df.shape[0] - promoted
plt.pie([promoted, no_promoted], labels=labels, autopct='%1.1f%%', startangle=90)
plt.title('Distribucion entre Promovidos entre No Promovidos')
plt.show()


# Ahora vamos a ver nuestros datos con respecto a nuestra variable objetiva

# In[226]:


target = "is_promoted"

for col in df.columns:
    if col == target:
        continue

    plt.figure(figsize=(6,4))

    if df[col].dtype == 'object':
        sns.countplot(data=df, x=col, hue=target)
        plt.title(f"Distribución de {col} por {target}")
    else:  # numéricas
        sns.histplot(data=df, x=col, hue=target, kde=True, bins=20, element="step")
        plt.title(f"Distribución de {col} por {target}")

    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


# Al parecer todas las variables tienen relación, lo único que quitaremos por el momento es el número de empleado que no nos sirve de nada para el modelo.

# In[227]:


df.drop('employee_id', axis=1, inplace=True)


# Vamos a ver que columnas tienen outliers

# In[228]:


numeric_cols = list(df.select_dtypes(include="number"))
numeric_cols.remove("is_promoted")

prop_outliers = {}
for c in numeric_cols:
  q1 = df[c].dropna().quantile(.25)
  q3 = df[c].dropna().quantile(.75)
  IQR = q3 - q1

  lower = q1 - 1.5*IQR
  high = q3 + 1.5*IQR

  prop_outliers[c] = df[(df[c] < lower) | (df[c] > high)].shape[0] / df.shape[0] * 100

for k,v in prop_outliers.items():
  print(f"La columna {k} tiene {v}% de outliers")


# In[229]:


for c in numeric_cols:
  sns.boxplot(data=df, x=c)
  plt.title(f"Outliers de la columna {c}")
  plt.show()


# Vemos que si hay outliers, pero no son por errores de medición, ahora vamos a ver si hay sesgo.

# In[230]:


skews = {}
for c in numeric_cols:
  skews[c] = df[c].skew()

for k,v in skews.items():
  print(f"la columna {k} tiene un sesgo de {v}")


# Vemos que son algunas columnas las que tienen  sesgo, entonces tambien hay que corregirlo.

# ## Dividiendo el dataset

# In[231]:


X = df.drop('is_promoted', axis=1)
y = df[['is_promoted']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=42)


# ## Preprocesamiento de Datos

# ### Imputando datos
# 
# Vamos a imputar los datos de las columnas avg_training_score, education, previous_year_rating que eran las que tenian valores faltantes.

# In[232]:


median = X_train['previous_year_rating'].median()
X_train['previous_year_rating'].fillna(median, inplace=True)
X_test['previous_year_rating'].fillna(median, inplace=True)
X_test['previous_year_rating'].fillna(median, inplace=True)

mean = X_train['avg_training_score'].mean()
X_train['avg_training_score'].fillna(mean, inplace=True)
X_test['avg_training_score'].fillna(mean, inplace=True)
X_val['avg_training_score'].fillna(mean, inplace=True)

mode = X_train['education'].mode()[0]
X_train['education'].fillna(mode, inplace=True)
X_test['education'].fillna(mode, inplace=True)
X_val['education'].fillna(mode, inplace=True)


# Vamos a ver si podemos eliminar columnas que no nos sean tan utiles. Vamos a ver por correlaciones en columnas.

# In[233]:


# Correlaciones con el target

Xy_train = pd.concat([X_train, y_train], axis=1)
Xy_train_num = Xy_train.select_dtypes(include="number")

corrs = Xy_train_num.corr(method="pearson")["is_promoted"].sort_values(ascending=False)

print(corrs)


# Vemos que no cambia mucho las variables relacionadas con los que estan promovidos

# Las que mas tienen correlacion con el target son avg_training_score y previous_year_rating.

# In[234]:


corr_matrix = Xy_train_num.corr(method="pearson")

plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Matriz de correlación (Pearson)")
plt.show()


# In[235]:


from scipy.stats import chi2_contingency

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k-1, r-1))

X_cat = X_train.select_dtypes(include='category')  

for col in X_cat.columns:
    v = cramers_v(X_cat[col], y['is_promoted'])
    print(f"Cramér's V ({col} vs is_promoted): {v:.4f}")


# Vemos que la única que tiene relacion con el target es awars_won, vamos a deshacernos de las demás.

# In[236]:


to_drop = list(X_cat.columns)
to_drop.remove("awards_won")

for df in [X_train, X_val, X_test]:
    df.drop(to_drop, axis=1, inplace=True)


# Las unicas de las varibales numéricas que vamos a conservar es avg_training_score y previous_year_rating que son las que mas relacion tienen con la variable objetivo.

# In[237]:


to_drop = list(X_train.select_dtypes(include='number').columns)
to_drop.remove("avg_training_score")
to_drop.remove("previous_year_rating")

for df in [X_train, X_val, X_test]:
    df.drop(to_drop, axis=1, inplace=True)


# In[238]:


X_train.shape


# ### Tratando Outliers
# 
# Vimos que es poco probable que los outliers se deban a errores, haremos winsorizacion.

# In[239]:


columns_with_outliers = [
    "previous_year_rating",
]

for c in columns_with_outliers:
    lower = X_train[c].quantile(0.01)
    upper = X_train[c].quantile(0.99)
    X_train[c + "_wins"] = X_train[c].clip(lower, upper)
    X_test[c + "_wins"] = X_test[c].clip(lower, upper)
    X_val[c + "_wins"] = X_val[c].clip(lower, upper)


# Veamos si se conserva el sesgo en nuestras variables

# In[240]:


skews = {}
for c in X_train.select_dtypes(include='number').columns:
  skews[c] = X_train[c].skew()

for k,v in skews.items():
  print(f"la columna {k} tiene un sesgo de {v}")


# Vemos que no hay sesgo en nuestras variables.

# ### Escalando las variables numéricas

# In[241]:


from sklearn.preprocessing import StandardScaler, RobustScaler

numerical_cols = list(X_train.select_dtypes(include='number'))

for c in numerical_cols:
  ss = StandardScaler()
  X_train[c + "_ss"] = ss.fit_transform(X_train[[c]])
  X_test[c + "_ss"] = ss.transform(X_test[[c]])
  X_val[c + "_ss"] = ss.transform(X_val[[c]])


# Nuestra unica variable categorica es la de si gano un premio el año pasado y ya esta codificada desde el principio, solo vamos a cambiarle el tipo a numerico

# In[242]:


for X in [X_train, X_test, X_val]:
    X['awards_won'] = X['awards_won'].astype('int')


# In[243]:


X_train.columns


# In[244]:


cols_finales = [c for c in X_train.columns if c.endswith('_ss') or c.startswith('awards_won')]

print(sorted(cols_finales))


# In[245]:


# -------------------------------------------------
# 1. Convertir y_train a Serie (si no lo es)
# -------------------------------------------------
y_train_series = y_train['is_promoted']

# -------------------------------------------------
# 2. Copiar las columnas que vamos a usar
# -------------------------------------------------
X_train_final = X_train[cols_finales].copy()
X_val_final = X_val[cols_finales].copy()
X_test_final = X_test[cols_finales].copy()

# -------------------------------------------------
# 3. Calcular target encoding correctamente
# -------------------------------------------------
# Media de is_promoted por valor de awards_won en train
target_mean_dict = y_train_series.groupby(X_train_final['awards_won']).mean().to_dict()

# Aplicar target encoding
X_train_final['awards_won_te'] = X_train_final['awards_won'].map(target_mean_dict)
X_val_final['awards_won_te'] = X_val_final['awards_won'].map(target_mean_dict).fillna(y_train_series.mean())
X_test_final['awards_won_te'] = X_test_final['awards_won'].map(target_mean_dict).fillna(y_train_series.mean())

# -------------------------------------------------
# 4. Eliminar la columna original awards_won
# -------------------------------------------------
X_train_final = X_train_final.drop(columns=['awards_won'])
X_val_final = X_val_final.drop(columns=['awards_won'])
X_test_final = X_test_final.drop(columns=['awards_won'])







X_train_final.columns


# In[246]:


X_train_final.info()


# In[247]:


y_train.info()


# ### OverSampling
# 
# Como tenemos clases desbalanceadas tenemos que corregir esto.

# #### SMOTE

# In[248]:


# Antes de Smote

y_train.value_counts()


# In[249]:


from imblearn.over_sampling import SMOTE  #variante de SMOTE

smote = SMOTE(sampling_strategy=1/3, random_state=42)
X_SMOTE, y_SMOTE = smote.fit_resample(X_train_final, y_train)


# In[250]:


# Despues de SMOTE
y_SMOTE.value_counts()


# ### Under Sampling

# #### ClusterCentroids

# In[251]:


# Antes de Under Sampling

y_train.value_counts()


# In[252]:


from imblearn.under_sampling import ClusterCentroids
ncr = ClusterCentroids()
X_CC, y_CC = ncr.fit_resample(X_train_final, y_train)


# In[253]:


y_CC.value_counts()


# ## Modelo

# A continuacion crearemos la plantilla de nuestro modelo

# In[254]:


get_ipython().system('pip install -q keras-tuner --upgrade')


# In[255]:


from keras.models import Model
from keras.layers import Dense, Input, Dropout
import keras_tuner as kt

import keras
from tensorflow.keras import regularizers
import tensorflow as tf


def f1_metric(y_true, y_pred):
    y_pred = tf.round(y_pred)
    y_true = tf.cast(y_true, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    return 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())

f1_metric_tf = tf.keras.metrics.MeanMetricWrapper(f1_metric, name="f1_metric")

num_features = X_SMOTE.shape[1]

def build_model(hp):
    inputs = Input(shape=(num_features,))
    x = inputs

    # Número de capas ocultas (1 o 2)
    n_layers = hp.Int("n_layers", 1, 2)
    for i in range(n_layers):
        units = hp.Int(f"units_layer_{i+1}", min_value=16, max_value=128, step=16)
        l2_reg = hp.Float(f"l2_layer_{i+1}", 1e-5, 1e-2, sampling="log")
        x = Dense(
            units=units,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_reg)
        )(x)
        dropout_rate = hp.Float(f"dropout_layer_{i+1}", 0.1, 0.5, step=0.1)
        x = Dropout(dropout_rate)(x)

    # Capa de salida para clasificación binaria
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inputs, outputs=outputs)

    # Optimizador y learning rate como hiperparámetros
    lr = hp.Float("lr", 1e-4, 1e-2, sampling="log")
    optimizer_name = hp.Choice("optimizer", ["adam", "sgd", "rmsprop", "adagrad"])
    if optimizer_name == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=lr)
    elif optimizer_name == "sgd":
        optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    elif optimizer_name == "rmsprop":
        optimizer = keras.optimizers.RMSprop(learning_rate=lr)
    elif optimizer_name == "adagrad":
        optimizer = keras.optimizers.Adagrad(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            f1_metric_tf,
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            "accuracy"
        ]
    )

    return model



# In[256]:


from sklearn.utils import class_weight
import keras_tuner as kt

def train_model(
    X, y, 
    X_val, y_val, 
    use_class_weights=False, 
    batch_size=64, 
    max_epochs=50,
    project_name='CC'
):
    # Calcular pesos si se requiere
    class_weights = None
    if use_class_weights:
        y_array = np.ravel(y)
        weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_array),
            y=y_array
        )
        class_weights = dict(enumerate(weights))
        print(f"Usando pesos de clase: {class_weights}")

    # Configurar tuner
    tuner = kt.Hyperband(
        build_model,
        objective=kt.Objective("val_f1_metric", "max"),
        executions_per_trial=1,
        max_epochs=max_epochs,
        factor=3,
        directory='output',
        project_name=project_name,
        overwrite=True
    )

    tuner.search_space_summary()
    print("\n--- Iniciando búsqueda con Hyperband ---")

    # Incluir class_weights solo si se usan
    tuner.search(
        X,
        y,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        class_weight=class_weights
    )

    print("\n--- Búsqueda Finalizada ---")
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    return best_hps


# In[ ]:


# Configuración de datasets
datasets = {
    "SMOTE":  {"X": X_SMOTE, "y": y_SMOTE, "use_class_weights": True},
    "CC":     {"X": X_CC,    "y": y_CC,    "use_class_weights": False},
    "TRAIN":  {"X": X_train_final, "y": y_train, "use_class_weights": True}
}

# Elegir el dataset que quieres probar (solo cambia el nombre)
selected_dataset = "SMOTE"  # puede ser "SMOTE", "CC" o "TRAIN"

# Extraer la configuración del dataset seleccionado
config = datasets[selected_dataset]

# Entrenar / tunear
best_hps = train_model(
    config["X"],
    config["y"],
    X_val_final,
    y_val,
    use_class_weights=config["use_class_weights"]
)


# In[ ]:


print(f"""
Mejores Hiperparámetros encontrados:
- Unidades Capa 1: {best_hps.get('units_layer_1')}
- Tasa de Aprendizaje (LR): {best_hps.get('lr'):.4f}
- Optimizador: {best_hps.get('optimizer')}
""")

best_model = build_model(best_hps)
best_model.save(f"best_model_{selected_dataset}.keras")


# In[ ]:


# Evaluar el modelo en los datos de test
results = best_model.evaluate(X_test_final, y_test, batch_size=32, verbose=1)

# Imprimir resultados
for name, value in zip(best_model.metrics_names, results):
    print(f"{name}: {value:.4f}")


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1️⃣ Predicciones probabilísticas sobre validación
y_val_prob = best_model.predict(X_val_final, batch_size=32)

# 2️⃣ Encontrar el mejor threshold usando validación
thresholds = np.arange(0.1, 0.9, 0.01)
best_f1 = 0
best_thresh = 0.5

for t in thresholds:
    y_pred_val = (y_val_prob > t).astype(int)
    f1 = f1_score(y_val, y_pred_val)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"Mejor threshold encontrado: {best_thresh:.2f} con F1: {best_f1:.4f}")

# 3️⃣ Aplicar threshold al test
y_test_prob = best_model.predict(X_test_final, batch_size=32)
y_pred = (y_test_prob > best_thresh).astype(int)

# 4️⃣ Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Matriz de Confusión")
plt.show()

# 5️⃣ Métricas finales
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}\n")
print("Reporte de clasificación:")
print(classification_report(y_test, y_pred, digits=4))


# In[ ]:


from sklearn.metrics import precision_recall_curve, auc

# Probabilidades predichas
y_pred_prob = best_model.predict(X_test_final, batch_size=32)

# Precision-Recall
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recall, precision)
print(f"Precision-Recall AUC: {pr_auc:.4f}")

# Graficar la curva
plt.figure(figsize=(6,5))
plt.plot(recall, precision, marker='.', label=f'PR-AUC = {pr_auc:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curva Precision-Recall')
plt.legend()
plt.grid(True)
plt.show()

