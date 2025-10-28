# Explicación Detallada del Preprocesamiento de Datos

Te voy a explicar cada decisión del código de forma clara y estructurada:

## 1. Carga del Dataset

```python
(conjunto_entrenamiento_raw, conjunto_test_raw), info_dataset = tfds.load('stl10', split=['train', 'test'], as_supervised=True, with_info=True)
```

**Decisiones tomadas:**

- **`as_supervised=True`**: Hace que cada ejemplo del dataset venga en formato tupla `(imagen, etiqueta)`, lo cual es más conveniente para entrenar modelos supervisados. Sin esto, tendríamos un diccionario y sería más engorroso acceder a los datos.

- **`with_info=True`**: Nos da metadatos útiles del dataset (número de clases, dimensiones de las imágenes, etc.) sin tener que calcularlo manualmente o buscarlo en documentación.

- **`split=['train', 'test']`**: Cargamos solo los conjuntos de entrenamiento y test. STL-10 también tiene un conjunto "unlabeled" que no necesitamos para clasificación supervisada básica.

## 2. Extracción de Metadatos

```python
num_clases = info_dataset.features['label'].num_classes
nombres_clases = info_dataset.features['label'].names
tamano_imagen = info_dataset.features['image'].shape
```

**Por qué hacemos esto:**

- **Flexibilidad**: Si cambias de dataset en el futuro, no necesitas modificar valores hardcodeados. El código se adapta automáticamente.
- **Claridad**: Es más legible usar `num_clases` que poner directamente `10` en el código.
- **Prevención de errores**: Si el dataset cambia de versión y tiene 12 clases en lugar de 10, tu código no se romperá.

## 3. Función de Preprocesamiento

### a) Normalización de Píxeles

```python
imagen = tf.cast(imagen, tf.float32) / 255.0
```

**Razones:**

- **De uint8 a float32**: Las imágenes vienen como enteros entre 0-255. Las redes neuronales funcionan mejor con números flotantes.

- **Dividir por 255.0**: Escala los valores al rango [0, 1]. Esto es crucial porque:
  - Las redes neuronales convergen más rápido con valores pequeños
  - Evita problemas numéricos (gradientes muy grandes o muy pequeños)
  - Facilita que los pesos iniciales estén en un rango apropiado

### b) Aplanamiento de la Imagen

```python
imagen = tf.reshape(imagen, [-1])
```

**Por qué aplanamos:**

- **Redes densas requieren entrada 1D**: Una red neuronal densa (fully connected) espera un vector, no una matriz 2D ni 3D.

- **El `-1` es inteligente**: Le dice a TensorFlow "calcula tú cuántos elementos hay". Para una imagen de 96×96×3, automáticamente se convierte en 27,648 elementos.

- **Alternativa sería manual**: Podrías poner `[96*96*3]`, pero `-1` es más flexible y menos propenso a errores.

### c) Codificación One-Hot

```python
etiqueta = tf.one_hot(etiqueta, depth=num_clases)
```

**Por qué one-hot:**

- **De etiqueta numérica a vector**: Convierte una etiqueta como `3` en `[0,0,0,1,0,0,0,0,0,0]`

- **Compatible con softmax**: La capa de salida de tu red usará softmax, que genera probabilidades para cada clase. Necesita que las etiquetas reales también estén en ese formato.

- **Para usar categorical_crossentropy**: Esta función de pérdida requiere etiquetas en formato one-hot. Si usaras `sparse_categorical_crossentropy`, no necesitarías one-hot.

## 4. Aplicar Preprocesamiento

```python
conjunto_entrenamiento = conjunto_entrenamiento_raw.map(preprocesado)
conjunto_test = conjunto_test_raw.map(preprocesado)
```

**Por qué usar `.map()`:**

- **Eficiencia**: No carga todo en memoria de golpe. Procesa las imágenes bajo demanda (lazy evaluation).
- **Paralelización**: TensorFlow puede procesar múltiples imágenes en paralelo automáticamente.
- **Elegancia**: Una línea en lugar de un bucle for manual.

## 5. Barajado de Datos

```python
conjunto_entrenamiento = conjunto_entrenamiento.shuffle(5000, reshuffle_each_iteration=True)
```

**Decisiones críticas:**

- **Buffer de 5000**: Es el tamaño completo del conjunto de entrenamiento. Esto asegura un barajado completamente aleatorio. Si pusieras 100, solo barajaría dentro de ventanas de 100 ejemplos, lo cual sería menos aleatorio.

- **`reshuffle_each_iteration=True`**: Baraja de nuevo en cada época. Esto es importante porque:
  - Evita que la red memorice el orden de los datos
  - Mejora la generalización
  - Previene el sobreajuste a patrones espurios del orden

## 6. División Entrenamiento/Validación

```python
tamano_validacion = int(0.2 * 5000)  # 1000 ejemplos
conjunto_validacion = conjunto_entrenamiento.take(tamano_validacion).batch(32)
conjunto_entrenamiento = conjunto_entrenamiento.skip(tamano_validacion).batch(32)
```

**Por qué 80/20:**

- **Estándar de la industria**: 80% entrenamiento, 20% validación es una práctica común y balanceada.

- **Suficientes datos para validar**: 1000 ejemplos dan una estimación estable del rendimiento sin desperdiciar demasiados datos de entrenamiento.

- **`.take()` y `.skip()` son complementarios**:
  - `take(1000)` agarra los primeros 1000
  - `skip(1000)` descarta esos mismos 1000 y se queda con el resto
  - Así no hay solapamiento entre validación y entrenamiento

## 7. Creación de Batches

```python
.batch(32)
```

**Por qué tamaño 32:**

- **Compromiso entre velocidad y estabilidad**:
  - Batch más grande (128, 256): Más rápido pero gradientes menos precisos y requiere más memoria
  - Batch más pequeño (8, 16): Gradientes más ruidosos pero mejor generalización
  - 32 es un punto medio popular

- **Potencia de 2**: Los procesadores (CPU/GPU) son más eficientes con tamaños que son potencias de 2 (16, 32, 64, 128).

- **Memoria GPU**: 32 suele caber cómodamente en la mayoría de GPUs sin quedarse sin memoria.

## Resumen Visual del Pipeline

```
Dataset crudo (imágenes 96×96×3, etiquetas 0-9)
    ↓
Normalización (píxeles 0-255 → 0-1)
    ↓
Aplanamiento (96×96×3 → vector de 27,648)
    ↓
One-hot encoding (etiqueta 3 → [0,0,0,1,0,0,0,0,0,0])
    ↓
Barajado (orden aleatorio, rebarajado cada época)
    ↓
División 80/20 (4000 train, 1000 val)
    ↓
Agrupación en batches de 32
    ↓
Listo para entrenar
```

Cada decisión tiene una razón práctica que mejora el entrenamiento, la eficiencia o la generalización del modelo. ¿Hay alguna parte específica que quieras que profundice más?