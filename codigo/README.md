# 🤟 Clasificador de Lengua de Señas Boliviana

Sistema de clasificación de videos de lengua de señas usando Deep Learning, optimizado para **CPU** y **principiantes**.

## 📊 Dataset

- **1,406 videos** de lengua de señas boliviana
- **71 clases** diferentes
- Videos cortos (~3 segundos promedio)
- Resolución 1280x720

## 🎯 Objetivo

Crear un **clasificador de videos** que reconozca señas individuales, como base para:

1. Sistema de clasificación en tiempo real (esta fase)
2. Detección en streaming (fase futura)

---

## 🚀 Guía de Uso Rápida

### 1️⃣ **Analizar el Dataset**

```bash
python step1_analizar_dataset.py
```

**¿Qué hace?**

- Escanea todos los videos
- Extrae características (duración, FPS, resolución)
- Genera estadísticas y visualizaciones

**Output:**

- `analisis_dataset/dataset_completo.csv`
- `analisis_dataset/estadisticas.json`
- Gráficas de distribución

---

### 2️⃣ **Preparar los Datos**

```bash
python step2_preparar_datos.py
```

**¿Qué hace?**

- Divide dataset en train/val/test (70%/15%/15%)
- Filtra clases con pocos ejemplos
- Crea mapeo de clases a índices

**Output:**

- `splits/train.csv` (984 videos)
- `splits/val.csv` (211 videos)
- `splits/test.csv` (211 videos)
- `splits/class_mapping.json`

---

### 3️⃣ **Probar el DataLoader**

```bash
python step3_crear_dataset.py
```

**¿Qué hace?**

- Prueba la carga de videos
- Verifica el formato de tensores
- Muestra ejemplos del dataset

**Output:**

- Información sobre el DataLoader
- Verificación de que todo funciona

---

### 4️⃣ **Probar los Modelos**

```bash
python step4_crear_modelo.py
```

**¿Qué hace?**

- Muestra 2 opciones de modelos:
  - **R(2+1)D**: Mejor accuracy, más lento
  - **Lightweight**: Más rápido, menor accuracy
- Prueba forward pass

**Output:**

- Información de arquitecturas
- Número de parámetros
- Recomendaciones

---

### 5️⃣ **Entrenar el Modelo** ⭐

#### Opción A: Modelo Ligero (RECOMENDADO PARA EMPEZAR)

```bash
python step5_entrenar.py --model lightweight --epochs 30 --batch_size 4
```

- ⏱️ **Tiempo estimado**: 2-4 horas
- 💾 **Parámetros**: ~88K
- 🎯 **Accuracy esperado**: 60-75%

#### Opción B: R(2+1)D (MEJOR ACCURACY)

```bash
python step5_entrenar.py --model r2plus1d --epochs 20 --batch_size 2
```

- ⏱️ **Tiempo estimado**: 6-10 horas
- 💾 **Parámetros**: ~31M
- 🎯 **Accuracy esperado**: 75-85%

**Argumentos disponibles:**

```
--model           lightweight o r2plus1d (default: lightweight)
--epochs          Número de épocas (default: 30)
--batch_size      Tamaño del batch (default: 4)
--lr              Learning rate (default: 0.001)
--patience        Épocas sin mejora para early stopping (default: 5)
--num_frames      Frames por video (default: 8)
--frame_size      Tamaño de frames (default: 112)
```

**Output:**

- `checkpoints_<modelo>/best_model.pth` - Mejor modelo
- `checkpoints_<modelo>/training_history.json` - Historial
- `checkpoints_<modelo>/training_curves.png` - Gráficas

---

### 6️⃣ **Evaluar el Modelo**

```bash
python step6_evaluar.py --model_path checkpoints_lightweight/best_model.pth
```

**¿Qué hace?**

- Evalúa en el test set
- Genera matriz de confusión
- Calcula métricas (accuracy, precision, recall)
- Muestra ejemplos de predicciones

---

### 7️⃣ **Hacer Predicciones**

```bash
python step7_predecir.py --model_path checkpoints_lightweight/best_model.pth --video_path ../videos/1/SALUDOS/HOLA.mp4
```

**¿Qué hace?**

- Clasifica un video nuevo
- Muestra top-5 predicciones
- Probabilidades de cada clase

---

## 📁 Estructura de Archivos

```
5 steps/
├── videos/                          # Dataset de videos
│   ├── 1/, 2/, 3/, 4/              # Carpetas organizadas
│   └── [categorías]/               # SALUDOS, NÚMEROS, etc.
│
└── codigo/                         # Todo el código aquí
    ├── step1_analizar_dataset.py   # Análisis del dataset
    ├── step2_preparar_datos.py     # División train/val/test
    ├── step3_crear_dataset.py      # DataLoader de PyTorch
    ├── step4_crear_modelo.py       # Definición de modelos
    ├── step5_entrenar.py           # Script de entrenamiento
    ├── step6_evaluar.py            # Evaluación del modelo
    ├── step7_predecir.py           # Predicción en videos nuevos
    │
    ├── analisis_dataset/           # Outputs del paso 1
    │   ├── dataset_completo.csv
    │   ├── estadisticas.json
    │   └── *.png
    │
    ├── splits/                     # Outputs del paso 2
    │   ├── train.csv
    │   ├── val.csv
    │   ├── test.csv
    │   └── class_mapping.json
    │
    └── checkpoints_*/              # Outputs del paso 5
        ├── best_model.pth
        ├── training_history.json
        └── training_curves.png
```

---

## 🛠️ Requisitos

### Software

```bash
# Ya instalado en tu entorno virtual
pip install torch torchvision opencv-python pandas matplotlib seaborn scikit-learn tqdm
```

### Hardware

- **CPU**: Cualquier CPU moderna (entrenamiento será lento)
- **RAM**: Mínimo 8GB recomendado
- **Disco**: ~5GB para dataset + modelos

---

## ⚙️ Optimizaciones para CPU

Este proyecto está **optimizado para CPU**:

1. **Frames reducidos**: 8 frames/video (vs 32 típico)
2. **Resolución baja**: 112x112 (vs 224x224 típico)
3. **Batch size pequeño**: 2-4 (vs 16-32 típico)
4. **Modelo ligero**: Opción con solo 88K parámetros
5. **Early stopping**: Detiene si no hay mejora

### Si tienes GPU disponible:

Modifica en `step5_entrenar.py`:

```bash
python step5_entrenar.py \
    --model r2plus1d \
    --epochs 50 \
    --batch_size 16 \
    --num_frames 16 \
    --frame_size 224
```

---

## 📈 Resultados Esperados

### Modelo Lightweight

| Métrica              | Valor Esperado |
| -------------------- | -------------- |
| Train Accuracy       | 70-80%         |
| Val Accuracy         | 60-75%         |
| Test Accuracy        | 60-75%         |
| Tiempo Entrenamiento | 2-4 horas      |

### Modelo R(2+1)D

| Métrica              | Valor Esperado |
| -------------------- | -------------- |
| Train Accuracy       | 85-95%         |
| Val Accuracy         | 75-85%         |
| Test Accuracy        | 75-85%         |
| Tiempo Entrenamiento | 6-10 horas     |

---

## 🐛 Solución de Problemas

### Error: "No se pudo abrir el video"

- Verifica que la ruta en `step3_crear_dataset.py` apunte correctamente a `../videos`
- Asegúrate de que los videos existan

### Entrenamiento muy lento

- Usa `--model lightweight` en vez de `r2plus1d`
- Reduce `--batch_size` a 2
- Reduce `--num_frames` a 6
- Considera entrenar menos épocas

### Out of Memory

- Reduce `--batch_size` a 2 o 1
- Reduce `--num_frames` a 6
- Usa modelo `lightweight`
- Reduce `--frame_size` a 96

### Accuracy no mejora

- Aumenta épocas (`--epochs 50`)
- Prueba diferentes learning rates (`--lr 0.0001` o `--lr 0.01`)
- Verifica que el dataset esté balanceado
- Considera usar modelo más grande (r2plus1d)

---

## 📚 Conceptos Clave (Para Principiantes)

### ¿Qué es un DataLoader?

Carga los datos en batches y los prepara para el modelo.

### ¿Qué es una época?

Una pasada completa por todo el dataset de entrenamiento.

### ¿Qué es Early Stopping?

Para el entrenamiento si el modelo no mejora, evita overfitting.

### ¿Qué es el Overfitting?

Cuando el modelo memoriza el train set pero falla en datos nuevos.

### ¿Qué es Accuracy?

Porcentaje de predicciones correctas.

### ¿Qué es un Checkpoint?

Guardado del estado del modelo durante el entrenamiento.

---

## 🎯 Plan de 1 Semana

### Día 1: Setup y Análisis

- ✅ Ejecutar steps 1-4
- ✅ Entender el dataset
- ✅ Probar que todo funciona

### Día 2-3: Entrenamiento Rápido

- 🏃 Entrenar modelo lightweight
- 📊 Analizar resultados
- 🔧 Ajustar hiperparámetros

### Día 4-6: Entrenamiento Final

- 🚀 Entrenar modelo R(2+1)D
- 📈 Monitorear métricas
- 💾 Guardar mejor modelo

### Día 7: Evaluación y Predicción

- ✅ Evaluar en test set
- 🎬 Probar con videos nuevos
- 📝 Documentar resultados

---

## 🔮 Próximos Pasos (Streaming)

Una vez que tengas un clasificador funcionando:

1. **Sliding Window**: Clasificar ventanas de video
2. **Buffer Management**: Procesar streaming en tiempo real
3. **Optimización**: TorchScript, ONNX, cuantización
4. **Deploy**: API con FastAPI o Flask

Los pasos 6 y 7 te prepararán para esto.

---

## 📞 Soporte

Si tienes dudas sobre algún paso:

1. Lee los comentarios en el código
2. Revisa la documentación en `0 docs/`
3. Verifica que seguiste los pasos en orden

---

## 🏆 Créditos

Dataset: Videos de Lengua de Señas Boliviana
Arquitectura: R(2+1)D de Facebook AI
Framework: PyTorch

---

## 📄 Licencia

Este proyecto es para uso educativo.

---

**¡Buena suerte con tu entrenamiento! 🚀**
