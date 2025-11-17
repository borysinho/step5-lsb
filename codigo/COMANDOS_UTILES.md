# 🛠️ Comandos Útiles y Trucos

## 📌 Variables de Entorno Útiles

```bash
# Definir una vez al inicio de la sesión
cd "/home/bquiroga/Documents/dev/sw1/proyecto/test4/5 steps/codigo"
export PYTHON="../../venv312/bin/python"
```

Ahora puedes usar `$PYTHON` en vez de la ruta completa.

---

## 🔍 Comandos de Diagnóstico

### Ver cuántos videos hay

```bash
$PYTHON -c "import pandas as pd; df = pd.read_csv('./splits/train.csv'); print(f'Train: {len(df)} videos')"
```

### Ver número de clases

```bash
$PYTHON -c "import json; data = json.load(open('./splits/class_mapping.json')); print(f'Clases: {data[\"num_classes\"]}')"
```

### Ver estadísticas del dataset

```bash
cat analisis_dataset/estadisticas.json | grep -E "total_videos|total_categorias|duracion_promedio"
```

### Ver progreso del entrenamiento

```bash
# Si el entrenamiento está corriendo, en otra terminal:
tail -f nohup.out  # Si usaste nohup
# O simplemente observa la salida en la terminal
```

### Ver accuracy del mejor modelo

```bash
$PYTHON -c "import torch; checkpoint = torch.load('checkpoints_lightweight/best_model.pth', map_location='cpu'); print(f'Val Accuracy: {checkpoint.get(\"val_acc\", \"N/A\")}')"
```

---

## ⚡ Ejecutar en Background

Si quieres entrenar y cerrar la terminal:

```bash
nohup $PYTHON step5_entrenar.py --model lightweight --epochs 30 --batch_size 4 > training.log 2>&1 &
```

Luego puedes:

```bash
# Ver el proceso
ps aux | grep step5_entrenar

# Ver el log en tiempo real
tail -f training.log

# Matar el proceso si es necesario
kill <PID>
```

---

## 🎨 Personalizar Entrenamiento

### Entrenar con diferentes hiperparámetros

```bash
# Learning rate más bajo (más estable)
$PYTHON step5_entrenar.py --model lightweight --lr 0.0001 --epochs 40

# Learning rate más alto (más rápido pero inestable)
$PYTHON step5_entrenar.py --model lightweight --lr 0.01 --epochs 20

# Más frames (mejor accuracy, más lento)
$PYTHON step5_entrenar.py --model lightweight --num_frames 12 --batch_size 2

# Menos frames (más rápido, menor accuracy)
$PYTHON step5_entrenar.py --model lightweight --num_frames 6 --batch_size 6

# Mayor resolución
$PYTHON step5_entrenar.py --model lightweight --frame_size 128

# Menor resolución (MUY rápido)
$PYTHON step5_entrenar.py --model lightweight --frame_size 96 --batch_size 8
```

### Continuar entrenamiento desde checkpoint

```bash
# Primero, modifica step5_entrenar.py para cargar checkpoint
# O usa este truco rápido:
$PYTHON step5_entrenar.py --model lightweight --epochs 50  # Entrenará más épocas
```

---

## 📊 Análisis de Resultados

### Ver matriz de confusión

```bash
# Si tienes un visor de imágenes
xdg-open checkpoints_lightweight/evaluation/confusion_matrix.png
# O
display checkpoints_lightweight/evaluation/confusion_matrix.png
# O
firefox checkpoints_lightweight/evaluation/confusion_matrix.png
```

### Ver curvas de entrenamiento

```bash
xdg-open checkpoints_lightweight/training_curves.png
```

### Leer resultados JSON de forma legible

```bash
cat checkpoints_lightweight/evaluation/evaluation_results.json | python -m json.tool
```

### Encontrar las clases con peor accuracy

```bash
$PYTHON -c "
import json
with open('checkpoints_lightweight/evaluation/evaluation_results.json') as f:
    data = json.load(f)
worst = sorted(data['class_accuracies'], key=lambda x: x['accuracy'])[:10]
for item in worst:
    print(f'{item[\"class\"]:30s} {item[\"accuracy\"]:.2f}%')
"
```

---

## 🎬 Predicción en Múltiples Videos

### Predecir en todos los videos de una categoría

```bash
for video in ../videos/1/SALUDOS/*.mp4; do
    echo "=== Procesando: $video ==="
    $PYTHON step7_predecir.py --model_path checkpoints_lightweight/best_model.pth --video_path "$video"
    echo ""
done
```

### Predecir y guardar resultados en archivo

```bash
$PYTHON step7_predecir.py \
    --model_path checkpoints_lightweight/best_model.pth \
    --video_path ../videos/1/SALUDOS/HOLA.mp4 \
    > prediccion_HOLA.txt
```

---

## 🧹 Limpieza

### Limpiar checkpoints intermedios (mantener solo best_model)

```bash
cd checkpoints_lightweight
ls checkpoint_epoch_*.pth | grep -v best_model.pth | xargs rm -f
cd ..
```

### Limpiar todo para empezar de nuevo

```bash
# ⚠️ CUIDADO: Esto borra TODO el progreso
rm -rf analisis_dataset splits checkpoints_* evaluation
```

### Limpiar solo el entrenamiento (mantener análisis y splits)

```bash
rm -rf checkpoints_*
```

---

## 🔧 Debugging

### Probar carga de un video específico

```bash
$PYTHON -c "
from step3_crear_dataset import VideoDataset
import pandas as pd

df = pd.read_csv('./splits/train.csv')
video_path = '../videos/' + df.iloc[0]['video_path']
print(f'Probando: {video_path}')

dataset = VideoDataset(
    csv_file='./splits/train.csv',
    video_root='../videos',
    num_frames=8,
    frame_size=(112, 112)
)

frames, label = dataset[0]
print(f'Frames shape: {frames.shape}')
print(f'Label: {label}')
"
```

### Verificar que PyTorch funciona

```bash
$PYTHON -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

### Verificar OpenCV

```bash
$PYTHON -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
```

### Ver memoria usada durante entrenamiento

```bash
# En otra terminal mientras entrenas
watch -n 5 free -h
# O
htop
```

---

## 📈 Optimización

### Encontrar el batch_size óptimo

Prueba incrementalmente hasta que se quede sin memoria, luego usa el anterior:

```bash
$PYTHON step5_entrenar.py --model lightweight --batch_size 2 --epochs 1
$PYTHON step5_entrenar.py --model lightweight --batch_size 4 --epochs 1
$PYTHON step5_entrenar.py --model lightweight --batch_size 8 --epochs 1
$PYTHON step5_entrenar.py --model lightweight --batch_size 16 --epochs 1
```

### Probar diferentes modelos rápidamente

```bash
# 1 época de cada uno para comparar velocidad
$PYTHON step5_entrenar.py --model lightweight --epochs 1
$PYTHON step5_entrenar.py --model r2plus1d --epochs 1 --batch_size 2
```

---

## 🎯 Evaluación Rápida

### Evaluar en un subset del test set

```bash
# Modifica temporalmente test.csv para tener menos muestras
head -n 50 splits/test.csv > splits/test_small.csv
# Evaluar
$PYTHON step6_evaluar.py --model_path checkpoints_lightweight/best_model.pth
# Restaurar
rm splits/test_small.csv
```

---

## 📝 Exportar Resultados

### Generar reporte completo

```bash
echo "=== REPORTE DEL MODELO ===" > reporte.txt
echo "" >> reporte.txt
echo "Dataset:" >> reporte.txt
cat analisis_dataset/estadisticas.json >> reporte.txt
echo "" >> reporte.txt
echo "Evaluación:" >> reporte.txt
cat checkpoints_lightweight/evaluation/evaluation_results.json >> reporte.txt
```

### Copiar todos los resultados a una carpeta

```bash
mkdir -p resultados_finales
cp checkpoints_lightweight/best_model.pth resultados_finales/
cp checkpoints_lightweight/training_curves.png resultados_finales/
cp checkpoints_lightweight/evaluation/* resultados_finales/
cp analisis_dataset/estadisticas.json resultados_finales/
```

---

## 🚀 Tips Avanzados

### Benchmark de velocidad

```bash
time $PYTHON step5_entrenar.py --model lightweight --epochs 1
```

### Ver tamaño del modelo

```bash
du -h checkpoints_lightweight/best_model.pth
```

### Convertir modelo a modo inferencia (más rápido)

```bash
$PYTHON -c "
import torch
model = torch.load('checkpoints_lightweight/best_model.pth', map_location='cpu')
# Aquí puedes aplicar optimizaciones como quantization
torch.save(model, 'checkpoints_lightweight/model_optimized.pth')
"
```

---

## 📚 Recursos Adicionales

### Ver documentación de PyTorch

```bash
$PYTHON -c "import torch; help(torch.nn.Conv3d)"
```

### Buscar un error en internet

```bash
# Copia el error y búscalo en:
# - Stack Overflow
# - PyTorch Forums
# - GitHub Issues
```

---

## 💡 Recuerda

1. **Siempre usa el entorno virtual**: `../../venv312/bin/python`
2. **Guarda checkpoints frecuentemente**: Ya está configurado
3. **Monitorea el entrenamiento**: No lo dejes solo sin supervisión inicial
4. **Prueba con pocos epochs primero**: Asegúrate que funciona antes de entrenar 30 épocas
5. **Haz backup del mejor modelo**: Copia `best_model.pth` a un lugar seguro

---

**Happy Training! 🎓🚀**
