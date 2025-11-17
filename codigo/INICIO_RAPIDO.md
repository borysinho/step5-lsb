# 🚀 INICIO RÁPIDO - Clasificador de Lengua de Señas

## ⚡ Opción 1: Ejecutar Todo Automáticamente

```bash
cd "5 steps/codigo"
bash run_pipeline.sh lightweight
```

Esto ejecutará automáticamente:

1. ✅ Análisis del dataset
2. ✅ Preparación de datos
3. ✅ Verificación del DataLoader
4. ✅ Verificación del modelo
5. ⏱️ Entrenamiento (~2-4 horas)
6. ✅ Evaluación

---

## ⚙️ Opción 2: Paso a Paso Manual

### Preparación (una sola vez)

```bash
cd "5 steps/codigo"
PYTHON="../../venv312/bin/python"
```

### Paso 1: Analizar Dataset

```bash
$PYTHON step1_analizar_dataset.py
```

### Paso 2: Preparar Datos

```bash
$PYTHON step2_preparar_datos.py
```

### Paso 3: Verificar DataLoader

```bash
$PYTHON step3_crear_dataset.py
```

### Paso 4: Verificar Modelo

```bash
$PYTHON step4_crear_modelo.py
```

### Paso 5: Entrenar (⭐ IMPORTANTE)

**Opción A - Rápido (2-4 horas):**

```bash
$PYTHON step5_entrenar.py --model lightweight --epochs 30 --batch_size 4
```

**Opción B - Mejor accuracy (6-10 horas):**

```bash
$PYTHON step5_entrenar.py --model r2plus1d --epochs 20 --batch_size 2
```

### Paso 6: Evaluar

```bash
$PYTHON step6_evaluar.py --model_path checkpoints_lightweight/best_model.pth --model_type lightweight
```

### Paso 7: Predecir

```bash
$PYTHON step7_predecir.py \
    --model_path checkpoints_lightweight/best_model.pth \
    --video_path ../videos/1/SALUDOS/HOLA.mp4
```

---

## 📊 ¿Qué Esperar?

### Durante el Entrenamiento

- Verás barras de progreso para cada época
- El modelo se guarda automáticamente cada 5 épocas
- Si no mejora por 5 épocas, se detiene (early stopping)

### Resultados Típicos

| Modelo      | Tiempo | Accuracy Esperado | Parámetros |
| ----------- | ------ | ----------------- | ---------- |
| Lightweight | 2-4h   | 60-75%            | 88K        |
| R(2+1)D     | 6-10h  | 75-85%            | 31M        |

---

## 🎯 Archivos Importantes

Después de ejecutar todo:

```
codigo/
├── analisis_dataset/
│   ├── dataset_completo.csv          ← Info de todos los videos
│   ├── estadisticas.json             ← Stats del dataset
│   └── *.png                          ← Gráficas
│
├── splits/
│   ├── train.csv                      ← Videos de entrenamiento
│   ├── val.csv                        ← Videos de validación
│   ├── test.csv                       ← Videos de prueba
│   └── class_mapping.json             ← Mapeo de clases
│
└── checkpoints_lightweight/           (o checkpoints_r2plus1d/)
    ├── best_model.pth                 ← 🎯 TU MODELO ENTRENADO
    ├── training_history.json          ← Historial de entrenamiento
    ├── training_curves.png            ← Gráficas de loss/accuracy
    └── evaluation/
        ├── confusion_matrix.png       ← Matriz de confusión
        └── evaluation_results.json    ← Métricas detalladas
```

---

## 💡 Consejos

### Si el entrenamiento es MUY lento:

1. Para el proceso (Ctrl+C)
2. Edita `config.py`:
   ```python
   NUM_FRAMES_CPU = 6        # Reduce de 8 a 6
   BATCH_SIZE_CPU = 2        # Reduce de 4 a 2
   FRAME_SIZE_CPU = (96, 96) # Reduce de 112 a 96
   ```
3. Reinicia el entrenamiento

### Si te quedas sin memoria:

```bash
$PYTHON step5_entrenar.py --model lightweight --batch_size 1 --num_frames 6
```

### Para monitorear el progreso:

El script muestra en tiempo real:

- Loss (debería bajar)
- Accuracy (debería subir)
- Learning rate (puede cambiar si usa scheduler)

---

## 🐛 Problemas Comunes

### "No se pudo abrir el video"

→ Verifica que `../videos` existe desde la carpeta `codigo/`

### "CUDA out of memory"

→ Estás usando CPU, este error NO debería aparecer
→ Si aparece, reduce batch_size a 1

### "El accuracy no mejora"

→ Normal al principio
→ Espera al menos 10 épocas
→ Si sigue sin mejorar, revisa que los videos se carguen bien

### "Entrenamiento interrumpido"

→ No te preocupes, los checkpoints están guardados
→ Puedes continuar o evaluar el último checkpoint

---

## 📈 Interpretando Resultados

### Training Curves (training_curves.png)

- **Loss bajando** = ✅ El modelo aprende
- **Loss subiendo** = ❌ Algo está mal
- **Accuracy subiendo** = ✅ Mejorando
- **Gap grande train vs val** = ⚠️ Overfitting

### Matriz de Confusión

- **Diagonal oscura** = ✅ Buenas predicciones
- **Manchas fuera de diagonal** = ❌ Confusiones comunes

### Accuracy

- **>80%** = 🎉 Excelente
- **70-80%** = ✅ Muy bien
- **60-70%** = ⚠️ Aceptable
- **<60%** = ❌ Necesita mejoras

---

## 🎓 Para Principiantes

### ¿Qué está pasando?

1. **Análisis**: Entendemos el dataset (cuántos videos, clases, etc.)
2. **Preparación**: Dividimos en train/val/test (como estudiar para un examen)
3. **DataLoader**: Prepara los videos en el formato que el modelo necesita
4. **Modelo**: La "red neuronal" que aprenderá a reconocer señas
5. **Entrenamiento**: El modelo "estudia" los videos del train set
6. **Validación**: Verificamos que no esté memorizando (overfitting)
7. **Evaluación**: Probamos en videos que NUNCA vio antes (test set)
8. **Predicción**: Usamos el modelo en videos nuevos

### ¿Por qué tarda tanto?

Tu CPU debe procesar:

- 984 videos de entrenamiento
- 8 frames por video
- 30 épocas
- = ~236,160 frames totales

Para cada frame, hace millones de cálculos matemáticos.

---

## 🚀 Siguiente Nivel: Streaming

Una vez que tengas tu modelo funcionando:

1. Lee `0 docs/05_implementacion_streaming.md`
2. Implementa sliding window
3. Prueba en webcam o streaming

**El código ya está preparado para esto!** Solo necesitas:

- Tomar frames de la cámara
- Aplicar `step7_predecir.py` en ventanas deslizantes
- Suavizar predicciones consecutivas

---

## ✅ Checklist

Antes de comenzar, verifica:

- [ ] Estás en la carpeta `5 steps/codigo/`
- [ ] El entorno virtual está activado
- [ ] Tienes espacio en disco (~5GB)
- [ ] Tienes 2-10 horas disponibles para entrenar
- [ ] Puedes dejar la computadora trabajando

Durante el entrenamiento:

- [ ] Monitorea que loss baje
- [ ] Monitorea que accuracy suba
- [ ] Verifica que se guarden checkpoints
- [ ] No apagues la computadora 😅

Después del entrenamiento:

- [ ] Revisa las gráficas en training_curves.png
- [ ] Revisa accuracy en evaluation_results.json
- [ ] Prueba predicción con step7_predecir.py
- [ ] Celebra tu logro 🎉

---

**¡Éxito en tu proyecto! 🤟**

Si tienes dudas, revisa:

1. Los comentarios en cada script
2. El README.md completo
3. La documentación en `0 docs/`
