#!/bin/bash

# Script para ejecutar todo el pipeline automáticamente
# Uso: bash run_pipeline.sh [modelo]
# Ejemplo: bash run_pipeline.sh lightweight
# Ejemplo: bash run_pipeline.sh r2plus1d

set -e  # Detener si hay error

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════╗"
echo "║   PIPELINE COMPLETO - CLASIFICADOR LENGUA DE SEÑAS    ║"
echo "╚════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Parámetros
MODEL=${1:-lightweight}  # Por defecto lightweight
PYTHON_BIN="../../venv312/bin/python"

echo -e "${YELLOW}📋 Configuración:${NC}"
echo "   Modelo: $MODEL"
echo "   Python: $PYTHON_BIN"
echo ""

# Paso 1: Análisis del Dataset
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}PASO 1: Análisis del Dataset${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"

if [ -f "./analisis_dataset/dataset_completo.csv" ]; then
    echo -e "${YELLOW}⚠️  Análisis ya existe. ¿Deseas omitir? (s/n)${NC}"
    read -r skip
    if [ "$skip" != "s" ]; then
        $PYTHON_BIN step1_analizar_dataset.py
    else
        echo "✅ Paso omitido"
    fi
else
    $PYTHON_BIN step1_analizar_dataset.py
fi

echo ""

# Paso 2: Preparación de Datos
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}PASO 2: Preparación de Datos${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"

if [ -f "./splits/train.csv" ]; then
    echo -e "${YELLOW}⚠️  Splits ya existen. ¿Deseas omitir? (s/n)${NC}"
    read -r skip
    if [ "$skip" != "s" ]; then
        $PYTHON_BIN step2_preparar_datos.py
    else
        echo "✅ Paso omitido"
    fi
else
    $PYTHON_BIN step2_preparar_datos.py
fi

echo ""

# Paso 3: Verificación del DataLoader
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}PASO 3: Verificación del DataLoader${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"

$PYTHON_BIN step3_crear_dataset.py

echo ""

# Paso 4: Verificación del Modelo
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}PASO 4: Verificación del Modelo${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"

$PYTHON_BIN step4_crear_modelo.py

echo ""

# Paso 5: Entrenamiento
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}PASO 5: Entrenamiento del Modelo${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"

echo -e "${YELLOW}⚠️  El entrenamiento puede tardar varias horas.${NC}"
echo -e "${YELLOW}¿Deseas continuar? (s/n)${NC}"
read -r continue

if [ "$continue" == "s" ]; then
    if [ "$MODEL" == "lightweight" ]; then
        $PYTHON_BIN step5_entrenar.py --model lightweight --epochs 30 --batch_size 4
    else
        $PYTHON_BIN step5_entrenar.py --model r2plus1d --epochs 20 --batch_size 2
    fi
else
    echo "⏹️  Entrenamiento omitido"
    echo ""
    echo -e "${YELLOW}Para entrenar manualmente:${NC}"
    echo "   $PYTHON_BIN step5_entrenar.py --model $MODEL --epochs 30"
    exit 0
fi

echo ""

# Paso 6: Evaluación
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}PASO 6: Evaluación del Modelo${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"

MODEL_PATH="./checkpoints_${MODEL}/best_model.pth"

if [ -f "$MODEL_PATH" ]; then
    $PYTHON_BIN step6_evaluar.py --model_path "$MODEL_PATH" --model_type "$MODEL"
else
    echo -e "${RED}❌ No se encontró el modelo entrenado: $MODEL_PATH${NC}"
    echo "   El entrenamiento puede haber fallado o no completarse."
    exit 1
fi

echo ""

# Resumen final
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ PIPELINE COMPLETADO${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"

echo ""
echo -e "${GREEN}📊 Archivos generados:${NC}"
echo "   📁 analisis_dataset/ - Análisis del dataset"
echo "   📁 splits/ - División train/val/test"
echo "   📁 checkpoints_${MODEL}/ - Modelo entrenado"
echo "   📁 checkpoints_${MODEL}/evaluation/ - Resultados de evaluación"

echo ""
echo -e "${YELLOW}🎯 Próximos pasos:${NC}"
echo "   1. Revisar las métricas en checkpoints_${MODEL}/evaluation/"
echo "   2. Probar predicción:"
echo "      $PYTHON_BIN step7_predecir.py --model_path $MODEL_PATH --video_path ../videos/1/SALUDOS/HOLA.mp4"

echo ""
echo -e "${GREEN}🎉 ¡Felicidades! Tu modelo está listo.${NC}"
