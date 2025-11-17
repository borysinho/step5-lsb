"""
PASO 2: Preparación de Datos
================================
Este script:
1. Lee el CSV generado en el Paso 1
2. Divide el dataset en train/val/test (70%/15%/15%)
3. Balancea las clases si es necesario
4. Genera archivos CSV con las divisiones
5. Crea un mapeo de clases a índices

Ejecutar: python step2_preparar_datos.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración
DATASET_CSV = Path('./analisis_dataset/dataset_completo.csv')
OUTPUT_DIR = Path('./splits')
OUTPUT_DIR.mkdir(exist_ok=True)

# Parámetros de división
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_SEED = 42

# Filtros
MIN_SAMPLES_PER_CLASS = 10  # Clases con menos de 10 videos se excluyen (necesario para stratify)

def cargar_dataset():
    """
    Carga el dataset desde el CSV
    """
    print("📂 Cargando dataset...")
    df = pd.read_csv(DATASET_CSV)
    print(f"   Total de videos: {len(df)}")
    print(f"   Total de categorías: {df['categoria'].nunique()}")
    return df

def filtrar_clases_pequenas(df, min_samples):
    """
    Filtra clases con muy pocos ejemplos
    """
    print(f"\n🔍 Filtrando clases con menos de {min_samples} videos...")
    
    # Contar videos por categoría
    class_counts = df['categoria'].value_counts()
    
    # Clases que cumplen el mínimo
    valid_classes = class_counts[class_counts >= min_samples].index
    
    # Filtrar dataset
    df_filtered = df[df['categoria'].isin(valid_classes)].copy()
    
    removed_classes = len(class_counts) - len(valid_classes)
    removed_videos = len(df) - len(df_filtered)
    
    print(f"   ❌ Clases eliminadas: {removed_classes}")
    print(f"   ❌ Videos eliminados: {removed_videos}")
    print(f"   ✅ Clases restantes: {len(valid_classes)}")
    print(f"   ✅ Videos restantes: {len(df_filtered)}")
    
    return df_filtered

def crear_splits(df):
    """
    Divide el dataset en train/val/test manteniendo distribución de clases
    """
    print(f"\n📊 Dividiendo dataset en train/val/test...")
    print(f"   Train: {TRAIN_SIZE*100:.0f}%")
    print(f"   Val: {VAL_SIZE*100:.0f}%")
    print(f"   Test: {TEST_SIZE*100:.0f}%")
    
    # Primero separar train del resto
    train_df, temp_df = train_test_split(
        df,
        train_size=TRAIN_SIZE,
        stratify=df['categoria'],
        random_state=RANDOM_SEED
    )
    
    # Luego dividir el resto en val y test
    val_ratio = VAL_SIZE / (VAL_SIZE + TEST_SIZE)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio,
        stratify=temp_df['categoria'],
        random_state=RANDOM_SEED
    )
    
    print(f"\n✅ División completada:")
    print(f"   Train: {len(train_df)} videos ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   Val: {len(val_df)} videos ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   Test: {len(test_df)} videos ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df

def crear_mapeo_clases(df):
    """
    Crea mapeo de nombres de clases a índices numéricos
    """
    print("\n🏷️  Creando mapeo de clases...")
    
    # Ordenar clases alfabéticamente para consistencia
    clases = sorted(df['categoria'].unique())
    
    # Crear diccionarios de mapeo
    class_to_idx = {clase: idx for idx, clase in enumerate(clases)}
    idx_to_class = {idx: clase for clase, idx in class_to_idx.items()}
    
    print(f"   Total de clases: {len(clases)}")
    print(f"\n   Primeras 10 clases:")
    for i, clase in enumerate(list(clases)[:10]):
        print(f"      {i}: {clase}")
    
    return class_to_idx, idx_to_class

def agregar_labels(df, class_to_idx):
    """
    Agrega columna con índices numéricos de las clases
    """
    df['label'] = df['categoria'].map(class_to_idx)
    return df

def guardar_splits(train_df, val_df, test_df):
    """
    Guarda los splits en archivos CSV
    """
    print("\n💾 Guardando splits...")
    
    train_path = OUTPUT_DIR / 'train.csv'
    val_path = OUTPUT_DIR / 'val.csv'
    test_path = OUTPUT_DIR / 'test.csv'
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"   ✅ Train guardado en: {train_path}")
    print(f"   ✅ Val guardado en: {val_path}")
    print(f"   ✅ Test guardado en: {test_path}")

def guardar_mapeo_clases(class_to_idx, idx_to_class):
    """
    Guarda el mapeo de clases en JSON
    """
    mapeo_path = OUTPUT_DIR / 'class_mapping.json'
    
    mapeo = {
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class,
        'num_classes': len(class_to_idx)
    }
    
    with open(mapeo_path, 'w', encoding='utf-8') as f:
        json.dump(mapeo, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Mapeo de clases guardado en: {mapeo_path}")

def visualizar_distribucion(train_df, val_df, test_df):
    """
    Visualiza la distribución de clases en cada split
    """
    print("\n📊 Generando visualizaciones...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Train
    train_counts = train_df['categoria'].value_counts().head(20)
    axes[0].barh(range(len(train_counts)), train_counts.values)
    axes[0].set_yticks(range(len(train_counts)))
    axes[0].set_yticklabels(train_counts.index, fontsize=8)
    axes[0].set_xlabel('Número de Videos')
    axes[0].set_title(f'Train Set - Top 20 Clases\n({len(train_df)} videos)')
    axes[0].invert_yaxis()
    
    # Val
    val_counts = val_df['categoria'].value_counts().head(20)
    axes[1].barh(range(len(val_counts)), val_counts.values)
    axes[1].set_yticks(range(len(val_counts)))
    axes[1].set_yticklabels(val_counts.index, fontsize=8)
    axes[1].set_xlabel('Número de Videos')
    axes[1].set_title(f'Val Set - Top 20 Clases\n({len(val_df)} videos)')
    axes[1].invert_yaxis()
    
    # Test
    test_counts = test_df['categoria'].value_counts().head(20)
    axes[2].barh(range(len(test_counts)), test_counts.values)
    axes[2].set_yticks(range(len(test_counts)))
    axes[2].set_yticklabels(test_counts.index, fontsize=8)
    axes[2].set_xlabel('Número de Videos')
    axes[2].set_title(f'Test Set - Top 20 Clases\n({len(test_df)} videos)')
    axes[2].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'distribucion_splits.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Visualización guardada en: {OUTPUT_DIR / 'distribucion_splits.png'}")

def generar_resumen(train_df, val_df, test_df, class_to_idx):
    """
    Genera un resumen de la preparación de datos
    """
    resumen = {
        'total_videos': int(len(train_df) + len(val_df) + len(test_df)),
        'num_classes': int(len(class_to_idx)),
        'train': {
            'num_videos': int(len(train_df)),
            'porcentaje': float(len(train_df) / (len(train_df) + len(val_df) + len(test_df)) * 100),
            'videos_por_clase': {
                'min': int(train_df['categoria'].value_counts().min()),
                'max': int(train_df['categoria'].value_counts().max()),
                'mean': float(train_df['categoria'].value_counts().mean())
            }
        },
        'val': {
            'num_videos': int(len(val_df)),
            'porcentaje': float(len(val_df) / (len(train_df) + len(val_df) + len(test_df)) * 100),
        },
        'test': {
            'num_videos': int(len(test_df)),
            'porcentaje': float(len(test_df) / (len(train_df) + len(val_df) + len(test_df)) * 100),
        }
    }
    
    resumen_path = OUTPUT_DIR / 'resumen_splits.json'
    with open(resumen_path, 'w', encoding='utf-8') as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Resumen guardado en: {resumen_path}")
    
    return resumen

def main():
    """
    Función principal
    """
    print("="*60)
    print("🎯 PASO 2: PREPARACIÓN DE DATOS")
    print("="*60)
    
    # 1. Cargar dataset
    df = cargar_dataset()
    
    # 2. Filtrar clases pequeñas
    df = filtrar_clases_pequenas(df, MIN_SAMPLES_PER_CLASS)
    
    # 3. Crear mapeo de clases
    class_to_idx, idx_to_class = crear_mapeo_clases(df)
    
    # 4. Agregar labels numéricos
    df = agregar_labels(df, class_to_idx)
    
    # 5. Crear splits
    train_df, val_df, test_df = crear_splits(df)
    
    # 6. Guardar splits
    guardar_splits(train_df, val_df, test_df)
    
    # 7. Guardar mapeo de clases
    guardar_mapeo_clases(class_to_idx, idx_to_class)
    
    # 8. Visualizar distribución
    visualizar_distribucion(train_df, val_df, test_df)
    
    # 9. Generar resumen
    resumen = generar_resumen(train_df, val_df, test_df, class_to_idx)
    
    print("\n" + "="*60)
    print("✅ PREPARACIÓN COMPLETADA")
    print("="*60)
    print(f"\n📊 RESUMEN:")
    print(f"   Total de clases: {resumen['num_classes']}")
    print(f"   Total de videos: {resumen['total_videos']}")
    print(f"\n   Train: {resumen['train']['num_videos']} videos ({resumen['train']['porcentaje']:.1f}%)")
    print(f"      Videos por clase: {resumen['train']['videos_por_clase']['min']}-{resumen['train']['videos_por_clase']['max']} (promedio: {resumen['train']['videos_por_clase']['mean']:.1f})")
    print(f"   Val: {resumen['val']['num_videos']} videos ({resumen['val']['porcentaje']:.1f}%)")
    print(f"   Test: {resumen['test']['num_videos']} videos ({resumen['test']['porcentaje']:.1f}%)")
    
    print("\n🎯 PRÓXIMO PASO:")
    print("   Ejecuta: python step3_crear_dataset.py")
    print("\n💡 TIP IMPORTANTE:")
    print("   Como tienes CPU, el entrenamiento será LENTO.")
    print("   Vamos a usar configuraciones optimizadas para CPU en los siguientes pasos.")

if __name__ == '__main__':
    main()
