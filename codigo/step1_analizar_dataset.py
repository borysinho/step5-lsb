"""
PASO 1: Análisis del Dataset
================================
Este script analiza todos los videos del dataset para entender:
- Cuántos videos hay en total
- Cuántas clases/categorías existen
- Estadísticas de los videos (duración, FPS, resolución)
- Distribución de clases

Ejecutar: python step1_analizar_dataset.py
"""

import cv2
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración
VIDEO_ROOT = Path('../videos')  # Carpeta con videos
OUTPUT_DIR = Path('./analisis_dataset')
OUTPUT_DIR.mkdir(exist_ok=True)

def analizar_video(video_path):
    """
    Analiza un video y extrae sus características
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration': duration,
            'resolution': f"{width}x{height}"
        }
    except Exception as e:
        print(f"Error analizando {video_path}: {e}")
        return None

def escanear_dataset():
    """
    Escanea todo el dataset y recopila información
    """
    print("🔍 Escaneando dataset...")
    
    data = []
    
    # Recorrer todas las carpetas
    for carpeta_nivel1 in sorted(VIDEO_ROOT.iterdir()):
        if not carpeta_nivel1.is_dir():
            continue
        
        print(f"\n📁 Procesando carpeta: {carpeta_nivel1.name}")
        
        # Recorrer categorías dentro de cada carpeta
        for categoria in sorted(carpeta_nivel1.iterdir()):
            if not categoria.is_dir():
                continue
            
            categoria_nombre = categoria.name
            
            # Recorrer videos en la categoría
            videos = list(categoria.glob('*.mp4'))
            
            for video_path in tqdm(videos, desc=f"  {categoria_nombre}", leave=False):
                info = analizar_video(video_path)
                
                if info:
                    data.append({
                        'carpeta': carpeta_nivel1.name,
                        'categoria': categoria_nombre,
                        'video_name': video_path.stem,
                        'video_path': str(video_path.relative_to(VIDEO_ROOT)),
                        **info
                    })
    
    return pd.DataFrame(data)

def generar_estadisticas(df):
    """
    Genera estadísticas del dataset
    """
    print("\n" + "="*60)
    print("📊 ESTADÍSTICAS DEL DATASET")
    print("="*60)
    
    # Información general
    print(f"\n📹 Total de videos: {len(df)}")
    print(f"📂 Total de categorías únicas: {df['categoria'].nunique()}")
    print(f"🗂️  Total de carpetas: {df['carpeta'].nunique()}")
    
    # Estadísticas por carpeta
    print("\n📁 Videos por carpeta:")
    print(df['carpeta'].value_counts().sort_index())
    
    # Categorías más comunes
    print("\n🏷️  Top 10 categorías con más videos:")
    print(df['categoria'].value_counts().head(10))
    
    # Estadísticas de videos
    print("\n⏱️  Duración de videos:")
    print(f"  Promedio: {df['duration'].mean():.2f} segundos")
    print(f"  Mínimo: {df['duration'].min():.2f} segundos")
    print(f"  Máximo: {df['duration'].max():.2f} segundos")
    print(f"  Mediana: {df['duration'].median():.2f} segundos")
    
    print("\n🎬 FPS:")
    print(f"  Promedio: {df['fps'].mean():.2f}")
    print(f"  Valores únicos: {df['fps'].unique()}")
    
    print("\n📐 Resoluciones:")
    print(df['resolution'].value_counts())
    
    print("\n🎞️  Frames por video:")
    print(f"  Promedio: {df['frame_count'].mean():.1f} frames")
    print(f"  Mínimo: {df['frame_count'].min()} frames")
    print(f"  Máximo: {df['frame_count'].max()} frames")
    
    # Guardar estadísticas
    stats = {
        'total_videos': len(df),
        'total_categorias': df['categoria'].nunique(),
        'total_carpetas': df['carpeta'].nunique(),
        'duracion_promedio': float(df['duration'].mean()),
        'fps_promedio': float(df['fps'].mean()),
        'resolucion_mas_comun': df['resolution'].mode()[0],
        'categorias': df['categoria'].value_counts().to_dict()
    }
    
    with open(OUTPUT_DIR / 'estadisticas.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Estadísticas guardadas en: {OUTPUT_DIR / 'estadisticas.json'}")
    
    return stats

def generar_visualizaciones(df):
    """
    Genera visualizaciones del dataset
    """
    print("\n📈 Generando visualizaciones...")
    
    # Configurar estilo
    sns.set_style("whitegrid")
    
    # 1. Distribución de duraciones
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(df['duration'], bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Duración (segundos)')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Duraciones de Videos')
    plt.axvline(df['duration'].mean(), color='red', linestyle='--', 
                label=f'Media: {df["duration"].mean():.2f}s')
    plt.legend()
    
    # 2. Videos por categoría (top 20)
    plt.subplot(1, 2, 2)
    top_categorias = df['categoria'].value_counts().head(20)
    plt.barh(range(len(top_categorias)), top_categorias.values)
    plt.yticks(range(len(top_categorias)), top_categorias.index)
    plt.xlabel('Número de Videos')
    plt.ylabel('Categoría')
    plt.title('Top 20 Categorías con Más Videos')
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'distribucion_videos.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Videos por carpeta
    plt.figure(figsize=(10, 6))
    df['carpeta'].value_counts().sort_index().plot(kind='bar', edgecolor='black')
    plt.xlabel('Carpeta')
    plt.ylabel('Número de Videos')
    plt.title('Videos por Carpeta')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'videos_por_carpeta.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Distribución de FPS
    plt.figure(figsize=(8, 5))
    df['fps'].value_counts().sort_index().plot(kind='bar', edgecolor='black')
    plt.xlabel('FPS')
    plt.ylabel('Número de Videos')
    plt.title('Distribución de FPS')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'distribucion_fps.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualizaciones guardadas en: {OUTPUT_DIR}")

def main():
    """
    Función principal
    """
    print("="*60)
    print("🎯 PASO 1: ANÁLISIS DEL DATASET")
    print("="*60)
    
    # Verificar que existe la carpeta de videos
    if not VIDEO_ROOT.exists():
        print(f"❌ Error: No se encontró la carpeta {VIDEO_ROOT}")
        print(f"   Asegúrate de que la ruta sea correcta.")
        return
    
    # Escanear dataset
    df = escanear_dataset()
    
    if len(df) == 0:
        print("❌ No se encontraron videos en el dataset")
        return
    
    # Guardar DataFrame completo
    csv_path = OUTPUT_DIR / 'dataset_completo.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\n💾 Dataset guardado en: {csv_path}")
    
    # Generar estadísticas
    stats = generar_estadisticas(df)
    
    # Generar visualizaciones
    generar_visualizaciones(df)
    
    print("\n" + "="*60)
    print("✅ ANÁLISIS COMPLETADO")
    print("="*60)
    print(f"\nRevisa los resultados en la carpeta: {OUTPUT_DIR}")
    print("\nArchivos generados:")
    print("  📄 dataset_completo.csv - Información de todos los videos")
    print("  📄 estadisticas.json - Estadísticas resumidas")
    print("  📊 distribucion_videos.png - Visualización de distribuciones")
    print("  📊 videos_por_carpeta.png - Videos por carpeta")
    print("  📊 distribucion_fps.png - Distribución de FPS")
    
    print("\n🎯 PRÓXIMO PASO:")
    print("  Ejecuta: python step2_preparar_datos.py")

if __name__ == '__main__':
    main()
