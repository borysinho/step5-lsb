"""
PASO 6: Evaluación del Modelo
================================
Este script evalúa el modelo entrenado en el test set y genera:
- Accuracy total y por clase
- Matriz de confusión
- Top-5 accuracy
- Análisis de errores comunes
- Visualizaciones

Ejecutar:
python step6_evaluar.py --model_path checkpoints_lightweight/best_model.pth
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json
import argparse
from pathlib import Path
from tqdm import tqdm

# Importar nuestros módulos
from step3_crear_dataset import crear_dataloaders
from step4_crear_modelo import crear_modelo


def evaluar_modelo(model, test_loader, device, class_names):
    """
    Evalúa el modelo en el test set
    
    Returns:
    --------
    results : dict
        Diccionario con todas las métricas
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_top5_correct = []
    
    print("\n🔍 Evaluando modelo en test set...")
    
    with torch.no_grad():
        for frames, labels in tqdm(test_loader, desc='Evaluando'):
            frames = frames.to(device)
            labels = labels.to(device)
            
            outputs = model(frames)
            
            # Top-1 prediction
            _, predicted = outputs.max(1)
            
            # Top-5 predictions
            _, top5_pred = outputs.topk(5, dim=1)
            top5_correct = top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred))
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_top5_correct.extend(top5_correct.any(dim=1).cpu().numpy())
    
    # Convertir a numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_top5_correct = np.array(all_top5_correct)
    
    # Calcular métricas
    accuracy = (all_preds == all_labels).mean() * 100
    top5_accuracy = all_top5_correct.mean() * 100
    
    results = {
        'accuracy': accuracy,
        'top5_accuracy': top5_accuracy,
        'predictions': all_preds,
        'labels': all_labels,
        'class_names': class_names
    }
    
    return results


def generar_matriz_confusion(results, save_path):
    """
    Genera y guarda la matriz de confusión
    """
    cm = confusion_matrix(results['labels'], results['predictions'])
    
    # Normalizar por fila (por clase real)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Encontrar las top 30 clases más comunes
    unique_labels = np.unique(results['labels'])
    class_counts = [(label, (results['labels'] == label).sum()) for label in unique_labels]
    class_counts.sort(key=lambda x: x[1], reverse=True)
    top_classes = [x[0] for x in class_counts[:30]]
    
    # Filtrar matriz para top clases
    cm_top = cm_normalized[np.ix_(top_classes, top_classes)]
    class_names_top = [results['class_names'][i] for i in top_classes]
    
    # Graficar
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm_top,
        annot=False,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names_top,
        yticklabels=class_names_top,
        cbar_kws={'label': 'Tasa de Clasificación'}
    )
    plt.xlabel('Predicción', fontsize=12)
    plt.ylabel('Etiqueta Real', fontsize=12)
    plt.title('Matriz de Confusión (Top 30 Clases)', fontsize=14, pad=20)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Matriz de confusión guardada en: {save_path}")


def analizar_errores(results, top_n=10):
    """
    Analiza los errores más comunes del modelo
    """
    preds = results['predictions']
    labels = results['labels']
    class_names = results['class_names']
    
    # Encontrar errores
    errors = preds != labels
    error_indices = np.where(errors)[0]
    
    # Contar pares (real, predicho)
    error_pairs = {}
    for idx in error_indices:
        real = labels[idx]
        pred = preds[idx]
        pair = (real, pred)
        error_pairs[pair] = error_pairs.get(pair, 0) + 1
    
    # Top errores
    top_errors = sorted(error_pairs.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    print(f"\n❌ Top {top_n} errores más comunes:")
    print("-" * 80)
    for i, ((real, pred), count) in enumerate(top_errors, 1):
        real_name = class_names[real]
        pred_name = class_names[pred]
        print(f"{i:2d}. Real: {real_name:25s} → Predicho: {pred_name:25s} ({count} veces)")


def accuracy_por_clase(results):
    """
    Calcula accuracy por cada clase
    """
    preds = results['predictions']
    labels = results['labels']
    class_names = results['class_names']
    
    unique_labels = np.unique(labels)
    
    class_accuracies = []
    for label in unique_labels:
        mask = labels == label
        if mask.sum() > 0:
            acc = (preds[mask] == labels[mask]).mean() * 100
            count = mask.sum()
            class_accuracies.append((label, acc, count))
    
    # Ordenar por accuracy
    class_accuracies.sort(key=lambda x: x[1], reverse=True)
    
    print("\n📊 Accuracy por clase:")
    print("-" * 80)
    print(f"{'Clase':<30s} {'Accuracy':>10s} {'Samples':>10s}")
    print("-" * 80)
    
    # Top 10 mejores
    print("\n✅ Top 10 mejores:")
    for label, acc, count in class_accuracies[:10]:
        name = class_names[label]
        print(f"{name:<30s} {acc:>9.2f}% {count:>10d}")
    
    # Top 10 peores
    print("\n❌ Top 10 peores:")
    for label, acc, count in class_accuracies[-10:]:
        name = class_names[label]
        print(f"{name:<30s} {acc:>9.2f}% {count:>10d}")
    
    return class_accuracies


def main():
    parser = argparse.ArgumentParser(description='Evaluar modelo de lengua de señas')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Ruta al modelo entrenado (.pth)')
    parser.add_argument('--model_type', type=str, default='lightweight',
                       choices=['r2plus1d', 'lightweight'],
                       help='Tipo de modelo')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Tamaño del batch')
    parser.add_argument('--num_frames', type=int, default=8,
                       help='Número de frames por video')
    parser.add_argument('--frame_size', type=int, default=112,
                       help='Tamaño de frames')
    
    args = parser.parse_args()
    
    print("="*60)
    print("📊 EVALUACIÓN DEL MODELO")
    print("="*60)
    print(f"   Modelo: {args.model_path}")
    print(f"   Tipo: {args.model_type}")
    
    # Device
    device = torch.device('cpu')
    print(f"   Device: {device}")
    
    # Cargar mapeo de clases
    with open('./splits/class_mapping.json', 'r') as f:
        class_mapping = json.load(f)
    num_classes = class_mapping['num_classes']
    idx_to_class = {int(k): v for k, v in class_mapping['idx_to_class'].items()}
    class_names = [idx_to_class[i] for i in range(num_classes)]
    
    print(f"   Número de clases: {num_classes}")
    
    # Crear dataloader de test
    print("\n🔄 Cargando test set...")
    _, _, test_loader = crear_dataloaders(
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        frame_size=(args.frame_size, args.frame_size),
        num_workers=2
    )
    
    print(f"   Test samples: {len(test_loader.dataset)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # Crear modelo
    print(f"\n🏗️  Creando modelo...")
    model = crear_modelo(
        model_type=args.model_type,
        num_classes=num_classes,
        pretrained=False
    )
    
    # Cargar checkpoint
    print(f"\n📥 Cargando checkpoint...")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    if 'val_acc' in checkpoint:
        print(f"   Val Accuracy del checkpoint: {checkpoint['val_acc']:.2f}%")
    if 'epoch' in checkpoint:
        print(f"   Época: {checkpoint['epoch']}")
    
    # Evaluar
    results = evaluar_modelo(model, test_loader, device, class_names)
    
    # Imprimir resultados
    print("\n" + "="*60)
    print("✅ RESULTADOS DE LA EVALUACIÓN")
    print("="*60)
    print(f"\n📊 Métricas Generales:")
    print(f"   Top-1 Accuracy: {results['accuracy']:.2f}%")
    print(f"   Top-5 Accuracy: {results['top5_accuracy']:.2f}%")
    print(f"   Total de muestras: {len(results['labels'])}")
    
    # Accuracy por clase
    class_accuracies = accuracy_por_clase(results)
    
    # Analizar errores
    analizar_errores(results, top_n=15)
    
    # Generar matriz de confusión
    print("\n📊 Generando visualizaciones...")
    output_dir = Path(args.model_path).parent / 'evaluation'
    output_dir.mkdir(exist_ok=True)
    
    cm_path = output_dir / 'confusion_matrix.png'
    generar_matriz_confusion(results, cm_path)
    
    # Guardar resultados
    results_json = {
        'model_path': args.model_path,
        'accuracy': float(results['accuracy']),
        'top5_accuracy': float(results['top5_accuracy']),
        'num_samples': len(results['labels']),
        'class_accuracies': [
            {
                'class': class_names[label],
                'accuracy': float(acc),
                'count': int(count)
            }
            for label, acc, count in class_accuracies
        ]
    }
    
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ Resultados guardados en: {results_path}")
    
    print("\n" + "="*60)
    print("✅ EVALUACIÓN COMPLETADA")
    print("="*60)
    
    print("\n🎯 PRÓXIMO PASO:")
    print("   Ejecuta: python step7_predecir.py --model_path <ruta_modelo> --video_path <ruta_video>")
    
    print("\n💡 INTERPRETACIÓN:")
    if results['accuracy'] >= 80:
        print("   🎉 Excelente! El modelo tiene muy buena precisión.")
    elif results['accuracy'] >= 70:
        print("   ✅ Bien! El modelo funciona correctamente.")
    elif results['accuracy'] >= 60:
        print("   ⚠️  Aceptable, pero hay margen de mejora.")
    else:
        print("   ❌ El accuracy es bajo. Considera:")
        print("      - Entrenar más épocas")
        print("      - Usar un modelo más grande (r2plus1d)")
        print("      - Aumentar datos de entrenamiento")
        print("      - Ajustar hiperparámetros")


if __name__ == '__main__':
    main()
