"""
PASO 7: Predicción en Videos Nuevos
=====================================
Este script clasifica un video individual y muestra:
- Top-5 predicciones con probabilidades
- Nombre de la clase predicha
- Confianza del modelo

Es la BASE para el sistema de streaming futuro.

Ejecutar:
python step7_predecir.py --model_path checkpoints_lightweight/best_model.pth --video_path ../videos/1/SALUDOS/HOLA.mp4
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import json
import argparse
from pathlib import Path

# Importar nuestros módulos
from step4_crear_modelo import crear_modelo


def cargar_video(video_path, num_frames=8, frame_size=(112, 112)):
    """
    Carga y procesa un video para predicción
    
    Similar a VideoDataset._load_video pero standalone
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video_path}")
    
    # Obtener información del video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calcular índices de frames
    if total_frames < num_frames:
        indices = list(range(total_frames)) + [total_frames - 1] * (num_frames - total_frames)
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, frame_size)
            frames.append(frame)
        else:
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((*frame_size, 3), dtype=np.uint8))
    
    cap.release()
    
    # Convertir a tensor
    frames = np.array(frames)
    frames = frames.astype(np.float32) / 255.0
    frames = torch.from_numpy(frames).permute(3, 0, 1, 2)  # (C, T, H, W)
    
    # Normalizar con ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
    frames = (frames - mean) / std
    
    # Agregar dimensión de batch
    frames = frames.unsqueeze(0)  # (1, C, T, H, W)
    
    return frames


def predecir(model, video_tensor, device, class_names, top_k=5):
    """
    Realiza predicción en un video
    
    Returns:
    --------
    predictions : list of tuples
        Lista de (class_name, probability) ordenadas por probabilidad
    """
    model.eval()
    
    with torch.no_grad():
        video_tensor = video_tensor.to(device)
        outputs = model(video_tensor)
        
        # Aplicar softmax para obtener probabilidades
        probs = F.softmax(outputs, dim=1)
        
        # Top-k predicciones
        top_probs, top_indices = probs.topk(top_k, dim=1)
        
        # Convertir a lista
        predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            class_name = class_names[idx.item()]
            probability = prob.item() * 100
            predictions.append((class_name, probability))
    
    return predictions


def visualizar_prediccion(video_path, predictions, mostrar_video=False):
    """
    Muestra las predicciones de forma visual
    """
    print("\n" + "="*60)
    print("🎬 VIDEO ANALIZADO")
    print("="*60)
    print(f"   Ruta: {video_path}")
    
    # Obtener info del video
    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        print(f"   Duración: {duration:.2f}s")
        print(f"   Resolución: {width}x{height}")
        print(f"   FPS: {fps:.2f}")
        print(f"   Frames: {frame_count}")
        cap.release()
    
    print("\n" + "="*60)
    print("🎯 PREDICCIONES")
    print("="*60)
    
    # Top predicción
    top_class, top_prob = predictions[0]
    print(f"\n✅ PREDICCIÓN: {top_class}")
    print(f"   Confianza: {top_prob:.2f}%")
    
    # Barra visual de confianza
    bar_length = 40
    filled = int(bar_length * top_prob / 100)
    bar = "█" * filled + "░" * (bar_length - filled)
    print(f"   {bar} {top_prob:.1f}%")
    
    # Top-5
    print(f"\n📊 Top-5 Predicciones:")
    print("-" * 60)
    for i, (class_name, prob) in enumerate(predictions, 1):
        # Barra proporcional
        bar_length = 30
        filled = int(bar_length * prob / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        print(f"{i}. {class_name:<30s} {bar} {prob:>6.2f}%")
    
    print("-" * 60)
    
    # Interpretación
    print("\n💡 INTERPRETACIÓN:")
    if top_prob >= 90:
        print("   🎉 Confianza MUY ALTA - Predicción muy segura")
    elif top_prob >= 70:
        print("   ✅ Confianza ALTA - Predicción confiable")
    elif top_prob >= 50:
        print("   ⚠️  Confianza MEDIA - Revisar top-5")
    else:
        print("   ❌ Confianza BAJA - El modelo no está seguro")
        print("      Posibles razones:")
        print("      - Seña no está en el dataset")
        print("      - Video de baja calidad")
        print("      - Seña ejecutada de forma diferente")


def main():
    parser = argparse.ArgumentParser(description='Predecir lengua de señas en un video')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Ruta al modelo entrenado (.pth)')
    parser.add_argument('--video_path', type=str, required=True,
                       help='Ruta al video a clasificar')
    parser.add_argument('--model_type', type=str, default='lightweight',
                       choices=['r2plus1d', 'lightweight'],
                       help='Tipo de modelo')
    parser.add_argument('--num_frames', type=int, default=8,
                       help='Número de frames por video')
    parser.add_argument('--frame_size', type=int, default=112,
                       help='Tamaño de frames')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Número de predicciones top a mostrar')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🔮 PREDICCIÓN DE LENGUA DE SEÑAS")
    print("="*60)
    print(f"   Modelo: {args.model_path}")
    print(f"   Video: {args.video_path}")
    
    # Verificar que existe el video
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"\n❌ Error: El video no existe: {video_path}")
        return
    
    # Device
    device = torch.device('cpu')
    print(f"   Device: {device}")
    
    # Cargar mapeo de clases
    print("\n📂 Cargando configuración...")
    with open('./splits/class_mapping.json', 'r') as f:
        class_mapping = json.load(f)
    num_classes = class_mapping['num_classes']
    idx_to_class = {int(k): v for k, v in class_mapping['idx_to_class'].items()}
    class_names = [idx_to_class[i] for i in range(num_classes)]
    
    print(f"   Clases disponibles: {num_classes}")
    
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
        print(f"   Accuracy del modelo: {checkpoint['val_acc']:.2f}%")
    
    # Cargar y procesar video
    print(f"\n🎬 Procesando video...")
    try:
        video_tensor = cargar_video(
            video_path,
            num_frames=args.num_frames,
            frame_size=(args.frame_size, args.frame_size)
        )
        print(f"   ✅ Video cargado: {video_tensor.shape}")
    except Exception as e:
        print(f"\n❌ Error al cargar video: {e}")
        return
    
    # Predecir
    print(f"\n🔮 Realizando predicción...")
    predictions = predecir(model, video_tensor, device, class_names, top_k=args.top_k)
    
    # Visualizar resultados
    visualizar_prediccion(video_path, predictions)
    
    print("\n" + "="*60)
    print("✅ PREDICCIÓN COMPLETADA")
    print("="*60)
    
    print("\n🚀 PRÓXIMOS PASOS PARA STREAMING:")
    print("\n   1. Procesar video en ventanas deslizantes")
    print("      - Ejemplo: ventana de 2 segundos")
    print("      - Avanzar cada 0.5 segundos")
    print("\n   2. Implementar buffer de frames")
    print("      - Acumular frames en tiempo real")
    print("      - Clasificar cuando tengamos suficientes")
    print("\n   3. Suavizar predicciones")
    print("      - Usar votación mayoritaria")
    print("      - Filtrar predicciones de baja confianza")
    print("\n   4. Optimizar para tiempo real")
    print("      - TorchScript para acelerar")
    print("      - Procesamiento asíncrono")
    
    print("\n💡 CONSEJO:")
    print("   Este script es la BASE para streaming.")
    print("   Ya tienes la función de predicción lista.")
    print("   Solo falta agregar el manejo de ventanas temporales.")


if __name__ == '__main__':
    main()
