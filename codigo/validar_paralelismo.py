#!/usr/bin/env python3
"""
SCRIPT DE VALIDACIÓN DE PARALELISMO
=====================================
Este script valida que el paralelismo se configure correctamente
tanto para CPU como para GPU.

Ejecutar: python validar_paralelismo.py
"""

import torch
import multiprocessing
import time
from step3_crear_dataset import crear_dataloaders

def validar_paralelismo():
    """
    Valida la configuración de paralelismo
    """
    print("="*70)
    print("🔍 VALIDACIÓN DE PARALELISMO EN DATALOADERS")
    print("="*70)

    # Detectar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Dispositivo detectado: {device}")

    if device.type == 'cuda':
        print("🚀 MODO GPU:")
        print(f"   GPU disponible: {torch.cuda.get_device_name(0)}")
        print(f"   Memoria total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"   GPUs disponibles: {torch.cuda.device_count()}")
        expected_workers = min(4, torch.cuda.device_count() * 2)
        print(f"   Workers esperados: {expected_workers}")
    else:
        print("💻 MODO CPU:")
        cpu_count = multiprocessing.cpu_count()
        print(f"   Núcleos de CPU: {cpu_count}")
        expected_workers = min(8, cpu_count // 2)
        print(f"   Workers esperados: {expected_workers}")

    print("\n🔄 Creando DataLoaders con configuración automática...")
    # Crear dataloaders con configuración automática
    try:
        train_loader, val_loader, test_loader = crear_dataloaders(
            batch_size=4,
            num_frames=8,
            frame_size=(112, 112),
            device=device
        )
        print("✅ DataLoaders creados exitosamente")
    except Exception as e:
        print(f"❌ Error al crear DataLoaders: {e}")
        return False

    # Verificar configuración
    print("\n🔧 Verificando configuración:")
    print(f"   Train loader workers: {train_loader.num_workers}")
    print(f"   Val loader workers: {val_loader.num_workers}")
    print(f"   Test loader workers: {test_loader.num_workers}")
    print(f"   Pin memory (GPU): {train_loader.pin_memory}")

    # Validar que los workers sean correctos
    actual_workers = train_loader.num_workers
    if device.type == 'cuda':
        expected_workers = min(4, torch.cuda.device_count() * 2) if torch.cuda.is_available() else 2
    else:
        expected_workers = min(8, multiprocessing.cpu_count() // 2)

    if actual_workers == expected_workers:
        print(f"✅ Configuración correcta: {actual_workers} workers")
    else:
        print(f"⚠️  Configuración inesperada: {actual_workers} workers (esperados: {expected_workers})")

    # Probar carga de datos con timing
    print("\n⏱️  Probando carga de datos...")
    start_time = time.time()

    try:
        # Cargar un batch de entrenamiento
        frames, labels = next(iter(train_loader))
        load_time = time.time() - start_time

        print("✅ Batch cargado exitosamente")
        print(f"   Tiempo de carga: {load_time:.3f} segundos")
        print(f"   Shape del batch: {frames.shape}")
        print(f"   Tipo de datos: {frames.dtype}")
        print(f"   Device del tensor: {frames.device}")

        # Verificar paralelismo en acción
        if actual_workers > 0:
            print(f"🎯 Paralelismo activo: {actual_workers} procesos trabajando")
            if device.type == 'cpu':
                print("   💡 CPU: Los workers están procesando videos en paralelo")
            else:
                print("   💡 GPU: Workers optimizados para transferencia GPU")
        else:
            print("⚠️  Sin paralelismo: num_workers = 0")
            if device.type == 'cpu':
                print("   💡 Sugerencia: Considera aumentar num_workers para CPU")

    except Exception as e:
        print(f"❌ Error al cargar batch: {e}")
        return False

    print("\n" + "="*70)
    print("✅ VALIDACIÓN COMPLETADA EXITOSAMENTE")
    print("="*70)

    if device.type == 'cpu' and actual_workers > 0:
        print("🎉 ¡PARALELISMO EN CPU CONFIGURADO CORRECTAMENTE!")
        print(f"   {actual_workers} procesos trabajando en paralelo para cargar datos")
    elif device.type == 'cuda':
        print("🎉 ¡CONFIGURACIÓN GPU OPTIMIZADA!")
        print(f"   {actual_workers} workers configurados para GPU")

    return True

def probar_sin_paralelismo():
    """
    Prueba de comparación sin paralelismo
    """
    print("\n🔄 Probando sin paralelismo (num_workers=0)...")
    device = torch.device('cpu')  # Forzar CPU para comparación

    start_time = time.time()
    train_loader, _, _ = crear_dataloaders(
        batch_size=4,
        num_frames=8,
        frame_size=(112, 112),
        device=device
    )
    # Forzar num_workers=0 para comparación
    train_loader.num_workers = 0

    try:
        frames, labels = next(iter(train_loader))
        load_time_no_parallel = time.time() - start_time
        print(f"   Tiempo sin paralelismo: {load_time_no_parallel:.3f} segundos")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # Comparar con paralelismo
    print("\n🔄 Probando con paralelismo...")
    start_time = time.time()
    train_loader_parallel, _, _ = crear_dataloaders(
        batch_size=4,
        num_frames=8,
        frame_size=(112, 112),
        device=device
    )

    try:
        frames, labels = next(iter(train_loader_parallel))
        load_time_parallel = time.time() - start_time
        print(f"   Tiempo con paralelismo: {load_time_parallel:.3f} segundos")
        speedup = load_time_no_parallel / load_time_parallel
        print(f"   🚀 Speedup: {speedup:.2f}x más rápido")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    print("🚀 Iniciando validación de paralelismo...")

    # Validación principal
    success = validar_paralelismo()

    if success:
        # Solo probar comparación si estamos en CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cpu':
            probar_sin_paralelismo()

    print("\n🎯 PRÓXIMO PASO:")
    print("   Ejecuta: python step5_entrenar.py")
    print("\n💡 NOTA:")
    print("   El paralelismo ahora se configura automáticamente según el dispositivo")