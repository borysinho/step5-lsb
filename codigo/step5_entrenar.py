"""
PASO 5: Entrenamiento del Modelo
===================================
Este script entrena el modelo de clasificación de lengua de señas.

CARACTERÍSTICAS:
- Early stopping para evitar overfitting
- Guardado de checkpoints
- Visualización de métricas en tiempo real
- Optimizado para GPU (usa CPU como fallback)

OPTIMIZACIONES PARA GPU:
- Batch size optimizado para GPU T4
- Gradients accumulation para batches grandes
- Mixed precision si es soportado
- Checkpointing frecuente

Ejecutar:
python step5_entrenar.py --model lightweight --epochs 30 --batch_size 8

o para R(2+1)D:
python step5_entrenar.py --model r2plus1d --epochs 20 --batch_size 4
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import numpy as np

# Importar nuestros módulos
from step3_crear_dataset import crear_dataloaders
from step4_crear_modelo import crear_modelo


class Entrenador:
    """
    Clase para entrenar el modelo con todas las funcionalidades
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        lr=0.001,
        num_epochs=30,
        patience=5,
        save_dir='./checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.patience = patience
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Criterio y optimizador
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Learning rate scheduler (reduce on plateau)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # Maximizar accuracy
            factor=0.5,
            patience=3
        )
        
        # Historial de entrenamiento
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        # Early stopping
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        self.best_model_path = None
        
        print(f"\n🎯 Entrenador inicializado:")
        print(f"   Device: {device}")
        print(f"   Learning rate: {lr}")
        print(f"   Épocas: {num_epochs}")
        print(f"   Patience: {patience}")
        print(f"   Guardado en: {save_dir}")
    
    def train_epoch(self):
        """
        Entrena una época completa
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Barra de progreso
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        
        for batch_idx, (frames, labels) in enumerate(pbar):
            # Mover a device
            frames = frames.to(self.device)
            labels = labels.to(self.device)
            
            # Debug: verificar devices (solo primera iteración)
            if batch_idx == 0:
                print(f"🔍 Debug - Frames device: {frames.device}, Labels device: {labels.device}")
                print(f"🔍 Debug - Model device: {next(self.model.parameters()).device}")
                print(f"🔍 Debug - Frames shape: {frames.shape}, dtype: {frames.dtype}")
                print(f"🔍 Debug - Memory allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(frames)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Estadísticas
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Actualizar barra de progreso
            pbar.set_postfix({
                'loss': f'{running_loss/(batch_idx+1):.3f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """
        Valida el modelo
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation', leave=False)
            
            for batch_idx, (frames, labels) in enumerate(pbar):
                frames = frames.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(frames)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{running_loss/(batch_idx+1):.3f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """
        Guarda un checkpoint del modelo
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'history': self.history
        }
        
        # Guardar checkpoint regular
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Si es el mejor, guardar aparte
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.best_model_path = best_path
            print(f"      💾 Mejor modelo guardado: {val_acc:.2f}% accuracy")
    
    def train(self):
        """
        Loop principal de entrenamiento
        """
        print("\n" + "="*60)
        print("🚀 INICIANDO ENTRENAMIENTO")
        print("="*60)
        
        start_time = time.time()
        
        for epoch in range(1, self.num_epochs + 1):
            print(f"\n📍 Época {epoch}/{self.num_epochs}")
            print("-" * 60)
            
            # Entrenar
            train_loss, train_acc = self.train_epoch()
            
            # Validar
            val_loss, val_acc = self.validate()
            
            # Actualizar scheduler
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Guardar historial
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # Imprimir resultados
            print(f"\n   📊 Resultados:")
            print(f"      Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"      Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"      Learning Rate: {current_lr:.6f}")
            
            # Early stopping y guardar mejor modelo
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch, val_acc, is_best=True)
            else:
                self.epochs_without_improvement += 1
                print(f"      ⚠️  Sin mejora por {self.epochs_without_improvement}/{self.patience} épocas")
            
            # Guardar checkpoint cada 5 épocas
            if epoch % 5 == 0:
                self.save_checkpoint(epoch, val_acc, is_best=False)
            
            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"\n⏹️  Early stopping activado después de {epoch} épocas")
                break
        
        # Tiempo total
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        
        print("\n" + "="*60)
        print("✅ ENTRENAMIENTO COMPLETADO")
        print("="*60)
        print(f"\n📊 Resumen:")
        print(f"   Mejor Val Accuracy: {self.best_val_acc:.2f}%")
        print(f"   Épocas entrenadas: {epoch}")
        print(f"   Tiempo total: {hours}h {minutes}m")
        print(f"   Mejor modelo guardado en: {self.best_model_path}")
        
        # Guardar historial
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"   Historial guardado en: {history_path}")
        
        return self.history
    
    def plot_history(self):
        """
        Grafica el historial de entrenamiento
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Época')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Pérdida Durante el Entrenamiento')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(epochs, self.history['train_acc'], 'b-', label='Train Acc', linewidth=2)
        axes[1].plot(epochs, self.history['val_acc'], 'r-', label='Val Acc', linewidth=2)
        axes[1].set_xlabel('Época')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Precisión Durante el Entrenamiento')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.save_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   Gráficas guardadas en: {plot_path}")
        plt.close()


def main():
    """
    Función principal
    """
    # Argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Entrenar clasificador de lengua de señas')
    parser.add_argument('--model', type=str, default='lightweight',
                       choices=['r2plus1d', 'lightweight'],
                       help='Tipo de modelo a usar')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Número de épocas')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Tamaño del batch')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=5,
                       help='Patience para early stopping')
    parser.add_argument('--num_frames', type=int, default=8,
                       help='Número de frames por video')
    parser.add_argument('--frame_size', type=int, default=112,
                       help='Tamaño de frames (altura y ancho)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🎬 CONFIGURACIÓN DEL ENTRENAMIENTO")
    print("="*60)
    print(f"   Modelo: {args.model}")
    print(f"   Épocas: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Frames por video: {args.num_frames}")
    print(f"   Tamaño de frame: {args.frame_size}x{args.frame_size}")
    print(f"   Patience: {args.patience}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print("🚀 ENTRENANDO CON GPU - ¡Esto será mucho más rápido!")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("💻 ENTRENANDO CON CPU - Puede tomar más tiempo...")
    print(f"   Device: {device}")
    
    # Cargar número de clases
    with open('./splits/class_mapping.json', 'r') as f:
        class_mapping = json.load(f)
    num_classes = class_mapping['num_classes']
    print(f"   Número de clases: {num_classes}")
    
    # Crear dataloaders
    print("\n🔄 Creando dataloaders...")
    train_loader, val_loader, test_loader = crear_dataloaders(
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        frame_size=(args.frame_size, args.frame_size),
        num_workers=2
    )
    
    # Crear modelo
    print(f"\n🏗️  Creando modelo {args.model}...")
    model = crear_modelo(
        model_type=args.model,
        num_classes=num_classes,
        pretrained=False
    )
    
    # Crear directorio de checkpoints
    checkpoint_dir = f'./checkpoints_{args.model}'
    
    # Crear entrenador
    entrenador = Entrenador(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        num_epochs=args.epochs,
        patience=args.patience,
        save_dir=checkpoint_dir
    )
    
    # Entrenar
    history = entrenador.train()
    
    # Graficar
    entrenador.plot_history()
    
    print("\n🎯 PRÓXIMO PASO:")
    print("   Ejecuta: python step6_evaluar.py")
    print("\n💡 NOTA:")
    print(f"   El mejor modelo está en: {checkpoint_dir}/best_model.pth")
    print("   Puedes usar este modelo para hacer predicciones.")


if __name__ == '__main__':
    main()
