"""
PASO 3: Implementación del Dataset de PyTorch
================================================
Este script crea una clase VideoDataset que:
1. Carga videos desde las rutas del CSV
2. Extrae frames uniformemente espaciados
3. Aplica transformaciones (resize, normalización)
4. Retorna tensores listos para el modelo

OPTIMIZADO PARA CPU:
- Usa solo 8 frames por video (en vez de 32)
- Resolución 112x112 (en vez de 224x224)
- Sin data augmentation pesado

Ejecutar: python step3_crear_dataset.py
"""

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from torchvision import transforms
import json

class VideoDataset(Dataset):
    """
    Dataset para videos de lengua de señas
    
    Parámetros:
    -----------
    csv_file : str o Path
        Ruta al archivo CSV con los datos (train.csv, val.csv o test.csv)
    video_root : str o Path
        Ruta raíz donde están los videos
    num_frames : int
        Número de frames a extraer por video (default: 8 para CPU)
    frame_size : tuple
        Tamaño de los frames (height, width) (default: 112x112 para CPU)
    transform : torchvision.transforms
        Transformaciones a aplicar (default: None)
    is_train : bool
        Si es True, aplica data augmentation (default: False)
    """
    
    def __init__(
        self,
        csv_file,
        video_root,
        num_frames=8,
        frame_size=(112, 112),
        transform=None,
        is_train=False
    ):
        self.df = pd.read_csv(csv_file)
        self.video_root = Path(video_root)
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.transform = transform
        self.is_train = is_train
        
        print(f"📂 Dataset cargado: {len(self.df)} videos")
        print(f"   Frames por video: {self.num_frames}")
        print(f"   Tamaño de frame: {self.frame_size}")
        print(f"   Modo entrenamiento: {self.is_train}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Retorna un video procesado y su etiqueta
        
        Returns:
        --------
        frames : torch.Tensor
            Tensor de shape (C, T, H, W) = (3, num_frames, height, width)
        label : int
            Índice de la clase
        """
        # Obtener información del video
        row = self.df.iloc[idx]
        video_path = self.video_root / row['video_path']
        label = row['label']
        
        # Cargar frames del video
        frames = self._load_video(video_path)
        
        # Aplicar transformaciones si existen
        if self.transform:
            frames = self.transform(frames)
        
        return frames, label
    
    def _load_video(self, video_path):
        """
        Carga un video y extrae frames uniformemente espaciados
        
        Parameters:
        -----------
        video_path : Path
            Ruta al archivo de video
            
        Returns:
        --------
        frames : torch.Tensor
            Tensor de shape (C, T, H, W)
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")
        
        # Obtener información del video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calcular índices de frames a extraer (uniformemente espaciados)
        if total_frames < self.num_frames:
            # Si el video tiene menos frames, repetir el último
            indices = list(range(total_frames)) + [total_frames - 1] * (self.num_frames - total_frames)
        else:
            # Extraer frames uniformemente
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            
            if ret:
                # Convertir BGR a RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize
                frame = cv2.resize(frame, self.frame_size)
                
                frames.append(frame)
            else:
                # Si falla, usar el último frame válido
                if frames:
                    frames.append(frames[-1])
                else:
                    # Si es el primer frame y falla, usar un frame negro
                    frames.append(np.zeros((*self.frame_size, 3), dtype=np.uint8))
        
        cap.release()
        
        # Convertir a numpy array: (T, H, W, C)
        frames = np.array(frames)
        
        # Normalizar a [0, 1]
        frames = frames.astype(np.float32) / 255.0
        
        # Transponer a (C, T, H, W) - formato PyTorch para videos
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2)
        
        # Normalizar con ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        frames = (frames - mean) / std
        
        return frames
    
    def get_class_name(self, label_idx):
        """
        Obtiene el nombre de la clase dado su índice
        """
        row = self.df[self.df['label'] == label_idx].iloc[0]
        return row['categoria']


def crear_dataloaders(
    train_csv='./splits/train.csv',
    val_csv='./splits/val.csv',
    test_csv='./splits/test.csv',
    video_root='../videos',  # Corregido: la ruta correcta desde 'codigo'
    batch_size=4,  # Pequeño para CPU
    num_frames=8,  # Pocos frames para CPU
    frame_size=(112, 112),  # Resolución baja para CPU
    num_workers=2  # Pocos workers para CPU
):
    """
    Crea los DataLoaders para train, val y test
    
    Parameters:
    -----------
    batch_size : int
        Tamaño del batch (pequeño para CPU, default: 4)
    num_workers : int
        Número de procesos para cargar datos (default: 2 para CPU)
        
    Returns:
    --------
    train_loader, val_loader, test_loader : DataLoader
    """
    print("="*60)
    print("🔄 Creando DataLoaders...")
    print("="*60)
    
    # Crear datasets
    print("\n📊 TRAIN:")
    train_dataset = VideoDataset(
        csv_file=train_csv,
        video_root=video_root,
        num_frames=num_frames,
        frame_size=frame_size,
        is_train=True
    )
    
    print("\n📊 VALIDATION:")
    val_dataset = VideoDataset(
        csv_file=val_csv,
        video_root=video_root,
        num_frames=num_frames,
        frame_size=frame_size,
        is_train=False
    )
    
    print("\n📊 TEST:")
    test_dataset = VideoDataset(
        csv_file=test_csv,
        video_root=video_root,
        num_frames=num_frames,
        frame_size=frame_size,
        is_train=False
    )
    
    # Crear dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False  # No necesario para CPU
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    print("\n" + "="*60)
    print("✅ DataLoaders creados exitosamente")
    print("="*60)
    print(f"\n📊 Configuración:")
    print(f"   Batch size: {batch_size}")
    print(f"   Num workers: {num_workers}")
    print(f"   Frames por video: {num_frames}")
    print(f"   Tamaño de frame: {frame_size}")
    print(f"\n   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


def test_dataloader():
    """
    Función de prueba para verificar que el DataLoader funciona correctamente
    """
    print("="*60)
    print("🧪 PRUEBA DEL DATALOADER")
    print("="*60)
    
    # Crear dataloaders
    train_loader, val_loader, test_loader = crear_dataloaders(
        batch_size=2,
        num_frames=8,
        frame_size=(112, 112)
    )
    
    # Cargar mapeo de clases
    with open('./splits/class_mapping.json', 'r') as f:
        class_mapping = json.load(f)
    idx_to_class = {int(k): v for k, v in class_mapping['idx_to_class'].items()}
    
    # Obtener un batch
    print("\n📦 Cargando un batch de prueba...")
    frames, labels = next(iter(train_loader))
    
    print(f"\n✅ Batch cargado exitosamente!")
    print(f"\n📊 Información del batch:")
    print(f"   Shape de frames: {frames.shape}")
    print(f"   - Batch size: {frames.shape[0]}")
    print(f"   - Canales: {frames.shape[1]} (RGB)")
    print(f"   - Frames temporales: {frames.shape[2]}")
    print(f"   - Alto: {frames.shape[3]}")
    print(f"   - Ancho: {frames.shape[4]}")
    print(f"\n   Shape de labels: {labels.shape}")
    print(f"   Labels: {labels.tolist()}")
    print(f"\n   Clases en este batch:")
    for i, label in enumerate(labels):
        print(f"      Video {i+1}: {idx_to_class[label.item()]}")
    
    print(f"\n   Rango de valores de frames:")
    print(f"      Min: {frames.min():.3f}")
    print(f"      Max: {frames.max():.3f}")
    print(f"      Mean: {frames.mean():.3f}")
    
    print("\n" + "="*60)
    print("✅ PRUEBA COMPLETADA - El DataLoader funciona correctamente!")
    print("="*60)
    
    print("\n🎯 PRÓXIMO PASO:")
    print("   Ejecuta: python step4_crear_modelo.py")
    print("\n💡 NOTA:")
    print("   El DataLoader está optimizado para CPU.")
    print("   Si tienes GPU disponible, puedes aumentar:")
    print("   - batch_size (de 4 a 16-32)")
    print("   - num_frames (de 8 a 16-32)")
    print("   - frame_size (de 112x112 a 224x224)")


if __name__ == '__main__':
    test_dataloader()
