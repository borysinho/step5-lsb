"""
PASO 3: Implementación del Dataset de PyTorch
================================================
Este script crea una clase VideoDataset que:
1. Carga videos desde las rutas del CSV
2. Extrae frames uniformemente espaciados
3. Aplica transformaciones (resize, normalización)
4. Retorna tensores listos para el modelo

OPTIMIZADO PARA CPU Y GPU:
- Configuración automática de paralelismo (num_workers)
- Usa 8 frames por video (configurable)
- Resolución 112x112 (configurable)
- Data augmentation para entrenamiento

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

        # Aplicar data augmentation si está en modo entrenamiento
        if self.is_train:
            frames = self._apply_data_augmentation(frames)

        return frames, label
        if self.is_train:
            self.transform = transforms.Compose([
                # Data augmentation para training
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            ])
        else:
            self.transform = None
        
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
    
    def _apply_data_augmentation(self, frames):
        """
        Aplica data augmentation a nivel de video completo
        
        Parameters:
        -----------
        frames : torch.Tensor
            Tensor de shape (C, T, H, W)
            
        Returns:
        --------
        frames : torch.Tensor
            Tensor con augmentación aplicada
        """
        # frames shape: (C, T, H, W) = (3, 8, 112, 112)
        
        # Random horizontal flip (50% probability)
        if torch.rand(1) < 0.5:
            frames = torch.flip(frames, dims=[3])  # Flip width dimension
        
        # Random brightness/contrast adjustment
        if torch.rand(1) < 0.3:  # 30% probability
            brightness_factor = torch.rand(1) * 0.4 + 0.8  # 0.8 to 1.2
            frames = torch.clamp(frames * brightness_factor, 0, 1)
        
        # Random temporal jitter (shuffle frame order slightly)
        if torch.rand(1) < 0.2:  # 20% probability
            # Shuffle frames within a small window
            perm = torch.randperm(frames.shape[1])
            frames = frames[:, perm, :, :]
        
        return frames
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
    batch_size=8,  # Optimizado para GPU
    num_frames=8,  # Pocos frames para eficiencia
    frame_size=(112, 112),  # Resolución baja para velocidad
    device=None  # Detectar automáticamente el dispositivo
):
    """
    Crea los DataLoaders para train, val y test
    
    Parameters:
    -----------
    batch_size : int
        Tamaño del batch (optimizado para GPU, default: 8)
    device : torch.device or None
        Dispositivo para determinar num_workers (None para autodetección)
        
    Returns:
    --------
    train_loader, val_loader, test_loader : DataLoader
    """
    print("="*60)
    print("🔄 Creando DataLoaders...")
    print("="*60)
    
    # Detectar dispositivo si no se proporciona
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configurar num_workers basado en el dispositivo
    if device.type == 'cuda':
        # Para GPU: usar pocos workers para evitar bottleneck
        num_workers = min(4, torch.cuda.device_count() * 2) if torch.cuda.is_available() else 2
        pin_memory = True
        print(f"🚀 Configuración GPU detectada - Usando {num_workers} workers")
    else:
        # Para CPU: usar más workers para paralelismo
        import multiprocessing
        num_workers = min(8, multiprocessing.cpu_count() // 2)
        pin_memory = False
        print(f"💻 Configuración CPU detectada - Usando {num_workers} workers para paralelismo")
    
    print(f"   Device: {device}")
    print(f"   Num workers: {num_workers}")
    print(f"   Pin memory: {pin_memory}")
    
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
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
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
    
    # Detectar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Crear dataloaders
    train_loader, val_loader, test_loader = crear_dataloaders(
        batch_size=2,
        num_frames=8,
        frame_size=(112, 112),
        device=device
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
    print("   El DataLoader está optimizado para GPU.")
    print("   Si tienes GPU disponible, puedes aumentar:")
    print("   - batch_size (de 4 a 16-32)")
    print("   - num_frames (de 8 a 16-32)")
    print("   - frame_size (de 112x112 a 224x224)")


if __name__ == '__main__':
    test_dataloader()
