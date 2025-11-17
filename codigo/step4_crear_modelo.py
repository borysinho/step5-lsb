"""
PASO 4: Creación del Modelo
==============================
Este script define un modelo 3D CNN ligero optimizado para CPU.

ARQUITECTURA:
- R(2+1)D: Factoriza convoluciones 3D en (2D espacial + 1D temporal)
- Mucho más ligero que ResNet3D completo
- Pre-entrenado en Kinetics (opcional)

Para CPU usamos:
- R(2+1)D-18 (versión más pequeña)
- Menos capas que versiones grandes
- Factorización reduce parámetros significativamente

Ejecutar: python step4_crear_modelo.py
"""

import torch
import torch.nn as nn
import torchvision.models.video as video_models
import json
from pathlib import Path


class SignLanguageClassifier(nn.Module):
    """
    Clasificador de lengua de señas basado en R(2+1)D
    
    R(2+1)D es ideal para CPU porque:
    - Factoriza conv3D = conv2D + conv1D
    - Menos parámetros que ResNet3D
    - Mejor para videos cortos
    
    Parameters:
    -----------
    num_classes : int
        Número de clases/señas a predecir
    pretrained : bool
        Si True, usa pesos pre-entrenados en Kinetics
    dropout : float
        Probabilidad de dropout para regularización
    """
    
    def __init__(self, num_classes=71, pretrained=False, dropout=0.5):
        super(SignLanguageClassifier, self).__init__()
        
        print(f"🏗️  Construyendo modelo...")
        print(f"   Arquitectura: R(2+1)D-18")
        print(f"   Número de clases: {num_classes}")
        print(f"   Pre-entrenado: {pretrained}")
        print(f"   Dropout: {dropout}")
        
        # Cargar modelo base R(2+1)D
        # Nota: para evitar warnings con pretrained, usamos weights parameter
        if pretrained:
            print("   ⚠️  Cargando pesos pre-entrenados (puede tardar la primera vez)...")
            try:
                from torchvision.models.video import R2Plus1D_18_Weights
                self.model = video_models.r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
                print("   ✅ Pesos pre-entrenados cargados!")
            except:
                print("   ⚠️  No se pudieron cargar pesos pre-entrenados, usando aleatorios")
                self.model = video_models.r2plus1d_18(weights=None)
        else:
            self.model = video_models.r2plus1d_18(weights=None)
        
        # Modificar la capa final para nuestro número de clases
        # El modelo original tiene fc para 400 clases (Kinetics)
        in_features = self.model.fc.in_features
        
        # Reemplazar con nueva capa fully connected
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
        
        self.num_classes = num_classes
        
        # Contar parámetros
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n📊 Estadísticas del modelo:")
        print(f"   Total de parámetros: {total_params:,}")
        print(f"   Parámetros entrenables: {trainable_params:,}")
        print(f"   Tamaño aprox: {total_params * 4 / 1024 / 1024:.1f} MB")
    
    def forward(self, x):
        """
        Forward pass
        
        Parameters:
        -----------
        x : torch.Tensor
            Input de shape (batch, channels, frames, height, width)
            Ejemplo: (4, 3, 8, 112, 112)
        
        Returns:
        --------
        logits : torch.Tensor
            Output de shape (batch, num_classes)
        """
        return self.model(x)


class ImprovedLightweightCNN3D(nn.Module):
    """
    Modelo mejorado con más capacidad y regularización
    """

    def __init__(self, num_classes=71, dropout=0.3):
        super(ImprovedLightweightCNN3D, self).__init__()

        print(f"🏗️  Construyendo modelo MEJORADO...")
        print(f"   Arquitectura: 3D CNN Mejorada")
        print(f"   Número de clases: {num_classes}")

        # Bloque 1: 3 -> 32 filtros (más capacidad)
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        # Bloque 2: 32 -> 64 filtros
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        # Bloque 3: 64 -> 128 filtros (agregado)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        # Bloque 4: 128 -> 256 filtros (agregado)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn4 = nn.BatchNorm3d(256)
        self.pool4 = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Clasificador mejorado
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        self.num_classes = num_classes

        # Inicialización de pesos
        self._initialize_weights()

        # Contar parámetros
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n📊 Estadísticas del modelo:")
        print(f"   Total de parámetros: {total_params:,}")
        print(f"   Tamaño aprox: {total_params * 4 / 1024 / 1024:.1f} MB")

    def _initialize_weights(self):
        """Inicialización mejorada de pesos"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Bloque 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool1(x)

        # Bloque 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool2(x)

        # Bloque 3 (nuevo)
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool3(x)

        # Bloque 4 (nuevo)
        x = self.conv4(x)
        x = self.bn4(x)
        x = torch.relu(x)
        x = self.pool4(x)

        # Clasificador
        x = self.classifier(x)
        return x
        
        # Bloque 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool3(x)
        
        # Global pooling
        x = self.global_pool(x)
        
        # Clasificador
        x = self.classifier(x)
        
        return x


def crear_modelo(model_type='r2plus1d', num_classes=71, pretrained=False):
    """
    Factory para crear modelos
    
    Parameters:
    -----------
    model_type : str
        'r2plus1d': R(2+1)D-18 (RECOMENDADO)
        'lightweight': 3D CNN ultra ligero (más rápido, menos accuracy)
    num_classes : int
        Número de clases
    pretrained : bool
        Solo para 'r2plus1d': usar pesos pre-entrenados
    
    Returns:
    --------
    model : nn.Module
    """
    if model_type == 'r2plus1d':
        return SignLanguageClassifier(
            num_classes=num_classes,
            pretrained=pretrained
        )
    elif model_type == 'lightweight':
        return ImprovedLightweightCNN3D(num_classes=num_classes)
    else:
        raise ValueError(f"Tipo de modelo desconocido: {model_type}")


def test_modelo():
    """
    Función de prueba para verificar que el modelo funciona
    """
    print("="*60)
    print("🧪 PRUEBA DEL MODELO")
    print("="*60)
    
    # Cargar número de clases
    with open('./splits/class_mapping.json', 'r') as f:
        class_mapping = json.load(f)
    num_classes = class_mapping['num_classes']
    
    print(f"\n🎯 Número de clases detectadas: {num_classes}\n")
    
    # Probar R(2+1)D
    print("\n" + "="*60)
    print("OPCIÓN 1: R(2+1)D-18 (RECOMENDADO)")
    print("="*60)
    print("✅ Ventajas:")
    print("   - Arquitectura probada")
    print("   - Puede usar pesos pre-entrenados")
    print("   - Mejor accuracy esperado")
    print("❌ Desventajas:")
    print("   - Más lento en CPU (~31M parámetros)")
    print("   - Necesita más memoria\n")
    
    model_r2plus1d = crear_modelo('r2plus1d', num_classes=num_classes, pretrained=False)
    
    # Probar Lightweight
    print("\n" + "="*60)
    print("OPCIÓN 2: 3D CNN Lightweight (ALTERNATIVA)")
    print("="*60)
    print("✅ Ventajas:")
    print("   - MUY rápido en CPU (~150K parámetros)")
    print("   - Poca memoria")
    print("❌ Desventajas:")
    print("   - Accuracy menor")
    print("   - Sin pesos pre-entrenados\n")
    
    model_lightweight = crear_modelo('lightweight', num_classes=num_classes)
    
    # Test forward pass
    print("\n" + "="*60)
    print("🧪 Probando forward pass...")
    print("="*60)
    
    # Crear input de prueba (batch=2, channels=3, frames=8, height=112, width=112)
    dummy_input = torch.randn(2, 3, 8, 112, 112)
    print(f"\n📦 Input shape: {dummy_input.shape}")
    
    # Test R(2+1)D
    print("\n🔄 Probando R(2+1)D...")
    with torch.no_grad():
        output_r2plus1d = model_r2plus1d(dummy_input)
    print(f"   ✅ Output shape: {output_r2plus1d.shape}")
    print(f"   ✅ Esperado: ({dummy_input.shape[0]}, {num_classes})")
    
    # Test Lightweight
    print("\n🔄 Probando Lightweight...")
    with torch.no_grad():
        output_lightweight = model_lightweight(dummy_input)
    print(f"   ✅ Output shape: {output_lightweight.shape}")
    print(f"   ✅ Esperado: ({dummy_input.shape[0]}, {num_classes})")
    
    print("\n" + "="*60)
    print("✅ PRUEBA COMPLETADA - Ambos modelos funcionan!")
    print("="*60)
    
    print("\n💡 RECOMENDACIÓN PARA CPU + 1 SEMANA:")
    print("\n   Opción A (Mejor accuracy, más lento):")
    print("      - Usar R(2+1)D sin pre-training")
    print("      - Entrenar 20-30 épocas")
    print("      - Batch size = 2-4")
    print("      - Tiempo estimado: 6-10 horas\n")
    
    print("   Opción B (Más rápido, menor accuracy):")
    print("      - Usar Lightweight")
    print("      - Entrenar 40-50 épocas")
    print("      - Batch size = 4-8")
    print("      - Tiempo estimado: 2-4 horas\n")
    
    print("   💡 TIP: Comienza con Lightweight para probar el pipeline,")
    print("      luego entrena R(2+1)D cuando tengas todo funcionando.")
    
    print("\n🎯 PRÓXIMO PASO:")
    print("   Ejecuta: python step5_entrenar.py")


if __name__ == '__main__':
    test_modelo()
