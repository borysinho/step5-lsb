# Configuración del Proyecto
# Modifica estos valores según tus necesidades

# =============================================================================
# CONFIGURACIÓN DEL DATASET
# =============================================================================
VIDEO_ROOT = '../videos'
SPLITS_DIR = './splits'
TRAIN_CSV = './splits/train.csv'
VAL_CSV = './splits/val.csv'
TEST_CSV = './splits/test.csv'
CLASS_MAPPING = './splits/class_mapping.json'

# División del dataset
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_SEED = 42

# Filtrado de clases
MIN_SAMPLES_PER_CLASS = 10  # Clases con menos muestras se eliminan

# =============================================================================
# CONFIGURACIÓN DE VIDEO PROCESSING
# =============================================================================

# Para CPU (RECOMENDADO)
NUM_FRAMES_CPU = 8              # Frames a extraer por video
FRAME_SIZE_CPU = (112, 112)     # Resolución (height, width)
BATCH_SIZE_CPU = 4              # Batch size pequeño
NUM_WORKERS_CPU = 2             # Workers para DataLoader

# Para GPU (si tienes disponible)
NUM_FRAMES_GPU = 16
FRAME_SIZE_GPU = (224, 224)
BATCH_SIZE_GPU = 16
NUM_WORKERS_GPU = 4

# =============================================================================
# CONFIGURACIÓN DE MODELO
# =============================================================================

# Modelos disponibles: 'lightweight', 'r2plus1d'
DEFAULT_MODEL = 'lightweight'

# Hiperparámetros
LEARNING_RATE = 0.001
DROPOUT = 0.5

# =============================================================================
# CONFIGURACIÓN DE ENTRENAMIENTO
# =============================================================================

# Para modelo Lightweight
EPOCHS_LIGHTWEIGHT = 30
PATIENCE_LIGHTWEIGHT = 5

# Para modelo R(2+1)D
EPOCHS_R2PLUS1D = 20
PATIENCE_R2PLUS1D = 5

# Directorios de checkpoints
CHECKPOINT_DIR_LIGHTWEIGHT = './checkpoints_lightweight'
CHECKPOINT_DIR_R2PLUS1D = './checkpoints_r2plus1d'

# Frecuencia de guardado
SAVE_CHECKPOINT_EVERY = 5  # Guardar cada N épocas

# =============================================================================
# CONFIGURACIÓN DE EVALUACIÓN
# =============================================================================

TOP_K_PREDICTIONS = 5  # Top-K para accuracy
TOP_N_ERRORS = 15      # Top errores a mostrar

# =============================================================================
# RUTAS DE OUTPUT
# =============================================================================

ANALYSIS_DIR = './analisis_dataset'
EVALUATION_DIR = './evaluation'

# =============================================================================
# TIPS PARA MODIFICAR
# =============================================================================

"""
PARA ENTRENAR MÁS RÁPIDO (menor accuracy):
- DEFAULT_MODEL = 'lightweight'
- EPOCHS_LIGHTWEIGHT = 20
- BATCH_SIZE_CPU = 8
- NUM_FRAMES_CPU = 6

PARA MEJOR ACCURACY (más lento):
- DEFAULT_MODEL = 'r2plus1d'
- EPOCHS_R2PLUS1D = 30
- BATCH_SIZE_CPU = 2
- NUM_FRAMES_CPU = 12

SI TIENES POCA RAM:
- BATCH_SIZE_CPU = 2
- NUM_WORKERS_CPU = 0
- NUM_FRAMES_CPU = 6
- FRAME_SIZE_CPU = (96, 96)

SI QUIERES EXPERIMENTAR:
- Aumenta LEARNING_RATE a 0.01 (más rápido pero menos estable)
- Reduce LEARNING_RATE a 0.0001 (más lento pero más estable)
- Aumenta DROPOUT a 0.7 (más regularización)
- Reduce PATIENCE para terminar antes
"""
