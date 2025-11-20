"""
Konfigurationsdatei für YOLOv8 Webcam Detection
================================================
Hier können alle wichtigen Parameter zentral angepasst werden.
"""

# ============================================================================
# MODEL KONFIGURATION
# ============================================================================

# YOLOv8 Model auswählen
# Optionen: 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'
# n = Nano (schnellstes)
# s = Small
# m = Medium
# l = Large
# x = Extra Large (beste Genauigkeit)
MODEL_NAME = 'yolov8n.pt'

# Minimum Konfidenz für Detektionen (0.0 - 1.0)
# Höhere Werte = weniger False Positives, aber auch weniger Detektionen
# Niedrigere Werte = mehr Detektionen, aber auch mehr False Positives
CONFIDENCE_THRESHOLD = 0.5

# ============================================================================
# WEBCAM KONFIGURATION
# ============================================================================

# Webcam Index
# 0 = Standard Webcam
# 1, 2, 3... = Externe Webcams
CAMERA_INDEX = 0

# Webcam Auflösung
# Kleinere Auflösung = bessere Performance
# Größere Auflösung = bessere Bildqualität
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ============================================================================
# VISUALISIERUNG
# ============================================================================

# Box-Linienstärke (in Pixeln)
BOX_THICKNESS = 2

# Text-Schriftgröße
TEXT_SCALE = 0.6
TEXT_THICKNESS = 2

# Info-Overlay Position
INFO_POSITION_X = 10
INFO_POSITION_Y = 10
INFO_WIDTH = 250
INFO_HEIGHT = 80

# ============================================================================
# PERFORMANCE
# ============================================================================

# FPS Berechnungs-Intervall (in Frames)
# Höhere Werte = stabilerer FPS Counter
FPS_UPDATE_INTERVAL = 10

# ============================================================================
# SCREENSHOTS
# ============================================================================

# Screenshot Ordner
SCREENSHOT_FOLDER = "screenshots"

# Screenshot Format
SCREENSHOT_FORMAT = "jpg"  # jpg, png

# Screenshot Qualität (nur für jpg, 0-100)
SCREENSHOT_QUALITY = 95

# ============================================================================
# ADVANCED SETTINGS
# ============================================================================

# Zeige nur bestimmte Klassen (leer = alle Klassen)
# Beispiel: FILTER_CLASSES = ['person', 'car', 'dog']
FILTER_CLASSES = []

# Maximale Anzahl Detektionen pro Frame (0 = unbegrenzt)
MAX_DETECTIONS = 0

# ============================================================================
# DEBUGGING
# ============================================================================

# Verbose Mode (zeigt mehr Details in der Konsole)
VERBOSE = False

# Zeige FPS in Konsole
PRINT_FPS = False

# Speichere Performance-Logs
SAVE_LOGS = False
LOG_FILE = "performance.log"

# ============================================================================
# COLORS (BGR Format)
# ============================================================================

# Farben für Info-Overlay
INFO_BACKGROUND_COLOR = (0, 0, 0)       # Schwarz
INFO_TEXT_COLOR = (0, 255, 0)           # Grün

# Standard Box-Farbe (wenn nicht klassenbasiert)
DEFAULT_BOX_COLOR = (0, 255, 0)         # Grün

# ============================================================================
# KEYBOARD SHORTCUTS
# ============================================================================

# Können hier dokumentiert werden
# 'q' = Beenden
# 's' = Screenshot
# 'p' = Pause (kann implementiert werden)
# '+' / '-' = Konfidenz erhöhen/senken (kann implementiert werden)


# ============================================================================
# HELPER FUNKTIONEN
# ============================================================================

def get_model_info(model_name: str) -> dict:
    """
    Gibt Informationen über das gewählte Model zurück.
    
    Args:
        model_name: Name des YOLOv8 Models
        
    Returns:
        Dictionary mit Model-Informationen
    """
    model_info = {
        'yolov8n.pt': {
            'size': 'Nano',
            'params': '3.2M',
            'speed': 'Sehr schnell',
            'accuracy': 'Gut',
            'use_case': 'Echtzeit auf CPU'
        },
        'yolov8s.pt': {
            'size': 'Small',
            'params': '11.2M',
            'speed': 'Schnell',
            'accuracy': 'Besser',
            'use_case': 'Gute Balance'
        },
        'yolov8m.pt': {
            'size': 'Medium',
            'params': '25.9M',
            'speed': 'Mittel',
            'accuracy': 'Sehr gut',
            'use_case': 'GPU empfohlen'
        },
        'yolov8l.pt': {
            'size': 'Large',
            'params': '43.7M',
            'speed': 'Langsam',
            'accuracy': 'Exzellent',
            'use_case': 'Nur mit GPU'
        },
        'yolov8x.pt': {
            'size': 'Extra Large',
            'params': '68.2M',
            'speed': 'Sehr langsam',
            'accuracy': 'Beste',
            'use_case': 'Nur mit starker GPU'
        }
    }
    
    return model_info.get(model_name, {'size': 'Unbekannt'})


def print_config():
    """Gibt die aktuelle Konfiguration aus."""
    print("\n" + "=" * 60)
    print("📋 Aktuelle Konfiguration:")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Konfidenz-Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Kamera Index: {CAMERA_INDEX}")
    print(f"Auflösung: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"Box-Dicke: {BOX_THICKNESS}px")
    
    model_info = get_model_info(MODEL_NAME)
    print(f"\nModel Info:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Wenn config.py direkt ausgeführt wird, zeige Konfiguration
    print_config()
