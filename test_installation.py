"""
Installation Test Script
========================
Testet ob alle Dependencies korrekt installiert sind.
"""

import sys


def test_imports():
    """Teste alle notwendigen Imports."""
    print("=" * 60)
    print("🧪 Teste Installation...")
    print("=" * 60)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Python Version
    print("\n1️⃣  Python Version:")
    python_version = sys.version_info
    print(f"   Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major >= 3 and python_version.minor >= 8:
        print("   ✅ Python Version OK (3.8+)")
        tests_passed += 1
    else:
        print("   ❌ Python Version zu alt! Benötigt 3.8+")
        tests_failed += 1
    
    # Test 2: OpenCV
    print("\n2️⃣  OpenCV:")
    try:
        import cv2
        print(f"   Version: {cv2.__version__}")
        print("   ✅ OpenCV installiert")
        tests_passed += 1
    except ImportError:
        print("   ❌ OpenCV nicht gefunden!")
        print("   Installiere: pip install opencv-python")
        tests_failed += 1
    
    # Test 3: NumPy
    print("\n3️⃣  NumPy:")
    try:
        import numpy as np
        print(f"   Version: {np.__version__}")
        print("   ✅ NumPy installiert")
        tests_passed += 1
    except ImportError:
        print("   ❌ NumPy nicht gefunden!")
        print("   Installiere: pip install numpy")
        tests_failed += 1
    
    # Test 4: Ultralytics (YOLOv8)
    print("\n4️⃣  Ultralytics (YOLOv8):")
    try:
        from ultralytics import YOLO
        import ultralytics
        print(f"   Version: {ultralytics.__version__}")
        print("   ✅ Ultralytics installiert")
        tests_passed += 1
    except ImportError:
        print("   ❌ Ultralytics nicht gefunden!")
        print("   Installiere: pip install ultralytics")
        tests_failed += 1
    
    # Test 5: PIL/Pillow
    print("\n5️⃣  Pillow:")
    try:
        from PIL import Image
        import PIL
        print(f"   Version: {PIL.__version__}")
        print("   ✅ Pillow installiert")
        tests_passed += 1
    except ImportError:
        print("   ❌ Pillow nicht gefunden!")
        print("   Installiere: pip install pillow")
        tests_failed += 1
    
    # Test 6: Webcam Verfügbarkeit
    print("\n6️⃣  Webcam:")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"   Auflösung: {frame.shape[1]}x{frame.shape[0]}")
                print("   ✅ Webcam verfügbar")
                tests_passed += 1
            else:
                print("   ⚠️  Webcam geöffnet, aber kein Frame lesbar")
                tests_failed += 1
            cap.release()
        else:
            print("   ⚠️  Webcam nicht verfügbar (optional)")
            print("   Tipp: Versuche VideoCapture(1) statt VideoCapture(0)")
            # Nicht als Fehler werten, da Webcam optional
            tests_passed += 1
    except Exception as e:
        print(f"   ⚠️  Fehler beim Webcam-Test: {e}")
        tests_passed += 1  # Nicht als kritischer Fehler
    
    # Zusammenfassung
    print("\n" + "=" * 60)
    print("📊 Test-Zusammenfassung:")
    print("=" * 60)
    print(f"✅ Erfolgreich: {tests_passed}")
    print(f"❌ Fehlgeschlagen: {tests_failed}")
    
    if tests_failed == 0:
        print("\n🎉 Alle Tests bestanden!")
        print("🚀 Du kannst jetzt 'python webcam_detection.py' ausführen!")
    else:
        print("\n⚠️  Einige Tests sind fehlgeschlagen.")
        print("📝 Installiere die fehlenden Pakete mit:")
        print("   pip install -r requirements.txt")
    
    print("=" * 60)
    
    return tests_failed == 0


def test_yolo_model():
    """Teste ob YOLOv8 Model geladen werden kann."""
    print("\n" + "=" * 60)
    print("🧪 Teste YOLOv8 Model...")
    print("=" * 60)
    
    try:
        from ultralytics import YOLO
        import numpy as np
        
        print("\n📦 Lade YOLOv8n Model...")
        print("   (Wird beim ersten Mal heruntergeladen - kann dauern!)")
        
        model = YOLO('yolov8n.pt')
        print("   ✅ Model erfolgreich geladen!")
        
        # Test mit Dummy-Bild
        print("\n🔍 Teste Inferenz mit Dummy-Bild...")
        dummy_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        results = model(dummy_image, verbose=False)
        
        print(f"   ✅ Inferenz erfolgreich!")
        print(f"   Anzahl Klassen: {len(model.names)}")
        print(f"   Beispiel-Klassen: {list(model.names.values())[:5]}...")
        
        print("\n🎉 YOLOv8 Test bestanden!")
        return True
        
    except Exception as e:
        print(f"\n❌ Fehler beim YOLOv8 Test: {e}")
        print("   Versuche: pip install --upgrade ultralytics")
        return False


def main():
    """Hauptfunktion."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "YOLOv8 Webcam Detection - Test Suite" + " " * 11 + "║")
    print("╚" + "═" * 58 + "╝")
    
    # Teste Basis-Installation
    basic_ok = test_imports()
    
    if basic_ok:
        # Teste YOLOv8 separat (kann lange dauern)
        print("\n❓ Möchtest du auch das YOLOv8 Model testen?")
        print("   (Dauert beim ersten Mal ~30-60 Sekunden)")
        response = input("   (j/n): ").lower().strip()
        
        if response in ['j', 'ja', 'y', 'yes']:
            test_yolo_model()
    
    print("\n")


if __name__ == "__main__":
    main()
