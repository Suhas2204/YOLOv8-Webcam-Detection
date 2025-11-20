"""
YOLOv8 Webcam Object Detection
================================
Real-time object detection using YOLOv8 and webcam feed.

Author: [Dein Name]
Date: November 2024
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from typing import Tuple, List


class WebcamDetector:
    """
    Klasse für Echtzeit-Objekterkennung mit YOLOv8 und Webcam.
    
    Attributes:
        model: YOLOv8 Model
        cap: OpenCV VideoCapture Objekt
        confidence_threshold: Minimum Konfidenz für Detektionen
    """
    
    def __init__(self, model_name: str = 'yolov8n.pt', confidence: float = 0.5):
        """
        Initialisiert den WebcamDetector.
        
        Args:
            model_name: Name des YOLOv8 Models (yolov8n.pt, yolov8s.pt, etc.)
            confidence: Minimum Konfidenz-Threshold (0.0 - 1.0)
        """
        print("🚀 Initialisiere YOLOv8 Webcam Detector...")
        
        # Lade YOLOv8 Model
        print(f"📦 Lade Model: {model_name}")
        self.model = YOLO(model_name)
        
        # Setze Konfidenz-Threshold
        self.confidence_threshold = confidence
        
        # Öffne Webcam (0 = Standard Webcam)
        print("📹 Öffne Webcam...")
        self.cap = cv2.VideoCapture(0)
        
        # Prüfe ob Webcam erfolgreich geöffnet wurde
        if not self.cap.isOpened():
            raise ValueError("❌ Fehler: Webcam konnte nicht geöffnet werden!")
        
        # Setze Auflösung (Optional, für bessere Performance)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("✅ Initialisierung erfolgreich!")
        
    
    def draw_detections(self, frame: np.ndarray, results) -> np.ndarray:
        """
        Zeichnet Bounding Boxes und Labels auf das Frame.
        
        Args:
            frame: Input Frame von der Webcam
            results: YOLOv8 Detection Results
            
        Returns:
            Frame mit gezeichneten Detektionen
        """
        # Durchlaufe alle Detektionen
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Extrahiere Box-Koordinaten
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Extrahiere Konfidenz und Klasse
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                # Filtere nach Konfidenz-Threshold
                if confidence < self.confidence_threshold:
                    continue
                
                # Farbe basierend auf Klasse (für Konsistenz)
                color = self._get_color(class_id)
                
                # Zeichne Bounding Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Erstelle Label Text
                label = f"{class_name}: {confidence:.2f}"
                
                # Berechne Label-Hintergrund
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                # Zeichne Label-Hintergrund
                cv2.rectangle(
                    frame,
                    (x1, y1 - text_height - baseline - 10),
                    (x1 + text_width, y1),
                    color,
                    -1  # Gefülltes Rechteck
                )
                
                # Zeichne Label Text
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - baseline - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),  # Weiße Schrift
                    2
                )
        
        return frame
    
    
    def _get_color(self, class_id: int) -> Tuple[int, int, int]:
        """
        Generiert eine konsistente Farbe für jede Klasse.
        
        Args:
            class_id: ID der erkannten Klasse
            
        Returns:
            BGR Farb-Tuple
        """
        # Einfache Farbgenerierung basierend auf class_id
        np.random.seed(class_id)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        return color
    
    
    def add_info_overlay(self, frame: np.ndarray, fps: float, detection_count: int) -> np.ndarray:
        """
        Fügt Informations-Overlay zum Frame hinzu.
        
        Args:
            frame: Input Frame
            fps: Aktuelle FPS
            detection_count: Anzahl der Detektionen
            
        Returns:
            Frame mit Info-Overlay
        """
        # Halbtransparenter Hintergrund für Info
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (250, 80), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # FPS anzeigen
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Anzahl Detektionen
        cv2.putText(
            frame,
            f"Detektionen: {detection_count}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        return frame
    
    
    def run(self):
        """
        Hauptschleife für die Webcam-Detection.
        Drücke 'q' zum Beenden.
        """
        print("\n🎥 Starte Webcam Detection...")
        print("💡 Drücke 'q' zum Beenden")
        print("💡 Drücke 's' um Screenshot zu speichern\n")
        
        # FPS Berechnung
        fps = 0
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                # Lese Frame von Webcam
                ret, frame = self.cap.read()
                
                if not ret:
                    print("❌ Fehler beim Lesen des Frames")
                    break
                
                # Führe YOLOv8 Inferenz durch
                results = self.model(frame, verbose=False)
                
                # Zähle Detektionen
                detection_count = len(results[0].boxes) if len(results) > 0 else 0
                
                # Zeichne Detektionen auf Frame
                frame = self.draw_detections(frame, results)
                
                # Berechne FPS
                frame_count += 1
                if frame_count >= 10:
                    end_time = time.time()
                    fps = frame_count / (end_time - start_time)
                    frame_count = 0
                    start_time = time.time()
                
                # Füge Info-Overlay hinzu
                frame = self.add_info_overlay(frame, fps, detection_count)
                
                # Zeige Frame
                cv2.imshow('YOLOv8 Webcam Detection', frame)
                
                # Keyboard Input
                key = cv2.waitKey(1) & 0xFF
                
                # 'q' zum Beenden
                if key == ord('q'):
                    print("\n👋 Beende Programm...")
                    break
                
                # 's' für Screenshot
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = f"screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Screenshot gespeichert: {filename}")
        
        except KeyboardInterrupt:
            print("\n⚠️  Programm durch Benutzer unterbrochen")
        
        finally:
            # Cleanup
            self.cleanup()
    
    
    def cleanup(self):
        """
        Gibt Ressourcen frei.
        """
        print("🧹 Räume auf...")
        self.cap.release()
        cv2.destroyAllWindows()
        print("✅ Cleanup abgeschlossen")


def main():
    """
    Hauptfunktion zum Starten der Webcam Detection.
    """
    # Importiere Konfiguration (optional)
    try:
        from config import MODEL_NAME, CONFIDENCE_THRESHOLD
        print("📝 Verwende Konfiguration aus config.py")
    except ImportError:
        # Fallback auf Default-Werte
        MODEL_NAME = 'yolov8n.pt'
        CONFIDENCE_THRESHOLD = 0.5
        print("📝 Verwende Standard-Konfiguration")
    
    try:
        # Erstelle Detector Instanz
        detector = WebcamDetector(
            model_name=MODEL_NAME,
            confidence=CONFIDENCE_THRESHOLD
        )
        
        # Starte Detection
        detector.run()
        
    except Exception as e:
        print(f"❌ Fehler: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()