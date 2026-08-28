Set-Content -Path .\export_model.py -Value 'from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.export(format="onnx", imgsz=640, half=True)
print("[SUCCESS] yolov8n.onnx has been successfully created!")'
