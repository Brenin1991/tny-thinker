from ultralytics import YOLO

class VisionDetector:
    def __init__(
        self,
        detection_model: str = "yolov8x.pt",
        classification_model: str = "yolov8x-cls.pt",
        top_k: int = 3
    ):
        self.det_model   = YOLO(detection_model)
        self.cls_model   = YOLO(classification_model)
        self.top_k       = top_k

    # ── detecção de objetos com bounding boxes ──────────────
    def detectar(self, imagem_path: str) -> dict:
        results = self.det_model(imagem_path)
        objetos = []
        for r in results:
            for box in r.boxes:
                objetos.append({
                    "classe":    self.det_model.names[int(box.cls)],
                    "confianca": float(box.conf),
                    "bbox":      box.xyxy[0].tolist()
                })
        return {"objetos": objetos}

    # ── classificação da imagem inteira (ImageNet 1000 classes) ─
    def classificar(self, imagem_path: str) -> dict:
        results = self.cls_model(imagem_path)
        classes = []
        for r in results:
            # top5 índices e probabilidades
            top_indices = r.probs.top5[:self.top_k]
            top_confs   = r.probs.top5conf[:self.top_k].tolist()
            for idx, conf in zip(top_indices, top_confs):
                classes.append({
                    "classe":    self.cls_model.names[idx],
                    "confianca": float(conf)
                })
        return {"classes": classes}

    # ── roda os dois juntos ─────────────────────────────────
    def analisar(self, imagem_path: str) -> dict:
        deteccao      = self.detectar(imagem_path)
        classificacao = self.classificar(imagem_path)
        return {
            "objetos": deteccao["objetos"],
            "classes": classificacao["classes"]
        }