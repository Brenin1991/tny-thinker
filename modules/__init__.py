from modules.vision.detector import VisionDetector

REGISTRY = {
    "vision": VisionDetector(),
}

def get(name):
    return REGISTRY.get(name)