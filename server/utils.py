import json
import time

def stream_texto(texto):
    for char in texto:
        yield f"data: {json.dumps({'token': char})}\n\n"
        time.sleep(0.01)
    yield f"data: {json.dumps({'end': True})}\n\n"

