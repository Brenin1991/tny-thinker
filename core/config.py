import torch

MODEL_PATH = "../models/modelo_finetuned_best.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parâmetros de geração
GEN_PARAMS = {
    "max_new_tokens": 200,
    "temperature": 1,
    "top_k": 100,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "frequency_penalty": 0.05
}