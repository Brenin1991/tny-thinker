import torch

MODEL_PATH = "models/modelo_finetuned_best.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parâmetros de geração
GEN_PARAMS = {
    "max_new_tokens": 200,
    "temperature": 0,
    "top_k": 40,
    "top_p": 1,
    "repetition_penalty": 0.9,
    "frequency_penalty": 0.1
}