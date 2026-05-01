import torch
import torch.nn.functional as F

from .config import DEVICE, GEN_PARAMS 
from .model import GPTConfig, GPTModel
from .tokenizer import SimpleWordTokenizer

def load_checkpoint(file_path):
    # Usamos o DEVICE importado do config.py
    checkpoint = torch.load(file_path, map_location=DEVICE)
    cfg = GPTConfig(**checkpoint["config"])
    model = GPTModel(cfg).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    tokenizer = SimpleWordTokenizer()
    tokenizer.stoi = checkpoint["tokenizer"]
    tokenizer.itos = {i: s for s, i in tokenizer.stoi.items()}
    
    return model, tokenizer, cfg, checkpoint.get("is_dialogue", False)

@torch.no_grad()
def gerar_resposta(prompt, model, tokenizer, cfg):
    """
    Gera uma resposta para o prompt dado usando os parâmetros do GEN_PARAMS.
    """
    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [tokenizer.stoi.get("<unk>", 0)]

    # Criamos o tensor diretamente no DEVICE correto
    idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    gerados = []
    total_conf = 0.0
    ultima_entropia = 0.0

    # Atalhos para os parâmetros de geração para o código ficar limpo
    params = GEN_PARAMS

    for step in range(params["max_new_tokens"]):
        idx_cond = idx[:, -cfg.max_seq_len:]
        logits = model(idx_cond)
        
        # Ajuste de Temperatura
        logits = logits[:, -1, :] / max(params["temperature"], 1e-4)

        # Penalidade de repetição e frequência
        tokens_vistos = idx.tolist()[0]
        freq = {}
        for t in tokens_vistos:
            freq[t] = freq.get(t, 0) + 1
        for token_id, count in freq.items():
            logits[0, token_id] /= params["repetition_penalty"]
            logits[0, token_id] -= params["frequency_penalty"] * count

        # Anti-loop (3 tokens iguais)
        if idx.shape[1] >= 3:
            last3 = idx[0, -3:].tolist()
            if len(set(last3)) == 1:
                logits[0, last3[-1]] = -float("inf")

        # Top-K e Top-P
        v, i = torch.topk(logits, min(params["top_k"], logits.size(-1)))
        probs = F.softmax(v, dim=-1)
        
        # Nucleus Sampling (Top-P)
        if params["top_p"] < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs_sorted = F.softmax(sorted_logits, dim=-1)
            cum_probs = torch.cumsum(probs_sorted, dim=-1)
            cutoff = cum_probs > params["top_p"]
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = False
            sorted_logits[cutoff] = -float("inf")
            filtered = torch.full_like(logits, -float("inf"))
            filtered.scatter_(1, sorted_idx, sorted_logits)
            probs = F.softmax(filtered, dim=-1)
            next_token = torch.multinomial(probs, 1)
        else:
            next_token = i.gather(-1, torch.multinomial(probs, 1))

        # Métricas de Confiança
        probs_final = F.softmax(logits, dim=-1)
        total_conf += torch.max(probs_final).item()
        ultima_entropia = -(probs_final * torch.log(probs_final + 1e-9)).sum().item()

        token_id = next_token.item()
        gerados.append(token_id)
        idx = torch.cat([idx, next_token], dim=1)

        # Critérios de Parada (Turno do usuário ou linha dupla)
        if step >= 2:
            trecho = tokenizer.decode(gerados[-4:])
            if "usuário" in trecho or "user" in trecho:
                gerados = gerados[:-4] if len(gerados) > 4 else gerados[:1]
                break

        if len(gerados) >= 2:
            trecho_nl = tokenizer.decode(gerados[-2:])
            if trecho_nl.count("\n") >= 2:
                gerados = gerados[:-2]
                break

    avg_conf = total_conf / max(len(gerados), 1)
    resposta = tokenizer.decode(gerados)

    # Limpeza simples da resposta
    linhas = [l.strip() for l in resposta.split("\n") if l.strip()]
    resposta = linhas[0] if linhas else resposta.strip()

    return resposta, avg_conf, ultima_entropia

def carregar_modelo_dinamico(nome_arquivo, model, tokenizer, cfg, is_dialogue):
    try:
        path = f"models/{nome_arquivo}"
        # MUDANÇA AQUI: de 'device' para 'DEVICE'
        checkpoint = torch.load(path, map_location=DEVICE) 
        
        cfg = GPTConfig(**checkpoint["config"])
        model = GPTModel(cfg).to(DEVICE) # MUDANÇA AQUI TAMBÉM
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        tokenizer = SimpleWordTokenizer()
        tokenizer.stoi = checkpoint["tokenizer"]
        tokenizer.itos = {i: s for s, i in tokenizer.stoi.items()}
        is_dialogue = checkpoint.get("is_dialogue", False)
        
        print(f"🔄 Modelo alterado para: {nome_arquivo}")
        return model, tokenizer, cfg, is_dialogue # Retorne os novos objetos
    except Exception as e:
        print(f"❌ Erro ao carregar: {e}")
        return None