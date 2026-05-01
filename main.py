from core.config import MODEL_PATH, DEVICE
from core.utils import load_checkpoint, gerar_resposta, carregar_modelo_dinamico

def main():
    print(f"Rodando em: {DEVICE}")
    
    # Carrega o modelo inicial
    model, tokenizer, cfg, is_dialogue = load_checkpoint(MODEL_PATH)
    
    if model is not None:
        print("Modelo carregado!" + str(model))
    else:
        print("Falha no carregamento inicial.")
        return

    while True:
        user_input = input("\nVocê: ").strip()
        if user_input.lower() in ["sair", "exit"]: break
        
        prompt = f"usuário: {user_input}\nassistente:" if is_dialogue else user_input
        resposta, conf, entropy = gerar_resposta(prompt, model, tokenizer, cfg)
        
        print(f"\nAssistente: {resposta}")

if __name__ == "__main__":
    main()