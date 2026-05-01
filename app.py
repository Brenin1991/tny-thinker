from flask import Flask, request, Response, jsonify, send_from_directory
from flask_cors import CORS
import os, uuid, glob

# Imports da pasta Core (IA)
from core.config import MODEL_PATH, DEVICE
from core.utils import load_checkpoint, gerar_resposta, carregar_modelo_dinamico

# Imports da nova pasta Server (Modularizada)
from server.database import init_db, get_db, now_iso
from server.actions import detectar_acao, calcular, gerar_link_busca
from server.utils import stream_texto

app = Flask(__name__, static_folder="client", static_url_path="/static")
CORS(app)

# Inicialização global do Modelo
model, tokenizer, cfg, is_dialogue = load_checkpoint(MODEL_PATH)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

init_db()

@app.route("/")
def index():
    return send_from_directory("client", "index.html")

# --- ROTAS DE CONVERSA (Exemplo de uma) ---
@app.route("/conversations", methods=["GET"])
def list_conversations():
    conn = get_db()
    rows = conn.execute(
        "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route("/conversations", methods=["POST"])
def create_conversation():
    data = request.json or {}
    conv_id = str(uuid.uuid4())
    title = data.get("title", "Nova conversa")
    now = now_iso()
    conn = get_db()
    conn.execute(
        "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (conv_id, title, now, now)
    )
    conn.commit()
    conn.close()
    return jsonify({"id": conv_id, "title": title, "created_at": now, "updated_at": now}), 201


@app.route("/conversations/<conv_id>", methods=["GET"])
def get_conversation(conv_id):
    conn = get_db()
    conv = conn.execute(
        "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?", (conv_id,)
    ).fetchone()
    if not conv:
        conn.close()
        return jsonify({"error": "not found"}), 404
    msgs = conn.execute(
        "SELECT id, role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY id ASC",
        (conv_id,)
    ).fetchall()
    conn.close()
    return jsonify({"conversation": dict(conv), "messages": [dict(m) for m in msgs]})


@app.route("/conversations/<conv_id>", methods=["DELETE"])
def delete_conversation(conv_id):
    conn = get_db()
    conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
    conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


@app.route("/conversations/<conv_id>/rename", methods=["PATCH"])
def rename_conversation(conv_id):
    data = request.json or {}
    title = data.get("title", "").strip()
    if not title:
        return jsonify({"error": "title required"}), 400
    conn = get_db()
    conn.execute(
        "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
        (title, now_iso(), conv_id)
    )
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


# ================================
# ✏️  ROTAS DE MENSAGEM
# ================================

@app.route("/conversations/<conv_id>/messages", methods=["POST"])
def add_message(conv_id):
    data    = request.json or {}
    role    = data.get("role", "user")
    content = data.get("content", "").strip()
    if not content:
        return jsonify({"error": "content required"}), 400
    conn = get_db()
    cur  = conn.execute(
        "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (conv_id, role, content, now_iso())
    )
    conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now_iso(), conv_id))
    conn.commit()
    msg_id = cur.lastrowid
    conn.close()
    return jsonify({"id": msg_id, "role": role, "content": content}), 201


@app.route("/messages/<int:msg_id>", methods=["PATCH"])
def edit_message(msg_id):
    data    = request.json or {}
    content = data.get("content", "").strip()
    if not content:
        return jsonify({"error": "content required"}), 400
    conn = get_db()
    conn.execute("UPDATE messages SET content = ? WHERE id = ?", (content, msg_id))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


@app.route("/messages/<int:msg_id>", methods=["DELETE"])
def delete_message(msg_id):
    conn = get_db()
    conn.execute("DELETE FROM messages WHERE id = ?", (msg_id,))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})

# --- ENDPOINT DE CHAT ---
@app.route("/chat-stream", methods=["POST"])
def chat_stream():
    global model, tokenizer, cfg, is_dialogue
    data = request.json
    user_input = data.get("message", "").strip()
    conv_id = data.get("conversation_id")

    if not user_input:
        return Response(stream_texto("mensagem vazia"), mimetype="text/event-stream")

    # --- LOGICA DE BANCO DE DADOS (CRUCIAL) ---
    conn = get_db()
    
    # 1. Se não tem ID ou ID não existe, cria nova conversa
    if not conv_id:
        conv_id = str(uuid.uuid4())
        title = user_input[:40] + ("..." if len(user_input) > 40 else "")
        conn.execute("INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                     (conv_id, title, now_iso(), now_iso()))
    
    # 2. Salva a mensagem do USUÁRIO
    conn.execute("INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                 (conv_id, "user", user_input, now_iso()))
    conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now_iso(), conv_id))
    conn.commit()
    conn.close()

    acao = detectar_acao(user_input)

    # --- FUNÇÃO AUXILIAR PARA SALVAR RESPOSTA DO BOT ---
    def salvar_e_stream(texto_bot):
        # Salva no banco
        c = get_db()
        c.execute("INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                  (conv_id, "assistant", texto_bot, now_iso()))
        c.commit()
        c.close()
        # Envia para o HTML
        yield from stream_texto(texto_bot)

    # ====================
    # AÇÕES RÁPIDAS
    # ====================
    if acao == "calculo":
        bot_text = f"Resultado: {calcular(user_input)}"
        return Response(salvar_e_stream(bot_text), mimetype="text/event-stream", headers={"X-Conversation-Id": conv_id})

    elif acao == "pesquisa":
        bot_text = f"Busca: {gerar_link_busca(user_input)}"
        return Response(salvar_e_stream(bot_text), mimetype="text/event-stream", headers={"X-Conversation-Id": conv_id})

    # ====================
    # MODELO DE IA
    # ====================
    def gerar_stream_modelo():
        prompt = f"usuário: {user_input}\nassistente:" if is_dialogue else user_input
        resposta, conf, _ = gerar_resposta(prompt, model, tokenizer, cfg)
        
        if conf < 0.30:
            resposta = "Não tenho certeza suficiente para responder isso."
        
        # Salva antes de terminar o stream
        c = get_db()
        c.execute("INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                  (conv_id, "assistant", resposta, now_iso()))
        c.commit()
        c.close()

        yield from stream_texto(resposta)

    return Response(gerar_stream_modelo(), mimetype="text/event-stream", headers={"X-Conversation-Id": conv_id})

@app.route("/models", methods=["GET"])
def list_models():
    models_dir = os.path.join(BASE_DIR, "models")
    files = glob.glob(os.path.join(models_dir, "*.pt"))
    # Retorna apenas os nomes dos arquivos
    return jsonify([os.path.basename(f) for f in files])

@app.route("/select-model", methods=["POST"])
def select_model():
    global model, tokenizer, cfg, is_dialogue
    data = request.json
    model_name = data.get("model_name")
    
    try:
        # ATUALIZADO: A função agora retorna os novos objetos
        resultado = carregar_modelo_dinamico(model_name, model, tokenizer, cfg, is_dialogue)
        
        if resultado:
            model, tokenizer, cfg, is_dialogue = resultado
            return jsonify({"status": "success", "model": model_name})
        
        return jsonify({"error": "Falha ao carregar"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)