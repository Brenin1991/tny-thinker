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

from modules import get

vision = get("vision")

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

@app.route("/chat-stream", methods=["POST"])
def chat_stream():
    global model, tokenizer, cfg, is_dialogue
    
    # Suporta JSON (só texto) e multipart/form-data (texto + imagem)
    if request.content_type and "multipart" in request.content_type:
        user_input = request.form.get("message", "").strip()
        conv_id    = request.form.get("conversation_id")
    else:
        data       = request.json or {}
        user_input = data.get("message", "").strip()
        conv_id    = data.get("conversation_id")

    if not user_input:
        return Response(stream_texto("mensagem vazia"), mimetype="text/event-stream")

    # ====================
    # VISÃO (early return)
    # ====================
    imagem = request.files.get("image")
    if imagem:
        import tempfile, os
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            imagem.save(tmp.name)
            tmp_path = tmp.name

        try:
            deteccoes = vision.analisar(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        objetos = deteccoes.get("objetos", [])
        classes = deteccoes.get("classes", [])

        partes = []
        if objetos:
            desc_det = ", ".join(
                f"{o['classe']} ({o['confianca']:.0%})"
                for o in objetos
            )
            partes.append(f"🔍 Detectado: {desc_det}")

        if classes:
            desc_cls = ", ".join(
                f"{c['classe']} ({c['confianca']:.0%})"
                for c in classes
            )
            partes.append(f"🏷️ Classificação: {desc_cls}")

        bot_text = "\n".join(partes) if partes else "🔍 Nenhum objeto detectado na imagem."

        conn = get_db()
        if not conv_id:
            conv_id = str(uuid.uuid4())
            title = user_input[:40] + ("..." if len(user_input) > 40 else "")
            conn.execute("INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                        (conv_id, title, now_iso(), now_iso()))
        conn.execute("INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                    (conv_id, "user", user_input, now_iso()))
        conn.execute("INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                    (conv_id, "assistant", bot_text, now_iso()))
        conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now_iso(), conv_id))
        conn.commit()
        conn.close()

        return Response(stream_texto(bot_text), mimetype="text/event-stream", headers={"X-Conversation-Id": conv_id})
    # --- LOGICA DE BANCO DE DADOS ---
    conn = get_db()

    if not conv_id:
        conv_id = str(uuid.uuid4())
        title = user_input[:40] + ("..." if len(user_input) > 40 else "")
        conn.execute("INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                     (conv_id, title, now_iso(), now_iso()))

    conn.execute("INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                 (conv_id, "user", user_input, now_iso()))
    conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now_iso(), conv_id))
    conn.commit()
    conn.close()

    acao = detectar_acao(user_input)

    def salvar_e_stream(texto_bot):
        c = get_db()
        c.execute("INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                  (conv_id, "assistant", texto_bot, now_iso()))
        c.commit()
        c.close()
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