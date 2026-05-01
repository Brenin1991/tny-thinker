import re
import math
import urllib.parse

def detectar_acao(texto: str):
    t = texto.lower().strip()
    if re.match(r"^[0-9\.\+\-\*/\(\) ]+$", t): return "calculo"
    if any(k in t for k in ["pesquisar", "buscar", "procura"]): return "pesquisa"
    return "normal"

def calcular(expr: str):
    try:
        allowed = {"__builtins__": None}
        return str(eval(expr, allowed, math.__dict__))
    except:
        return "erro no cálculo"

def gerar_link_busca(texto: str):
    query = urllib.parse.quote(texto)
    return f"https://www.google.com/search?q={query}"