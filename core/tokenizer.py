import re
from typing import List, Dict

class SimpleWordTokenizer:
    def __init__(self, min_freq: int = 2, max_vocab: int = 30000):
        self.min_freq = min_freq
        self.max_vocab = max_vocab
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.nl_token  = "<nl>"
        self._re_tok   = re.compile(r"\n|\w+|[^\w\s]", flags=re.UNICODE)

    def _tokenize(self, text: str) -> List[str]:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        raw = self._re_tok.findall(text.lower())
        return [self.nl_token if t == "\n" else t for t in raw]

    def encode(self, text: str) -> List[int]:
        tokens = self._tokenize(text)
        unk_id = self.stoi[self.unk_token]
        return [self.stoi.get(t, unk_id) for t in tokens]

    def decode(self, ids: List[int]) -> str:
        tokens = [self.itos.get(i, self.unk_token) for i in ids]
        return self._detokenize(tokens)

    def _detokenize(self, tokens: List[str]) -> str:
        if not tokens:
            return ""
        no_space_before = {",", ".", "!", "?", ";", ":", ")", "]", "}"}
        no_space_after  = {"(", "[", "{"}
        out = []
        for i, tok in enumerate(tokens):
            if tok == self.nl_token:
                out.append("\n"); continue
            if i == 0:
                out.append(tok); continue
            prev = tokens[i - 1]
            add_space = tok not in no_space_before and prev not in no_space_after and prev != self.nl_token
            if add_space:
                out.append(" ")
            out.append(tok)
        return "".join(out)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)