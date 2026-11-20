"""
Riassunto della riunione con un modello linguistico LOCALE (Ollama): nessun dato lascia il PC.

Riunioni lunghe: la trascrizione viene divisa in blocchi, ogni blocco produce appunti
(fase "map"), poi gli appunti vengono fusi nel riassunto finale (fase "reduce").
"""
import json
import urllib.request
import urllib.error

OLLAMA = "http://localhost:11434"
MODEL = "qwen3.5:9b"
CHUNK_CHARS = 14000        # ~4k token di trascrizione per blocco
NUM_CTX = 16384

SYSTEM = ("Sei un assistente che redige verbali di riunioni aziendali in italiano. "
          "Scrivi solo ciò che risulta dalla trascrizione: non inventare nomi, date, numeri o decisioni. "
          "La trascrizione è automatica e può contenere errori di riconoscimento: interpreta con buon senso "
          "ma non aggiungere fatti. Se qualcosa non è chiaro, omettilo. "
          "Ogni riga inizia con il nome di chi parla seguito da due punti: quando qualcuno dice "
          "\"mando io\", \"faccio io\", \"controllo\" e simili, il responsabile è chi parla. "
          "Parole strane a inizio frase (es. \"V. A. Baini\" per \"va bene\") sono errori di trascrizione, "
          "non nomi di persona.")

MAP_PROMPT = """Questa è la parte {i} di {n} della trascrizione di una riunione.
Estrai appunti sintetici, in elenco puntato, raggruppati sotto questi titoli:
ARGOMENTI (cosa si è discusso, con i punti chiave e chi li ha sostenuti)
DECISIONI (solo quelle prese esplicitamente)
AZIONI (chi deve fare cosa, ed entro quando se detto)
QUESTIONI APERTE (dubbi, problemi non risolti, cose da verificare)
Scrivi "nessuna" sotto un titolo se non c'è nulla.

TRASCRIZIONE:
{text}"""

REDUCE_PROMPT = """Ecco {what} della riunione "{title}" del {date} (durata {duration}, partecipanti: {people}).

{body}

Scrivi il verbale finale in Markdown, in italiano, con ESATTAMENTE queste sezioni:

## Sintesi
(5-10 righe: scopo della riunione e conclusioni principali)

## Argomenti discussi
(un sotto-elenco per argomento, con i punti chiave e le posizioni di chi è intervenuto)

## Decisioni
(elenco puntato; "Nessuna decisione esplicita." se non ce ne sono)

## Azioni da fare
(tabella Markdown con colonne: Chi | Cosa | Entro quando — usa "da definire" se non detto)

## Questioni aperte
(elenco puntato)

Non aggiungere altre sezioni, premesse o commenti finali."""


class OllamaError(RuntimeError):
    pass


def available(model: str = MODEL) -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=3) as r:
            names = [m["name"] for m in json.load(r).get("models", [])]
        return model in names or f"{model}:latest" in names
    except Exception:
        return False


def _chat(prompt: str, model: str, progress=None, temperature=0.2) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "stream": True,
        "think": False,
        "keep_alive": "2m",
        "options": {"temperature": temperature, "num_ctx": NUM_CTX},
    }).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    out = []
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            for line in r:
                if not line.strip():
                    continue
                msg = json.loads(line)
                if "error" in msg:
                    raise OllamaError(msg["error"])
                out.append(msg.get("message", {}).get("content", ""))
                if progress and len(out) % 20 == 0:
                    progress(len(out))
                if msg.get("done"):
                    break
    except urllib.error.URLError as e:
        raise OllamaError(f"Ollama non raggiungibile ({e.reason}). È avviato?") from e
    return "".join(out).strip()


def transcript_text(data: dict) -> str:
    names = data["names"]
    lines = []
    for t in data["turns"]:
        name = names.get(t["speaker"], f"Persona {t['speaker']}")
        if lines and lines[-1][0] == name:
            lines[-1][1] += " " + t["text"]
        else:
            lines.append([name, t["text"]])
    return "\n".join(f"{n}: {txt}" for n, txt in lines)


def _chunks(text: str, size: int):
    """Divide per righe (interventi) senza spezzare a metà frase quando possibile."""
    out, cur = [], ""
    for line in text.split("\n"):
        while len(line) > size:                     # intervento lunghissimo: taglia sui punti
            cut = line.rfind(". ", 0, size)
            cut = cut + 1 if cut > size // 2 else size
            if cur:
                out.append(cur)
                cur = ""
            out.append(line[:cut])
            line = line[cut:].lstrip()
        if len(cur) + len(line) + 1 > size and cur:
            out.append(cur)
            cur = ""
        cur += line + "\n"
    if cur.strip():
        out.append(cur)
    return out


def summarize(data: dict, model: str = MODEL, status=lambda msg, frac: None) -> str:
    """Restituisce il riassunto in Markdown (senza intestazione)."""
    def hms(sec):
        sec = int(max(0, sec))
        return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"
    text = transcript_text(data)
    people = ", ".join(dict.fromkeys(data["names"][k] for k in sorted(data["names"], key=int)))
    meta = dict(title=data.get("title") or "Riunione", date=data.get("date", ""),
                duration=hms(data.get("duration", 0)), people=people)
    parts = _chunks(text, CHUNK_CHARS)
    if len(parts) == 1:
        status("Scrivo il riassunto…", 0.1)
        return _chat(REDUCE_PROMPT.format(what="la trascrizione", body="TRASCRIZIONE:\n" + text, **meta),
                     model, lambda k: status(None, min(0.95, 0.1 + k / 1500)))
    notes = []
    for i, p in enumerate(parts, 1):
        status(f"Riassunto: leggo la parte {i} di {len(parts)}…", (i - 1) / (len(parts) + 1))
        notes.append(f"--- Parte {i} ---\n" + _chat(MAP_PROMPT.format(i=i, n=len(parts), text=p), model))
    status("Riassunto: compongo il verbale…", len(parts) / (len(parts) + 1))
    return _chat(REDUCE_PROMPT.format(what="gli appunti presi, parte per parte,", body="\n\n".join(notes),
                                      **meta), model)


def write_summary(out_dir, data: dict, model: str = MODEL, status=lambda m, f: None):
    md = summarize(data, model, status)
    people = ", ".join(dict.fromkeys(data["names"][k] for k in sorted(data["names"], key=int)))
    head = (f"# {data.get('title') or 'Riunione'}\n\n"
            f"**Data:** {data.get('date', '')}  \n**Partecipanti:** {people}\n\n"
            f"> Riassunto generato automaticamente in locale ({model}): verificare con la trascrizione.\n\n")
    path = out_dir / "riassunto.md"
    path.write_text(head + md + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    import sys
    from pathlib import Path
    d = Path(sys.argv[1])
    data = json.loads((d / "trascrizione.json").read_text(encoding="utf-8"))
    p = write_summary(d, data, status=lambda m, f: m and print(m))
    print(p.read_text(encoding="utf-8"))
