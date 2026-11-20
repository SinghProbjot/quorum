"""
Riconoscimento di "chi parla quando" (diarizzazione), stile pyannote:

1. segmentazione locale con pyannote-segmentation-3.0 su finestre di 10 s (passo 5 s):
   in ogni finestra il modello separa fino a 3 voci "locali";
2. per ogni voce locale di ogni finestra si calcola un'impronta vocale (embedding)
   usando TUTTO il suo audio nella finestra (più audio = impronta più affidabile);
3. le impronte di tutta la riunione vengono raggruppate (clustering gerarchico
   + rifinitura sui centroidi): ogni gruppo è una persona;
4. le voci note (profili salvati) vengono riconosciute confrontando i centroidi.
"""
from dataclasses import dataclass, field
import numpy as np
import onnxruntime as ort
import sherpa_onnx

SR = 16000
WIN = 160000          # 10 s
STEP = 80000          # 5 s
FRAME_SHIFT = 270     # campioni per frame del modello di segmentazione
FRAME_OFFSET = 495    # metà del campo recettivo
# classi powerset: 0=silenzio, 1..3 = una sola voce locale, 4..6 = sovrapposizione
POWERSET_SINGLE = {1: 0, 2: 1, 3: 2}


@dataclass
class Item:
    """Una voce locale in una finestra."""
    runs: list            # [(inizio, fine)] in secondi, solo nella parte centrale della finestra
    speech: float         # secondi di parlato usati per l'impronta
    emb: np.ndarray = None
    label: int = -1


@dataclass
class Result:
    segments: list                      # [(inizio, fine, persona)]
    centroids: dict = field(default_factory=dict)   # persona -> impronta media normalizzata
    talk: dict = field(default_factory=dict)        # persona -> secondi parlati
    profiles: dict = field(default_factory=dict)    # persona -> impronta NON centrata (confrontabile tra riunioni)


def _norm(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


class Diarizer:
    def __init__(self, seg_model: str, emb_model: str, threads: int = 6):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.seg = ort.InferenceSession(seg_model, so, providers=["CPUExecutionProvider"])
        self.ex = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=emb_model, num_threads=threads))

    # ------------------------------------------------------------------ #
    def embed(self, samples: np.ndarray) -> np.ndarray:
        st = self.ex.create_stream()
        st.accept_waveform(SR, samples)
        st.input_finished()
        return np.array(self.ex.compute(st), dtype=np.float32)

    # ------------------------------------------------------------------ #
    def extract(self, audio: np.ndarray, progress=lambda p: None, min_embed=0.6):
        """Segmentazione + impronte. Restituisce la lista di Item."""
        n = len(audio)
        starts = list(range(0, max(n - WIN, 0) + 1, STEP))
        if starts[-1] + WIN < n:
            starts.append(starts[-1] + STEP)
        items = []
        batch = 16
        for b0 in range(0, len(starts), batch):
            chunk_starts = starts[b0:b0 + batch]
            x = np.zeros((len(chunk_starts), 1, WIN), dtype=np.float32)
            for k, s in enumerate(chunk_starts):
                seg = audio[s:s + WIN]
                x[k, 0, :len(seg)] = seg
            y = self.seg.run(None, {"x": x})[0]          # (B, frames, 7)
            for k, s in enumerate(chunk_starts):
                items += self._window_items(audio, s, y[k].argmax(-1),
                                            first=(s == starts[0]), last=(s == starts[-1]),
                                            min_embed=min_embed)
            progress(min(1.0, (b0 + batch) / len(starts)))
        return items

    def _window_items(self, audio, start, cls, first, last, min_embed):
        nfr = len(cls)
        t = (start + np.arange(nfr) * FRAME_SHIFT + FRAME_OFFSET) / SR   # tempo di ogni frame
        w0 = start / SR
        lo = -1e9 if first else w0 + 2.5          # parte centrale (ogni istante appartiene a 1 finestra)
        hi = 1e9 if last else w0 + 7.5
        hi = min(hi, len(audio) / SR)
        out = []
        for c, local in POWERSET_SINGLE.items():
            mask = cls == c
            if mask.sum() * FRAME_SHIFT / SR < 0.2:
                continue
            # audio esclusivo di questa voce in tutta la finestra -> impronta
            idx = np.flatnonzero(mask)
            pieces, runs = [], []
            for a, b in _runs(idx):
                sa, sb = start + a * FRAME_SHIFT + FRAME_OFFSET - FRAME_SHIFT // 2, \
                    start + b * FRAME_SHIFT + FRAME_OFFSET + FRAME_SHIFT // 2
                pieces.append(audio[max(sa, 0):sb])
                ta, tb = max(t[a], lo), min(t[b], hi)
                if tb - ta > 0.05:
                    runs.append((float(ta), float(tb)))
            if not runs:
                continue
            samples = np.concatenate(pieces)
            it = Item(runs=_merge(runs, 0.3), speech=len(samples) / SR)
            if it.speech >= min_embed:
                it.emb = self.embed(samples)
            out.append(it)
        return out

    # ------------------------------------------------------------------ #
    @staticmethod
    def cluster(items, num_speakers=0, threshold=0.6, min_cluster_speech=4.0, center=True):
        """Raggruppa le impronte in persone. threshold = distanza coseno massima (auto)."""
        from scipy.cluster.hierarchy import linkage, fcluster
        strong = [i for i in items if i.emb is not None]
        if not strong:
            for i in items:
                i.label = 0
            return _result(items, {})
        X = _norm(np.stack([i.emb for i in strong]))
        X_raw = X
        if center and len(strong) > 2:
            # togli la componente comune a tutta la riunione (stanza, microfono, lingua):
            # resta solo ciò che distingue una persona dall'altra
            X = _norm(X - X.mean(0))
        if len(strong) == 1:
            labels = np.zeros(1, int)
        else:
            Z = linkage(X, method="average", metric="cosine")
            if num_speakers > 0:
                labels = fcluster(Z, t=num_speakers, criterion="maxclust") - 1
            else:
                labels = fcluster(Z, t=threshold, criterion="distance") - 1
        w = np.array([i.speech for i in strong])
        # rifinitura: riassegna ogni impronta al centroide più simile
        for _ in range(5):
            cents = {}
            for c in np.unique(labels):
                m = labels == c
                cents[c] = _norm((X[m] * w[m, None]).sum(0))
            # in automatico: gruppi con pochissimo parlato sono quasi sempre frammenti di un'altra voce
            if num_speakers <= 0 and len(cents) > 1:
                tiny = [c for c in cents if w[labels == c].sum() < min_cluster_speech]
                for c in tiny:
                    if len(cents) > 1:
                        del cents[c]
            keys = list(cents)
            C = np.stack([cents[k] for k in keys])
            new = np.array([keys[j] for j in (X @ C.T).argmax(1)])
            if np.array_equal(new, labels):
                break
            labels = new
        # rinumera 0..k-1 e calcola i centroidi finali
        remap = {c: k for k, c in enumerate(dict.fromkeys(labels.tolist()))}
        labels = np.array([remap[c] for c in labels])
        cents = {k: _norm((X[labels == k] * w[labels == k, None]).sum(0)) for k in range(len(remap))}
        raw = {k: _norm((X_raw[labels == k] * w[labels == k, None]).sum(0)) for k in range(len(remap))}
        for it, l in zip(strong, labels):
            it.label = int(l)
        # impronte deboli (troppo poco audio): prendono la voce del vicino nel tempo
        for it in items:
            if it.emb is None:
                it.label = -1
        _fill_weak(items)
        r = _result(items, cents)
        r.profiles = raw
        return r

    def __call__(self, audio, num_speakers=0, threshold=0.6, progress=lambda p: None):
        items = self.extract(audio, progress)
        return self.cluster(items, num_speakers, threshold)


# ---------------------------------------------------------------------- #
def _runs(idx):
    """Indici ordinati -> intervalli contigui [(a, b)]."""
    if len(idx) == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) > 1)
    a = np.r_[idx[0], idx[cuts + 1]]
    b = np.r_[idx[cuts], idx[-1]]
    return list(zip(a.tolist(), b.tolist()))


def _merge(runs, gap):
    runs = sorted(runs)
    out = [list(runs[0])]
    for a, b in runs[1:]:
        if a - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [tuple(r) for r in out]


def _fill_weak(items):
    """Le voci senza impronta prendono la persona del segmento più vicino nel tempo."""
    labeled = [(r, it.label) for it in items if it.label >= 0 for r in it.runs]
    if not labeled:
        for it in items:
            it.label = 0
        return
    for it in items:
        if it.label < 0:
            mid = (it.runs[0][0] + it.runs[-1][1]) / 2
            it.label = min(labeled, key=lambda x: min(abs(x[0][0] - mid), abs(x[0][1] - mid)))[1]


def _result(items, cents):
    segs = sorted((a, b, it.label) for it in items for a, b in it.runs)
    merged = []
    for a, b, l in segs:
        if merged and merged[-1][2] == l and a - merged[-1][1] <= 0.5:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b, l])
    talk = {}
    for a, b, l in merged:
        talk[l] = talk.get(l, 0) + b - a
    return Result(segments=[tuple(m) for m in merged], centroids=cents, talk=talk)


# ---------------------------------------------------------------------- #
#  Profili vocali: riconoscere le persone già nominate in riunioni precedenti
# ---------------------------------------------------------------------- #
def match_profiles(centroids: dict, profiles: dict, min_sim: float):
    """centroids: persona->impronta; profiles: nome->impronta.
    Assegna i nomi in modo univoco (prima le coppie più simili)."""
    if not centroids or not profiles:
        return {}
    names = list(profiles)
    P = _norm(np.stack([np.asarray(profiles[n]) for n in names]))
    pairs = []
    for c, v in centroids.items():
        sims = P @ _norm(v)
        for j, s in enumerate(sims):
            pairs.append((float(s), c, names[j]))
    pairs.sort(reverse=True)
    out, used = {}, set()
    for s, c, n in pairs:
        if s < min_sim:
            break
        if c in out or n in used:
            continue
        out[c] = (n, s)
        used.add(n)
    return out
