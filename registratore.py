"""
Quorum: registratore riunioni. Registra dal microfono, trascrive in locale (Whisper),
riconosce le voci dei partecipanti (diarizzazione + profili vocali) e prepara
il riassunto con un modello linguistico locale. Nessun dato lascia il PC.

Uso:
    pythonw registratore.py                 -> interfaccia grafica
    python  registratore.py FILE [PERSONE]  -> trascrive un file audio esistente
"""
import os
import sys
import re
import json
import queue
import threading
import time
import datetime
import traceback
from pathlib import Path

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
REC_DIR = BASE / "registrazioni"
SR = 16000  # frequenza usata da Whisper e dalla diarizzazione

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def _setup_cuda_dlls():
    """Rende visibili le DLL CUDA installate via pip (cuBLAS / cuDNN) per usare la GPU."""
    nv = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    for sub in ("cublas", "cudnn", "cuda_nvrtc"):
        p = nv / sub / "bin"
        if p.is_dir():
            try:
                os.add_dll_directory(str(p))
            except OSError:
                pass
            os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")


_setup_cuda_dlls()

import numpy as np  # noqa: E402
import sounddevice as sd  # noqa: E402
import soundfile as sf  # noqa: E402


# --------------------------------------------------------------------------- #
#  Utilità
# --------------------------------------------------------------------------- #
def hms(sec: float) -> str:
    sec = int(max(0, sec))
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def safe_name(text: str) -> str:
    return re.sub(r"[^\w\- ]", "", text, flags=re.UNICODE).strip()[:60]


def new_meeting_dir(title: str = "") -> Path:
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    name = stamp + (f"_{safe_name(title)}" if safe_name(title) else "")
    d = REC_DIR / name
    i = 2
    while d.exists():
        d = REC_DIR / f"{name}_{i}"
        i += 1
    d.mkdir(parents=True)
    return d


def input_devices():
    """Microfoni disponibili (solo API MME per evitare doppioni)."""
    apis = sd.query_hostapis()
    mme = next((i for i, a in enumerate(apis) if a["name"] == "MME"), None)
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and (mme is None or d["hostapi"] == mme):
            out.append((i, d["name"]))
    return out


# --------------------------------------------------------------------------- #
#  Registrazione
# --------------------------------------------------------------------------- #
class Recorder:
    """Registra il microfono direttamente su WAV (salvataggio continuo su disco)."""

    def __init__(self, path: Path, device=None):
        info = sd.query_devices(device, "input")
        self.sr = int(info["default_samplerate"]) or 44100
        self.path = path
        self.q = queue.Queue()
        self.level = 0.0
        self.paused = False
        self.frames = 0
        self._running = True
        self.file = sf.SoundFile(str(path), "w", samplerate=self.sr, channels=1, subtype="PCM_16")
        self.stream = sd.InputStream(samplerate=self.sr, channels=1, dtype="float32",
                                     device=device, callback=self._cb,
                                     blocksize=int(self.sr * 0.1))
        self._writer = threading.Thread(target=self._write_loop, daemon=True)

    def _cb(self, indata, frames, t, status):
        mono = indata[:, 0].copy()
        self.level = float(np.sqrt(np.mean(mono ** 2)))
        if not self.paused:
            self.q.put(mono)

    def _write_loop(self):
        last_flush = time.time()
        while self._running or not self.q.empty():
            try:
                data = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            self.file.write(data)
            self.frames += len(data)
            if time.time() - last_flush > 5:  # aggiorna l'header: file valido anche se il PC si spegne
                self.file.flush()
                last_flush = time.time()

    @property
    def seconds(self) -> float:
        return self.frames / self.sr

    def start(self):
        self._writer.start()
        self.stream.start()

    def stop(self):
        try:
            self.stream.stop()
            self.stream.close()
        finally:
            self._running = False
            self._writer.join()
            self.file.close()


# --------------------------------------------------------------------------- #
#  Trascrizione + riconoscimento voci
# --------------------------------------------------------------------------- #
_whisper = {"model": None, "device": None}


def load_whisper(device: str):
    from faster_whisper import WhisperModel
    if _whisper["model"] is not None and _whisper["device"] == device:
        return _whisper["model"]
    kw = dict(device=device, compute_type="float16" if device == "cuda" else "int8",
              download_root=str(MODELS / "whisper"))
    try:
        m = WhisperModel("large-v3-turbo", local_files_only=True, **kw)
    except Exception:
        m = WhisperModel("large-v3-turbo", **kw)  # primo avvio: scarica il modello
    _whisper.update(model=m, device=device)
    return m


# Riconoscimento voci: parametri scelti sul banco di prova (riunioni simulate 3-6 persone)
EMB_MODEL = "3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx"
DIAR_THRESHOLD = 0.8        # distanza coseno massima tra impronte della stessa persona
MIN_SPEAKER_SPEECH = 8.0    # secondi: sotto questa soglia una "voce" è un frammento di un'altra
PROFILE_MIN_SIM = 0.78      # somiglianza minima per riconoscere una voce già salvata
PROFILES_FILE = BASE / "profili_voci.json"
_diarizer = {}


def diarize(audio: np.ndarray, num_speakers: int, progress):
    from diarizzazione import Diarizer
    if "d" not in _diarizer:
        _diarizer["d"] = Diarizer(str(MODELS / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"),
                                  str(MODELS / "emb" / EMB_MODEL), threads=max(2, (os.cpu_count() or 4) - 2))
    d = _diarizer["d"]
    items = d.extract(audio, progress)
    return Diarizer.cluster(items, num_speakers, DIAR_THRESHOLD, MIN_SPEAKER_SPEECH)


def load_profiles() -> dict:
    """nome -> {"emb": [...], "n": riunioni, "model": ...} (solo profili dello stesso modello)."""
    try:
        allp = json.loads(PROFILES_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {k: v for k, v in allp.items() if v.get("model") == EMB_MODEL}


def update_profiles(names: dict, voices: dict):
    """Aggiorna i profili con le voci a cui l'utente ha dato un nome (media mobile)."""
    try:
        allp = json.loads(PROFILES_FILE.read_text(encoding="utf-8"))
    except Exception:
        allp = {}
    groups = {}
    for sid, name in names.items():
        if sid in voices and name and not re.fullmatch(r"Persona \d+", name):
            groups.setdefault(name, []).append(np.asarray(voices[sid], dtype=np.float32))
    for name, embs in groups.items():
        v = np.mean(embs, 0)
        v /= np.linalg.norm(v) + 1e-9
        old = allp.get(name)
        if old and old.get("model") == EMB_MODEL:
            n = min(old.get("n", 1), 9)           # le riunioni recenti pesano almeno il 10%
            v = (np.asarray(old["emb"]) * n + v) / (n + 1)
            v /= np.linalg.norm(v) + 1e-9
            allp[name] = {"emb": v.round(5).tolist(), "n": old.get("n", 1) + 1, "model": EMB_MODEL}
        else:
            allp[name] = {"emb": v.round(5).tolist(), "n": 1, "model": EMB_MODEL}
    if groups:
        PROFILES_FILE.write_text(json.dumps(allp, ensure_ascii=False), encoding="utf-8")


# Frasi "fantasma" tipiche di Whisper su silenzi/rumori
HALLUCINATIONS = re.compile(
    r"sottotitoli|amara\.org|qtss|grazie per la visione|iscriviti al canale", re.IGNORECASE)


def speaker_turns(dia, duration, gap=1.0, max_len=45.0):
    """Unisce i segmenti consecutivi della stessa voce in interventi da trascrivere."""
    if not dia:
        return [[0.0, duration, 0]]
    turns = []
    for s, e, spk in dia:
        last = turns[-1] if turns else None
        if last and last[2] == spk and s - last[1] <= gap and e - last[0] <= max_len:
            last[1] = max(last[1], e)
        else:
            turns.append([s, e, spk])
    return turns


def transcribe_turns(audio: np.ndarray, turns, progress, log):
    """Trascrive ogni intervento separatamente: niente passaggi saltati e voce gia' nota."""
    duration = len(audio) / SR
    opts = dict(language="it", beam_size=5, vad_filter=False, condition_on_previous_text=False,
                initial_prompt="Riunione di lavoro in italiano. Trascrizione accurata, con punteggiatura.")
    for device in ("cuda", "cpu"):
        try:
            model = load_whisper(device)
            out = []
            for i, (s, e, spk) in enumerate(turns):
                a, b = max(0.0, s - 0.25), min(duration, e + 0.25)
                clip = audio[int(a * SR):int(b * SR)]
                if len(clip) < SR * 0.3:
                    continue
                segs, _ = model.transcribe(clip, **opts)
                text = " ".join(x.text.strip() for x in segs
                                if x.no_speech_prob < 0.7 and not HALLUCINATIONS.search(x.text.strip()))
                if text.strip():
                    out.append((s, e, text, spk))
                progress((i + 1) / len(turns))
            return out, device
        except Exception as ex:
            if device == "cpu":
                raise
            log(f"GPU non disponibile ({ex.__class__.__name__}), uso la CPU (più lento)…")
            _whisper.update(model=None, device=None)


def build_turns(tagged):
    """Numera le voci in ordine di comparsa e unisce gli interventi consecutivi della stessa voce.
    Restituisce (interventi, {voce_diarizzazione: numero})."""
    order = {}
    turns = []
    for s, e, text, spk in tagged:
        if spk not in order:
            order[spk] = len(order) + 1
        sid = str(order[spk])
        if turns and turns[-1]["speaker"] == sid and s - turns[-1]["end"] < 30:
            turns[-1]["text"] += " " + text
            turns[-1]["end"] = e
        else:
            turns.append({"start": s, "end": e, "speaker": sid, "text": text})
    for t in turns:
        t["text"] = re.sub(r"\s+", " ", t["text"]).strip()
        t["start"], t["end"] = round(t["start"], 2), round(t["end"], 2)
    return turns, {k: str(v) for k, v in order.items()}


def render_txt(data: dict) -> str:
    names = data["names"]
    lines = [data.get("title") or "Riunione",
             f"Data: {data['date']}   Durata: {hms(data['duration'])}",
             "Partecipanti: " + ", ".join(dict.fromkeys(names[k] for k in sorted(names, key=int))),
             "", "-" * 70, ""]
    merged = []
    for t in data["turns"]:
        name = names.get(t["speaker"], f"Persona {t['speaker']}")
        if merged and merged[-1][1] == name:  # stessa persona (anche dopo un'unione di voci)
            merged[-1][2] += " " + t["text"]
        else:
            merged.append([t["start"], name, t["text"]])
    for start, name, text in merged:
        lines += [f"[{hms(start)}] {name}:", text, ""]
    return "\n".join(lines)


def save_outputs(out_dir: Path, data: dict) -> Path:
    (out_dir / "trascrizione.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    txt = out_dir / "trascrizione.txt"
    txt.write_text(render_txt(data), encoding="utf-8")
    return txt


def process_meeting(audio_path: Path, out_dir: Path, num_speakers: int = 0, title: str = "",
                    status=lambda msg, pct: None) -> Path:
    from faster_whisper import decode_audio
    t0 = time.time()
    status("Carico l'audio…", 0)
    audio = decode_audio(str(audio_path), sampling_rate=SR)
    duration = len(audio) / SR
    if duration < 1:
        raise RuntimeError("Registrazione vuota o troppo corta.")

    status("Riconosco le voci…", 2)
    dia = diarize(audio, num_speakers, lambda p: status(None, 2 + p * 28))

    pieces = speaker_turns([tuple(x) for x in dia.segments], duration)
    status(f"Trascrivo {len(pieces)} interventi…", 30)
    tagged, device = transcribe_turns(audio, pieces, lambda p: status(None, 30 + p * 68),
                                      lambda m: status(m, None))
    status("Compongo il testo…", 98)
    turns, order = build_turns(tagged)
    speakers = sorted({t["speaker"] for t in turns}, key=int)
    voices = {order[k]: np.round(v, 5).tolist() for k, v in dia.profiles.items() if k in order}
    names = {s: f"Persona {s}" for s in speakers}
    recognized = {}
    from diarizzazione import match_profiles
    prof = load_profiles()
    for sid, (name, sim) in match_profiles({k: v for k, v in voices.items() if k in names},
                                           {n: p["emb"] for n, p in prof.items()}, PROFILE_MIN_SIM).items():
        names[sid] = name
        recognized[sid] = round(sim, 3)
    created = datetime.datetime.fromtimestamp(audio_path.stat().st_mtime - duration)
    data = {
        "title": title or audio_path.parent.name,
        "date": created.strftime("%d/%m/%Y %H:%M"),
        "duration": duration,
        "audio": audio_path.name,
        "names": names,
        "recognized": recognized,
        "voices": voices,
        "turns": turns,
        "processing": {"device": device, "seconds": round(time.time() - t0, 1)},
    }
    txt = save_outputs(out_dir, data)
    _whisper.update(model=None, device=None)   # libera la memoria della GPU per il riassunto
    status(f"Fatto in {hms(time.time() - t0)} ({'GPU' if device == 'cuda' else 'CPU'})", 100)
    return txt


# --------------------------------------------------------------------------- #
#  Interfaccia grafica
# --------------------------------------------------------------------------- #
def run_gui():
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog

    REC_DIR.mkdir(exist_ok=True)
    root = tk.Tk()
    root.title("Quorum")
    root.geometry("520x510")
    root.minsize(480, 490)
    try:
        root.tk.call("tk", "scaling", 1.3)
    except tk.TclError:
        pass
    style = ttk.Style()
    if "vista" in style.theme_names():
        style.theme_use("vista")

    ui_q: "queue.Queue[tuple]" = queue.Queue()
    jobs: "queue.Queue[tuple]" = queue.Queue()
    state = {"rec": None, "dir": None, "busy": False}

    # ---- lavoratore in background (una trascrizione alla volta) ----
    def worker():
        while True:
            job = jobs.get()
            ui_q.put(("busy", True))
            if job[0] == "summary":
                out_dir = job[1]
                try:
                    import riassunto
                    data = json.loads((out_dir / "trascrizione.json").read_text(encoding="utf-8"))
                    path = riassunto.write_summary(
                        out_dir, data,
                        status=lambda m, f: ui_q.put(("status", m, None if f is None else f * 100)))
                    ui_q.put(("status", "Riassunto pronto.", 100))
                    os.startfile(path)
                except Exception as e:
                    traceback.print_exc()
                    ui_q.put(("error", f"Riassunto non riuscito: {e}"))
                ui_q.put(("busy", jobs.qsize() > 0))
                continue
            audio_path, out_dir, n_spk, title = job
            try:
                txt = process_meeting(audio_path, out_dir, n_spk, title,
                                      status=lambda m, p: ui_q.put(("status", m, p)))
                ui_q.put(("done", out_dir, txt))
            except Exception as e:
                traceback.print_exc()
                (out_dir / "errore.txt").write_text(traceback.format_exc(), encoding="utf-8")
                ui_q.put(("error", f"{e.__class__.__name__}: {e}"))
            ui_q.put(("busy", jobs.qsize() > 0))

    threading.Thread(target=worker, daemon=True).start()

    # ---- widget ----
    pad = dict(padx=14, pady=4)
    frm = ttk.Frame(root)
    frm.pack(fill="both", expand=True, pady=8)

    ttk.Label(frm, text="Titolo riunione (facoltativo)").pack(anchor="w", **pad)
    title_var = tk.StringVar()
    ttk.Entry(frm, textvariable=title_var).pack(fill="x", **pad)

    row = ttk.Frame(frm)
    row.pack(fill="x", **pad)
    ttk.Label(row, text="Microfono").grid(row=0, column=0, sticky="w")
    ttk.Label(row, text="Persone").grid(row=0, column=1, sticky="w", padx=(10, 0))
    devs = input_devices()
    dev_var = tk.StringVar()
    dev_box = ttk.Combobox(row, textvariable=dev_var, state="readonly",
                           values=[n for _, n in devs])
    try:
        default_name = sd.query_devices(sd.default.device[0])["name"]
        dev_box.current(next(i for i, (_, n) in enumerate(devs) if n == default_name))
    except Exception:
        if devs:
            dev_box.current(0)
    dev_box.grid(row=1, column=0, sticky="ew")
    spk_var = tk.StringVar(value="Automatico")
    ttk.Combobox(row, textvariable=spk_var, state="readonly", width=11,
                 values=["Automatico"] + [str(i) for i in range(1, 11)]).grid(row=1, column=1, padx=(10, 0))
    row.columnconfigure(0, weight=1)

    timer_lbl = tk.Label(frm, text="00:00:00", font=("Segoe UI", 34, "bold"), fg="#333")
    timer_lbl.pack(pady=(12, 0))
    level = tk.Canvas(frm, height=10, bg="#e5e5e5", highlightthickness=0)
    level.pack(fill="x", padx=40, pady=(2, 10))
    level_bar = level.create_rectangle(0, 0, 0, 10, fill="#3a9d5d", width=0)

    btns = ttk.Frame(frm)
    btns.pack(pady=4)
    rec_btn = tk.Button(btns, text="●  Registra", font=("Segoe UI", 13, "bold"), width=13,
                        bg="#d64545", fg="white", activebackground="#b33", relief="flat",
                        cursor="hand2")
    rec_btn.grid(row=0, column=0, padx=5)
    pause_btn = tk.Button(btns, text="❚❚  Pausa", font=("Segoe UI", 11), width=10, relief="flat",
                          bg="#ddd", state="disabled", cursor="hand2")
    pause_btn.grid(row=0, column=1, padx=5)

    status_var = tk.StringVar(value="Pronto.")
    ttk.Label(frm, textvariable=status_var, foreground="#555").pack(anchor="w", padx=14, pady=(14, 2))
    prog = ttk.Progressbar(frm, maximum=100)
    prog.pack(fill="x", padx=14)

    import riassunto
    sum_var = tk.BooleanVar(value=riassunto.available())
    ttk.Checkbutton(frm, text=f"Crea il riassunto dopo la trascrizione (locale, {riassunto.MODEL})",
                    variable=sum_var).pack(anchor="w", padx=14, pady=(10, 0))

    tools = ttk.Frame(frm)
    tools.pack(fill="x", padx=14, pady=(10, 0))
    for c in range(4):
        tools.columnconfigure(c, weight=1)

    # ---- azioni ----
    def n_speakers():
        v = spk_var.get()
        return int(v) if v.isdigit() else 0

    def start_rec():
        idx = dev_box.current()
        device = devs[idx][0] if idx >= 0 else None
        out_dir = new_meeting_dir(title_var.get())
        try:
            rec = Recorder(out_dir / "audio.wav", device)
            rec.start()
        except Exception as e:
            out_dir.rmdir() if not any(out_dir.iterdir()) else None
            messagebox.showerror("Microfono", f"Impossibile avviare la registrazione:\n{e}")
            return
        state.update(rec=rec, dir=out_dir)
        rec_btn.config(text="■  Stop", bg="#444", activebackground="#222")
        pause_btn.config(state="normal", text="❚❚  Pausa")
        dev_box.config(state="disabled")
        timer_lbl.config(fg="#d64545")
        status_var.set(f"Registrazione in corso → {out_dir.name}")

    def stop_rec(process=True):
        rec, out_dir = state["rec"], state["dir"]
        if not rec:
            return
        rec.stop()
        state.update(rec=None)
        rec_btn.config(text="●  Registra", bg="#d64545", activebackground="#b33")
        pause_btn.config(state="disabled", text="❚❚  Pausa")
        dev_box.config(state="readonly")
        timer_lbl.config(fg="#333")
        if process:
            status_var.set("Registrazione salvata, in coda per la trascrizione…")
            jobs.put((out_dir / "audio.wav", out_dir, n_speakers(), title_var.get().strip()))
            title_var.set("")

    def toggle_rec():
        stop_rec() if state["rec"] else start_rec()

    def toggle_pause():
        rec = state["rec"]
        if rec:
            rec.paused = not rec.paused
            pause_btn.config(text="▶  Riprendi" if rec.paused else "❚❚  Pausa")
            timer_lbl.config(fg="#c28a00" if rec.paused else "#d64545")

    def transcribe_file():
        f = filedialog.askopenfilename(title="Scegli un file audio/video", filetypes=[
            ("Audio/Video", "*.wav *.mp3 *.m4a *.ogg *.flac *.mp4 *.mkv *.webm *.aac *.wma"),
            ("Tutti i file", "*.*")])
        if not f:
            return
        src = Path(f)
        out_dir = new_meeting_dir(title_var.get() or src.stem)
        jobs.put((src, out_dir, n_speakers(), title_var.get().strip() or src.stem))
        title_var.set("")
        status_var.set(f"In coda: {src.name}")

    def rename_existing():
        d = filedialog.askdirectory(title="Cartella della riunione", initialdir=str(REC_DIR))
        if d and (Path(d) / "trascrizione.json").exists():
            rename_dialog(Path(d))
        elif d:
            messagebox.showwarning("Rinomina voci", "In questa cartella non c'è una trascrizione.")

    rec_btn.config(command=toggle_rec)
    pause_btn.config(command=toggle_pause)
    ttk.Button(tools, text="Trascrivi file…", command=transcribe_file).grid(row=0, column=0, sticky="ew", padx=2)
    ttk.Button(tools, text="Rinomina voci…", command=rename_existing).grid(row=0, column=1, sticky="ew", padx=2)
    def summary_existing():
        d = filedialog.askdirectory(title="Cartella della riunione", initialdir=str(REC_DIR))
        if not d:
            return
        if not (Path(d) / "trascrizione.json").exists():
            messagebox.showwarning("Riassunto", "In questa cartella non c'è una trascrizione.")
        elif not riassunto.available():
            messagebox.showwarning("Riassunto", f"Ollama non è avviato o manca il modello {riassunto.MODEL}.")
        else:
            jobs.put(("summary", Path(d)))
            status_var.set("Riassunto in coda…")

    ttk.Button(tools, text="Riassunto…", command=summary_existing).grid(row=0, column=2, sticky="ew", padx=2)
    ttk.Button(tools, text="Apri cartella", command=lambda: os.startfile(REC_DIR)).grid(row=0, column=3, sticky="ew", padx=2)

    # ---- finestra per dare un nome alle voci ----
    def rename_dialog(out_dir: Path, auto_summary=False):
        data = json.loads((out_dir / "trascrizione.json").read_text(encoding="utf-8"))
        audio_file = out_dir / data.get("audio", "audio.wav")
        win = tk.Toplevel(root)
        win.title(f"Chi è chi? – {out_dir.name}")
        win.transient(root)
        ttk.Label(win, text="Ascolta un campione di ogni voce e scrivi il nome.\n"
                            "Dai lo stesso nome a due voci per unirle.\n"
                            "I nomi salvati vengono ricordati: nelle prossime riunioni\n"
                            "le voci già note saranno riconosciute da sole (✓).",
                  justify="left").pack(anchor="w", padx=12, pady=(10, 6))
        body = ttk.Frame(win)
        body.pack(fill="both", expand=True, padx=12)
        entries = {}
        for r, sid in enumerate(sorted(data["names"], key=int)):
            mine = [t for t in data["turns"] if t["speaker"] == sid]
            talk = sum(t["end"] - t["start"] for t in mine)
            sample = max(mine, key=lambda t: len(t["text"])) if mine else None
            var = tk.StringVar(value=data["names"][sid])
            entries[sid] = var
            ttk.Entry(body, textvariable=var, width=18).grid(row=r * 2, column=0, sticky="w", pady=(6, 0))
            rec_sim = data.get("recognized", {}).get(sid)
            ttk.Label(body, text=("✓ riconosciuto  " if rec_sim else "") + f"parla per {hms(talk)}",
                      foreground="#2e7d32" if rec_sim else "#777").grid(row=r * 2, column=1, padx=8)

            def play(t=sample):
                if not t or not audio_file.exists():
                    return
                info = sf.info(str(audio_file))
                a, b = t["start"], min(t["end"], t["start"] + 8)
                data_, sr_ = sf.read(str(audio_file), start=int(a * info.samplerate),
                                     stop=int(b * info.samplerate), dtype="float32")
                sd.stop()
                sd.play(data_, sr_)

            ttk.Button(body, text="▶ ascolta", width=9, command=play).grid(row=r * 2, column=2)
            if sample:
                txt = sample["text"][:110] + ("…" if len(sample["text"]) > 110 else "")
                ttk.Label(body, text=f"«{txt}»", foreground="#555", wraplength=420).grid(
                    row=r * 2 + 1, column=0, columnspan=3, sticky="w")

        def finish():
            sd.stop()
            win.destroy()
            os.startfile(out_dir / "trascrizione.txt")
            if auto_summary and sum_var.get() and not (out_dir / "riassunto.md").exists():
                if riassunto.available():
                    jobs.put(("summary", out_dir))
                    status_var.set("Riassunto in preparazione…")
                else:
                    status_var.set(f"Riassunto saltato: Ollama non attivo o manca {riassunto.MODEL}.")

        def save():
            data["names"] = {k: (v.get().strip() or f"Persona {k}") for k, v in entries.items()}
            save_outputs(out_dir, data)
            try:
                update_profiles(data["names"], data.get("voices", {}))
            except Exception:
                traceback.print_exc()
            finish()

        bar = ttk.Frame(win)
        bar.pack(fill="x", padx=12, pady=12)
        ttk.Button(bar, text="Salva e apri", command=save).pack(side="right")
        ttk.Button(bar, text="Apri senza nomi",
                   command=finish).pack(side="right", padx=6)

    # ---- ciclo di aggiornamento UI ----
    def tick():
        rec = state["rec"]
        if rec:
            timer_lbl.config(text=hms(rec.seconds))
            w = level.winfo_width()
            lvl = 0 if rec.paused else min(1.0, rec.level * 8)
            level.coords(level_bar, 0, 0, w * lvl, 10)
        else:
            level.coords(level_bar, 0, 0, 0, 10)
        while True:
            try:
                msg = ui_q.get_nowait()
            except queue.Empty:
                break
            kind = msg[0]
            if kind == "status":
                if msg[1]:
                    status_var.set(msg[1])
                if msg[2] is not None:
                    prog["value"] = msg[2]
            elif kind == "busy":
                state["busy"] = msg[1]
            elif kind == "done":
                prog["value"] = 100
                rename_dialog(msg[1], auto_summary=True)
            elif kind == "error":
                prog["value"] = 0
                status_var.set("Errore durante la trascrizione (l'audio è salvato).")
                messagebox.showerror("Trascrizione", msg[1])
        root.after(100, tick)

    def on_close():
        if state["rec"]:
            if not messagebox.askyesno("Registrazione in corso",
                                       "Stai registrando. Fermare e salvare l'audio?\n"
                                       "(potrai trascriverlo dopo con «Trascrivi file…»)"):
                return
            stop_rec(process=False)
        elif state["busy"] and not messagebox.askyesno(
                "Trascrizione in corso", "C'è una trascrizione in corso. Chiudere comunque?\n"
                                         "(l'audio resta salvato)"):
            return
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.bind("<F9>", lambda e: toggle_rec())
    tick()
    root.mainloop()


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--file":          # sintassi storica, ancora accettata
        args = args[1:]
    if args:
        src = Path(args[0])
        n = int(args[1]) if len(args) > 1 else 0
        REC_DIR.mkdir(exist_ok=True)
        out = new_meeting_dir(src.stem)
        txt = process_meeting(src, out, n, src.stem,
                              status=lambda m, p: print(f"[{p if p is None else round(p)}%] {m}") if m else None)
        print(txt.read_text(encoding="utf-8"))
    else:
        run_gui()
