<div style="background-color: #ffffff; color: #000000; padding: 10px;">
<img src="img\logo_aisc_bmftr.jpg">
</div>

# FrisperWhisper: Transkription mit Fine-Tuned Whisper-Modellen

## Inhaltsverzeichnis

1. [Einführung](#1-einführung)
2. [Installation und Setup](#2-installation-und-setup)
3. [Transkription](#3-transkription)
4. [Modellvergleich und Evaluierung](#4-modellvergleich-und-evaluierung)
5. [SLURM-Integration](#5-slurm-integration)

**📋 Für Whisper Fine-Tuning:** Siehe [README_fine-tuning.md](README_fine-tuning.md) für komplette Anleitung zum Fine-Tuning von Whisper-Modellen.

---

## 1. Einführung

Dieses Projekt ermöglicht die Transkription französischer Audioaufnahmen mit fine-tuned Whisper-Modellen. Die Hauptfunktionalitäten umfassen:

- **Transkription einzelner Dateien oder Verzeichnisse** mit lokalen oder HuggingFage Hub-Modellen
- **Systematischer Modellvergleich** zwischen Whisper Large V3 und fine-tuned Modellen
- **Umfassende Evaluierung** mit WER, CER und BLEU-Metriken
- **CSV-Export** mit Zeitstempel-Informationen und Metadaten
- **SLURM-Integration** für Cluster-Processing

---

## 2. Installation und Setup

### 2.1 UV Virtual Environment

Überprüfung und Installation von UV:

```bash
$ which uv || echo "UV not found"
$ curl -LsSf https://astral.sh/uv/install.sh | sh  # Falls UV nicht installiert
```

Erstellen und Aktivieren der virtuellen Umgebung:

```bash
$ uv venv .venv
$ source .venv/bin/activate
(.venv)$ uv sync --active
```

### 2.2 Datenstruktur

```
pilotproject-frisperwhisper/
├── data/                           # Hauptdatenverzeichnis
│   ├── LangAge/                    # Originale LangAge-Daten (16kHz)
│   ├── ESLO/                       # ESLO 30-39 Jahre Daten
│   └── LangAgeDataSet/             # HuggingFace Dataset (für Train/Test-Info)
├── FrisperWhisper/                 # Fine-tuned Modelle
│   └── largeV1/                    # Modell-Checkpoints
├── results/                        # Transkriptionsergebnisse
└── scripts/                        # Python-Skripte
```

---

## 3. Transkription

### 3.1 Einzelne Audio-Transkription

Das Skript `transcribe_with_finetuned.py` transkribiert Audio-Dateien mit CSV-Output:

**Lokales Fine-Tuned Modell:**

```bash
(.venv)$ python scripts/transcribe_with_finetuned.py \
    -i data/LangAge16kHz/a001a.wav \
    -m FrisperWhisper/largeV3/checkpoint-2000 \
    -o transcription_result.csv \
    --language french
```

**HuggingFace Hub Modell:**

```bash
(.venv)$ python scripts/transcribe_with_finetuned.py \
    -i data/LangAge16kHz/a001a.wav \
    -m openai/whisper-large-v3 \
    -o transcription_result.csv \
    --use_pipeline \
    --device cuda
```

### 3.2 Verzeichnis-Transkription

**Ganzes Verzeichnis verarbeiten:**

```bash
(.venv)$ python scripts/transcribe_with_finetuned.py \
    -i data/ESLO \
    -m FrisperWhisper/largeV3/checkpoint-2000 \
    -o eslo_transcriptions.csv \
    --language french
```

**Mit automatischer Segmentierung (für lange Audios):**

```bash
(.venv)$ python scripts/transcribe_with_finetuned.py \
    -i data/ESLO \
    -m openai/whisper-large-v3 \
    -o eslo_transcriptions.csv \
    --use_pipeline \
    --device cuda
```

### 3.3 Parameter und Optionen

| Parameter | Beschreibung | Beispiel |
|-----------|-------------|----------|
| `-i, --input` | Audio-Datei oder Verzeichnis | `data/audio.wav` |
| `-m, --model` | Lokaler Pfad oder HuggingFace Model-ID | `openai/whisper-large-v3` |
| `-o, --output` | Ausgabe-CSV-Datei | `results.csv` |
| `--language` | Sprache (Standard: `french`) | `--language french` |
| `--device` | Gerät (`cpu`, `cuda`, `auto`) | `--device cuda` |
| `--use_pipeline` | HuggingFace Pipeline für lange Audios | `--use_pipeline` |

### 3.4 CSV-Format

**Einzelne Datei:**
```csv
ID,Start,Stop,Transcription
1,00:00.000,00:12.500,"Bonjour, comment allez-vous?"
2,00:12.500,00:25.320,"Je vais très bien, merci."
```

**Verzeichnis (mehrere Dateien):**
```csv
ID,Filename,Start,Stop,Transcription
1,audio1.wav,00:00.000,00:12.500,"Bonjour, comment allez-vous?"
2,audio1.wav,00:12.500,00:25.320,"Je vais très bien, merci."
3,audio2.wav,00:00.000,00:08.100,"C'est magnifique."
```

### 3.5 Unterstützte Formate

**Audio-Formate:** `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.opus`

---

## Zusätzliche Informationen

### Fine-Tuning

Für komplette Anleitung zum Fine-Tuning von Whisper-Modellen siehe: **[README_fine-tuning.md](README_fine-tuning.md)**

### Support

Bei Problemen oder Fragen zur Transkription:
1. Prüfen Sie die Log-Dateien in den Output-Verzeichnissen
2. Überprüfen Sie CUDA-Verfügbarkeit für GPU-Verarbeitung
3. Stellen Sie sicher, dass alle Pfade korrekt sind