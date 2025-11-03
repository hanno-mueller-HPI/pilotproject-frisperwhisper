# Fine-Tuning Whisper für französische Sprache

> **Hinweis:** Für die Nutzung von fine-tuned Modellen zur Transkription siehe [README.md](README.md).

## Inhaltsverzeichnis

1. [Preambel](#1-preambel)
   - [1.1. Anforderungen](#11-anforderungen)
2. [Fine-Tuning Whisper](#2-fine-tuning-whisper)
   - [2.1. Datenaufbereitung](#21-datenaufbereitung)
     - [2.1.1. TRS zu TextGrid + Dataset Kombination](#211-trs-zu-textgrid--dataset-kombination)
     - [2.1.2. ESLO-TextGrid zu LangAge-TextGrid](#212-eslo-textgrid-zu-langage-textgrid)
     - [2.1.3. Dataset Kombination](#213-dataset-kombination)
     - [2.1.4. Audio-Resampling](#214-audio-resampling)
     - [2.1.5. Dataset Dictionary](#215-dataset-dictionary)
     - [2.1.6. Log-Mel Spektrogramme](#216-log-mel-spektrogramme)
   - [2.2. Fine-Tuning](#22-fine-tuning)
3. [Fine-Tuning mit SLURM](#3-fine-tuning-mit-slurm)
   - [3.1. Datenaufbereitung](#31-datenaufbereitung)
   - [3.2. Fine-Tuning](#32-fine-tuning)
4. [Daten Vergleichen](#4-daten-vergleichen)
5. [Sonstiges](#5-sonstiges)
   - [5.1. Daten-Anomalien erkennen](#51-daten-anomalien-erkennen)

---

## 1. Preambel

Dieses Dokument beschreibt das Fine-Tuning eines Whisper-Modells für französische Sprache mit dem [LangAge](https://www.uni-potsdam.de/en/langage-corpora/access-to-corpus)-Korpus und [ESLO](http://eslo.huma-num.fr/)-Daten. Das resultierende Modell transkribiert Interjektionen, Wortwiederholungen und abgebrochene Wörter.

**Workflow-Übersicht basierend auf LangAge und ESLO:**

Der Fine-Tuning-Prozess kombiniert zwei französische Sprachdatensätze:
- **LangAge-Korpus**: Hauptdatensatz mit demographisch-strukturierten Sprachdaten
- **ESLO-Daten**: Ergänzungsdaten für spezifische Altersgruppen (26-46 Jahre inkl. 30-39 Jahre, 65+ Jahre)

Die Kombination erfolgt durch symbolische Links, wodurch ein einheitlicher Datensatz für das Training entsteht, ohne Speicherplatz zu verschwenden.

### 1.1. Anforderungen

Um sicherzustellen, dass über verschiedene Betriebssysteme und Maschinen hinweg die gleichen Anforderungen erfüllt sind, wird das Anlegen einer *virtual environment* empfohlen. Diese lässt sich mit UV einrichten.

```bash
$ which uv || echo "UV not found" # überprüft die UV Installation
```

Sollte UV nicht installiert sein, lässt es sich wie folgt installieren:

```bash
$ curl -LsSf https://astral.sh/uv/install.sh | sh
```

Anschließend kann die virtuelle Umgebung erstellt und aktiviert werden:

```bash
$ uv venv .venv # erstellt eine virtual environment mit dem Namen ".venv"
$ source .venv/bin/activate # aktiviert die virtual environment
```

Dann werden die benötigten Pakete installiert. UV sorgt dafür, dass die exakten Versionen installiert werden:

```bash
(.venv)$ uv sync --active  # installiert exakte Versionen
```

---

## 2. Fine-Tuning Whisper

### 2.1. Datenaufbereitung

Downloade oder hinterlege die LangAge und die ESLO Daten in die folgenden beiden Ordner:

```bash
(.venv)$ mkdir -p data/LangAge
(.venv)$ mkdir -p data/ESLO
```

#### 2.1.1. TRS zu TextGrid + Dataset Kombination

**TRS zu TextGrid Konvertierung**

Falls die TextGrid-Dateien nicht vorhanden sind oder aktualisiert werden müssen, konvertiere die TRS-Dateien zu TextGrids:

```bash
(.venv)$ python scripts/trs2tg.py data/LangAge
```

#### 2.1.2. ESLO-TextGrid zu LangAge-TextGrid

ESLO-TextGrid-Dateien müssen in das LangAge-Format konvertiert werden für einheitliche Verarbeitung:

```bash
(.venv)$ python scripts/transformTextgrids2LangAgeFormat.py --input data/ESLO
```

**Hinweis:** Dieses Skript harmonisiert die TextGrid-Strukturen zwischen den beiden Datensätzen. Die originalen ESLO-TextGrids werden dabei überschrieben.

#### 2.1.3. Dataset Kombination

Erstelle eine kombinierte Struktur mit symbolischen Links:

```bash
# Erstelle kombiniertes Verzeichnis
(.venv)$ mkdir -p data/ESLOLangAgeCombined

# Füge symbolische Links hinzu
(.venv)$ ln -s ../LangAge/* ./data/ESLOLangAgeCombined/
(.venv)$ ln -s ../ESLO/* ./data/ESLOLangAgeCombined/
```

Dies erstellt ein einheitliches Verzeichnis mit allen Daten ohne Duplizierung.

#### 2.1.4. Audio-Resampling

Audio-Dateien werden von 44,1 kHz auf 16 kHz heruntergetastet (Whisper-Anforderung):

```bash
(.venv)$ cp data/ESLOLangAgeCombined data/ESLOLangAgeCombined16kHz
(.venv)$ python scripts/resample44k16k.py -i data/ESLOLangAgeCombined16kHz -p 40
```

**Parameter:**
- `-i`: Eingabeverzeichnis
- `-p`: Anzahl paralleler Prozesse (empfohlen: 50% der verfügbaren Kerne)

**Hinweis:** Dieser Schritt überschreibt die ursprünglichen Dateien mit den resampelten Versionen. Deshalb werden die Dateien vorher kopiert

#### 2.1.5. Dataset Dictionary

Erstelle ein HuggingFace Dataset Dictionary mit Train/Test-Splits:

```bash
(.venv)$ python scripts/Textgrids2DatasetBatch.py \
    -f data/ESLOLangAgeCombined16kHz \
    -o data/ESLOLangAgeDataSet \
    -n 150 \
    --batch_size 500 \
    --audio_batch_processes 8
```

**Parameter:**
- `-f`: Verzeichnis mit TextGrid-Dateien
- `-o`: Ausgabeverzeichnis für Dataset Dictionary
- `-n`: Anzahl paralleler Prozesse
- `--batch_size`: Batch-Größe für Verarbeitung
- `--audio_batch_processes`: Audio-Verarbeitungsprozesse

**Automatische Datenbereinigung:**
- Entfernung kurzer Segmente (< 100ms)
- Filterung stiller Audio-Segmente
- Entfernung von "(buzz)", "XXX"-Mustern und Klammer-Inhalten
- Sprecher-Filter (z.B. Interviewer ausschließen)

#### 2.1.6. Log-Mel Spektrogramme

Konvertiere Audio zu Log-Mel Spektrogrammen (Whisper-Eingabeformat):

```bash
(.venv)$ python scripts/Dataset2LogMelSpecBatch.py \
    -i data/ESLOLangAgeDataSet \
    -o data/ESLOLangAgeLogMelSpec \
    --model_size large-v3 \
    --num_cpus 150 \
    --batch_size 1000
```

**Parameter:**
- `-i`: Eingabe-Dataset Dictionary
- `-o`: Ausgabe-Dataset mit Log-Mel Spektrogrammen
- `--model_size`: Whisper-Modellgröße (`large-v3` verwendet 128 mel bins)
- `--num_cpus`: CPU-Kerne für Verarbeitung
- `--batch_size`: Verarbeitungs-Batch-Größe

### 2.2. Fine-Tuning

Starte das eigentliche Fine-Tuning:

```bash
(.venv)$ python scripts/finetune_whisper_from_LogMel.py \
    --dataset_path data/ESLOLangAgeLogMelSpec \
    --output_dir FrisperWhisper \
    --version v1 \
    --model_size large-v3 \
    --num_gpus 4 \
    --num_cpus 40 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16
```

**Checkpoint-Management:**

Fortsetzen von vorherigem Training:

```bash
# Automatisch vom letzten Checkpoint
(.venv)$ python scripts/finetune_whisper_from_LogMel.py \
    --dataset_path data/ESLOLangAgeLogMelSpec \
    --output_dir FrisperWhisper \
    --version v1 \
    --resume_from_checkpoint true

# Von spezifischem Checkpoint
(.venv)$ python scripts/finetune_whisper_from_LogMel.py \
    --dataset_path data/ESLOLangAgeLogMelSpec \
    --output_dir FrisperWhisper \
    --version v1 \
    --resume_from_checkpoint checkpoint-1000
```

## 3. Fine-Tuning mit SLURM

### 3.1. Datenaufbereitung

**Dataset-Erstellung für kombinierte LangAge+ESLO Daten:**

```bash
(.venv)$ INPUT_FOLDER=data/ESLOLangAgeCombined16kHz \
         OUTPUT_FOLDER=data/ESLOLangAgeDataSet \
         NUM_PROCESSES=150 \
         BATCH_SIZE=500 \
         AUDIO_BATCH_PROCESSES=8 \
         sbatch scripts/create_dataset.sbatch
```

**Log-Mel Spektrogramm-Erstellung:**

```bash
(.venv)$ INPUT_DATASET=data/ESLOLangAgeDataSet \
         OUTPUT_DATASET=data/ESLOLangAgeLogMelSpec \
         MODEL_SIZE=large-v3 \
         NUM_CPUS=150 \
         BATCH_SIZE=1000 \
         sbatch scripts/run_dataset_preprocess_batch.sbatch
```

**Parameter für Dataset-Erstellung:**
- `INPUT_FOLDER`: Pfad zum Eingabeordner mit TextGrid- und Audio-Dateien (Standard: `data/LangAge16kHz`)
- `OUTPUT_FOLDER`: Pfad zum Ausgabeordner für das Dataset Dictionary (Standard: `data/LangAgeDataSet`)
- `NUM_PROCESSES`: Anzahl paralleler Prozesse für TextGrid-Verarbeitung (Standard: `120`)
- `BATCH_SIZE`: Batch-Größe für die Verarbeitung von Einträgen (Standard: `500`)
- `AUDIO_BATCH_PROCESSES`: Anzahl der Prozesse für Audio-Batch-Verarbeitung (Standard: `8`)
- `CSV_FILE`: CSV-Datei mit Test-Set Intervallen (optional)

**Parameter für Log-Mel Spektrogramme:**
- `INPUT_DATASET`: Pfad zum Eingabe-Dataset Dictionary
- `OUTPUT_DATASET`: Pfad zum Ausgabe-Dataset mit Log-Mel Spektrogrammen
- `MODEL_SIZE`: Whisper-Modellgröße (`large-v3` empfohlen für 128 mel bins)
- `NUM_CPUS`: Anzahl CPU-Kerne für Verarbeitung (Standard: `100`)
- `BATCH_SIZE`: Batch-Größe für Dataset-Chunks (Standard: `400`)
- `WRITER_BATCH_SIZE`: Batch-Größe für Festplattenspeicherung (Standard: `100`)
- `MAX_MEMORY_PER_WORKER`: Maximaler Speicher pro Worker in GB (Standard: `4.0`)
- `MAX_SAMPLES`: Maximale Anzahl Samples pro Split (optional, für Tests)

### 3.2. Fine-Tuning

**Training mit kombinierten LangAge+ESLO Daten:**

```bash
(.venv)$ DATASET_PATH=data/ESLOLangAgeLogMelSpec \
         OUTPUT_DIR=FrisperWhisper/largeV3_ESLOLangAge \
         MODEL_SIZE=large-v3 \
         NUM_GPUS=4 \
         NUM_CPUS=24 \
         MAX_STEPS=15000 \
         LEARNING_RATE=1.5e-5 \
         WARMUP_STEPS=1500 \
         sbatch scripts/train_whisper.sbatch
```

**Training von Checkpoint fortsetzen:**

```bash
# Automatisch vom letzten Checkpoint
(.venv)$ DATASET_PATH=data/ESLOLangAgeLogMelSpec \
         OUTPUT_DIR=FrisperWhisper/largeV3_ESLOLangAge \
         RESUME_CHECKPOINT=true \
         sbatch scripts/train_whisper.sbatch

# Von spezifischem Checkpoint
(.venv)$ DATASET_PATH=data/ESLOLangAgeLogMelSpec \
         OUTPUT_DIR=FrisperWhisper/largeV3_ESLOLangAge \
         RESUME_CHECKPOINT=checkpoint-2000 \
         sbatch scripts/train_whisper.sbatch
```

**Anpassbare Training-Parameter:**
- `DATASET_PATH`: Pfad zum Dataset mit Log-Mel Spektrogrammen (Standard: `data/LangAgeLogMelSpec`)
- `OUTPUT_DIR`: Ausgabeordner für das trainierte Modell (Standard: `FrisperWhisper/largeV3.2`)
- `MODEL_SIZE`: Whisper Modellgröße (Standard: `large-v3`)
- `NUM_GPUS`: Anzahl GPUs (Standard: `4`)
- `NUM_CPUS`: Anzahl CPUs für SLURM (Standard: `24`)
- `DATALOADER_WORKERS`: Anzahl Dataloader Workers (Standard: `20`)
- `TRAIN_BATCH_SIZE`: Training Batch-Größe pro GPU (Standard: `1`)
- `EVAL_BATCH_SIZE`: Evaluation Batch-Größe pro GPU (Standard: `1`)
- `GRADIENT_ACCUMULATION`: Gradient Accumulation Steps (Standard: `16`)
- `LEARNING_RATE`: Lernrate (Standard: `1.5e-5`)
- `MAX_STEPS`: Maximale Trainingsschritte (Standard: `10000`)
- `WARMUP_STEPS`: Warmup-Schritte (Standard: `1000`)
- `SAVE_STEPS`: Wie oft Checkpoints gespeichert werden (Standard: `500`)
- `EVAL_STEPS`: Wie oft evaluiert wird (Standard: `500`)
- `LOGGING_STEPS`: Wie oft geloggt wird (Standard: `50`)
- `WEIGHT_DECAY`: Weight Decay für Regularisierung (Standard: `0.05`)
- `LR_SCHEDULER_TYPE`: Learning Rate Scheduler Typ (Standard: `linear`)
- `RESUME_CHECKPOINT`: Checkpoint zum Fortsetzen (leer = von vorne, `true` = letzter, `checkpoint-XXXX` = spezifisch)

---

## 4. Daten Vergleichen

Systematischer Vergleich zwischen Whisper Large V3 und fine-tuned Modell:

```bash
(.venv)$ python scripts/run_whisper_comparison.py \
    --input data/LangAge16kHz \
    --output results/comparison_v1 \
    --fine_tuned_model FrisperWhisper/largeV1 \
    --checkpoint checkpoint-2000 \
    --dataset_path data/ESLOLangAgeDataSet \
    --cpus 32 \
    --gpus 4 \
    --batch_size 16 \
    --transcription_batch_processes 8 \
    --steps all
```

**SLURM-Batch-Skripte:**

```bash
# LangAge-Daten
(.venv)$ sbatch scripts/run_whisper_comparison.sbatch
```

Das Skript erstellt eine umfassende CSV mit:
- **Metriken**: WER, CER, BLEU-Scores
- **Train/Test-Spalten**: Zeigt an, ob Segment im Training/Test-Set war
- **Marker-Spalten**: Interjektionen und spezielle Muster
- **Metadaten**: Sprecher, Zeitstempel, demographische Informationen

---

**Mit Train/Test-Spalten:**

Um die CSV mit Spalten zu erweitern, die anzeigen, ob ein Segment im Training oder Test-Set war, kann der Parameter `--dataset_path` verwendet werden:

```bash
(.venv)$ python scripts/run_whisper_comparison.py \
    --input data/ESLOLangAgeCombined16kHz \
    --output results/ESLOLangAgev2 \
    --fine_tuned_model FrisperWhisper/largeV3_ESLOLangAge_V2 \
    --checkpoint checkpoint-4000 \
    --dataset_path data/ESLOLangAgeDataSet \
    --cpus 32 \
    --gpus 4 \
    --batch_size 16 \
    --transcription_batch_processes 8 \
    --steps all
```

### Parameter

- `--input`: Eingabeverzeichnis mit TextGrid- und Audio-Dateien (erforderlich)
- `--output`: Ausgabeverzeichnis für Ergebnisse (erforderlich)
- `--fine_tuned_model`: Pfad zum fine-tuned Whisper-Modell (erforderlich)
- `--checkpoint`: Spezifischer Checkpoint (z.B. `checkpoint-2000`). Falls nicht angegeben, wird das finale Modell verwendet
- `--dataset_path`: Pfad zum HuggingFace Dataset (optional). Fügt `train` und `test` Spalten zur CSV hinzu
- `--cpus`: Anzahl CPU-Kerne (Standard: 8)
- `--gpus`: Anzahl GPUs (Standard: 1). Multi-GPU wird unterstützt
- `--batch_size`: Batch-Größe für Transkription (Standard: 32)
- `--transcription_batch_processes`: Anzahl paralleler Batch-Prozesse (Standard: 4)
- `--steps`: Pipeline-Schritte (`all`, `metadata`, `transcription`, `metrics`; Standard: `all`)
- `--file_limit`: Dateien-Limit für Tests (optional)
- `--resume_from_transcriptions`: Fortsetzen von existierenden Transkriptionen (optional)

### Output-Dateien

- **`whisper_comparison_results.csv`**: Hauptdatei mit allen Ergebnissen
- **`whisper_comparison_results_sample.csv`**: Erste 100 Zeilen zur schnellen Inspektion
- **`README.md`**: Dokumentation der Ausführung mit Parametern
- **`whisper_comparison_results_intermediate/`**: Zwischendateien (JSON)
  - `segments_with_metadata.json`
  - `segments_with_transcriptions.json`

### CSV-Spalten

Die erzeugte CSV enthält folgende Informationen:

**Metadaten:**
- `filename`, `speaker_id`, `interview_number`
- `startTime`, `endTime`, `interval`
- `gender`, `dialect`, `segment_duration`
- `train`, `test`: Binäre Indikatoren (1 = Segment war im jeweiligen Set, 0 = nicht)

**Transkripte:**
- `transcript_original`: Original-Transkript aus TextGrid
- `transcript_fine_tuned`: Transkript vom fine-tuned Modell

**Metriken:**
- `WER_*`: Word Error Rate (niedriger ist besser)
- `CER_*`: Character Error Rate (niedriger ist besser)
- `BLEU_*`: BLEU Score (höher ist besser)

**Marker-Spalten:** Binäre Indikatoren für Interjektionen (`ah`, `euh`, etc.) und spezielle Muster (`(buzz)`, `XXX`, etc.)

---

## 5. Sonstiges

### 5.1 Daten-Anomalien erkennen

Das folgende Skript identifiziert problematische Intervalle in TextGrid-Dateien:

```bash
(.venv)$ python scripts/extract_empty_intervals_and_overlaps.py -f input_folder -o output_folder
```

**Erkannte Anomalien:**
- Intervalle mit 0 ms Länge
- Überlappende Zeitstempel
- Inkonsistente Segmentierung

**Output:** Zwei CSV-Dateien mit detaillierter Anomalie-Liste für Datenbereinigung.
