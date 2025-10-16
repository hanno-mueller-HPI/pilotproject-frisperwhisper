# KI-Transkripte als linguistische Forschungsdaten: Feintuning eines Modells für das gesprochene Französisch mit Integration prosodischer Daten

## Preambel

Dieses README beschreibt wie mithilfe des LangAge-Korpus ein ASR-Modell angepasst wird. Dafür werden die wav-Dateien des LangAge-Korpus verarbeitet (z.B. downsampling von 44,1 kHz zu 16 kHz). Es wird angenommen, dass die Original-Dateien gespeichert sind in `./data_original`.

## Anforderungen

Um sicherzustellen, dass über verschiedene Betriebssysteme und Maschinen hinweg die gleiche Anforderungen erfüllt sind, wird das Anlegen einer *virtual environment* empfohlen. Diese lässt sich mit UV einrichten.

```bash
$ which uv || echo "UV not found" # überprüft die UV Installation
```

Sollte UV nicht installiert sein, lässt es sich wie folgt installieren.

```bash
$ curl -LsSf https://astral.sh/uv/install.sh | sh
```

Anschließend kann die virtuelle Umgebung erstellt und aktiviert werden.

```bash
$ uv venv .venv # erstellt eine virtual environment mit dem Namen ".venv"
$ source .venv/bin/activate # aktiviert die virtual environment
```

Dann werden die benötigten Pakete installiert. UV sorgt dafür, dass die exakten Versionen installiert werden.

```bash
(.venv)$ uv sync --active  # installiert exakte Versionen
```

## Fine-Tuning Whisper

### Workflow-Übersicht

Der Datenverarbeitungsprozess erfolgt in den folgenden Schritten:

1. **TRS zu TextGrid**: `trs2tg.py` transformiert Transkripte zu TextGrids.
2. **Audio-Resampling**: `resample44k16k.py` konvertiert alle WAV-Dateien von 44,1 kHz auf 16 kHz
3. **Dataset Dictionary**: `Textgrid2DatasetBatch.py` verarbeitet TextGrid-Dateien und erstellt Train/Test-Splits mit einer Multiprocessing-Architektur. Die erstellten Splits werden in einem HuggingFace Dataset Dictionary gespeichert. Dabei wird das `modules/clean_TextGrids.py` Modul verwendet, um bestimmte Segmente herauszufiltern:
   - Segmente mit einer Dauer unter 100 ms
   - Segmente von spk1 (Interviewer)
   - Segmente mit Klammer-Inhalten `( )`, `[ ]`, `< >`
   - Segmente mit (buzz)-Mustern
   - Segmente mit XXX-Mustern (zwei oder mehr aufeinanderfolgende X-Zeichen)
   - Leere oder sehr kurze Audio-Segmente
4. **Log-Mel Spektrogramme**: `DataSet2LogMelSpecBatch.py` transfomiert im Dataset Dictionary hinterlegte Audios zu Log-Mel Spektrogrammen. Diese werden für das Training/Fine-Tuning von Whisper verwendet.
5. **Fine-Tuning**: `finetune_whisper_from_LogMel.py`  verwendet die verarbeiteten Transkripte und Audios, um Whisper zu trainieren.

### Sicherheitskopie

Um sicherzustellen, dass die Original-Dateien nicht verändert werden, wird eine Kopie der Original-Dateien erstellt:

```bash
(.venv)$ mkdir data
(.venv)$ cp data_original data/LangAge16kHz # audios werden noch von 44,1 kHz zu 16 kHz resampled (s. unten)
```

### TRS zu TextGrid

Transkripte werden mit dem Skript `trs2tg.py` zu TextGrids transformiert.

```bash
(.venv)$ python scripts/trs2tg.py data/LangAge16kHz
```

### Downsampling 44,1 kHz zu 16 kHz

Als nächstes werden die Original-Audioaufnahmen komprimiert, und zwar von 44,1 kHz Sampling-Rate auf 16 kHz Sampling-Rate. Dies kann einige Zeit (d.h. Tage) dauern. Das Skript überschreibt die ursprünglichen Dateien mit den resampelten Versionen. Für den folgenden Befehl gibt es zwei Optionen:

- `--input_folder`: Spezifiziert den zu verarbeitenden Ordner.
- `--processes`: Spezifiziert wie viele Aufnahmen gleichzeitig verarbeitet werden sollen; es können maximal so viele Prozesse ausgewählt werden, wie Cores zur Verfügung stehen. In der Praxis empfiehlt es sich, nicht mehr als die Hälfte der verfügbaren Cores zu verwenden.

```bash
(.venv)$ python scripts/resample44k16k.py -i data/LangAge16kHz -p 40
```

### Dataset Dictionary

Die Textgrids und Audios (16 kHz) werden als DataSet Dictionary (DataSetDict) gespeichert. Das DataSetDict besteht aus zwei Datasets, eines für Training (z.B. 80%) und eines zum Testen (z.B. 20%). Das Test-Set besteht, unter anderem, aus den händisch-kurierten Intervallen, die in einer CSV-Datei spezifiziert werden können. 

Das Skript `Textgrid2DatasetBatch.py` nutzt eine Multiprocessing-Architektur mit robusten Audio-Handling für große Dateien und für Memory-Effizienz. Zusätzlich werden die Daten automatisch via `modules/clean_TextGrids.py` bereinigt:

**Automatische Datenbereinigung:**
- Entfernung zu kurzer Segmente (< 100ms)
- Filterung stiller Audio-Segmente (nur Nullen)
- Entfernung leerer Transkripte
- Filterung spezifischer Sprecher (z.B. "spk1")
- Entfernung von "(buzz)" Mustern (anonymisierte Passagen)
- Entfernung von Segmenten mit unverständlichen Äußerungen anhand der [] Muster
- Entfernung der Interviews h015a, e025a, d048a, die gesondert evaluiert werden sollen

Folgende Optionen stehen zur Verfügung:

- `-f`, `--folder`: Pfad zum Ordner mit TextGrid-Dateien (erforderlich)
- `-o`, `--output_folder`: Pfad für das finale DataSetDict (erforderlich)  
- `-n`, `--number_of_processes`: Anzahl paralleler Prozesse für TextGrid-Verarbeitung (Standard: 4)
- `-c`, `--csv_file`: CSV-Datei mit Test-Set Intervallen (optional, siehe unten)
- `--batch_size`: Batch-Größe für die Verarbeitung von Einträgen (Standard: 500)
- `--audio_batch_processes`: Anzahl der Prozesse für Audio-Batch-Verarbeitung (Standard: 2)

```bash
(.venv)$ python scripts/Textgrids2DatasetBatch.py -f data/LangAge16kHz -o data -n 150 --batch_size 500 --audio_batch_processes 8
```

#### CSV-Format für Test-Set Definition

Falls eine CSV-Datei für die Test-Set Definition verwendet wird, muss sie folgende Spalten enthalten:
- `path`: Pfad zur TextGrid-Datei
- `speaker`: Sprecher-ID
- `interval`: Intervall-Index (beginnend bei 0)

Beispiel (`test_intervals.csv`):
```csv
path,speaker,interval
data/a001.TextGrid,speaker1,5
data/a001.TextGrid.TextGrid,speaker1,12
data/a001.TextGrid.TextGrid,speaker2,3
```

#### Train/Test-Split Logik

Das Skript teilt die Daten folgendermaßen auf:

1. **Test-Set**: Falls eine CSV-Datei angegeben wird, werden die dort spezifizierten Intervalle garantiert ins Test-Set aufgenommen
2. **Verbleibendes Test-Set**: Die restlichen 20% werden zufällig aus den verbleibenden Daten ausgewählt
3. **Train-Set**: Alle übrigen Daten (ca. 80%) werden für das Training verwendet

Diese Methode stellt sicher, dass wichtige oder händisch-kuratierte Intervalle im Test-Set landen, während gleichzeitig eine ausgewogene Aufteilung gewährleistet wird.

### Log-Mel Spektrogramme

Whisper wird mit Audios in Form von Log-Mel Spektrogrammen 'gefüttert'. Diese Spektrogramme werden mit `DataSet2LogMelSpecBatch.py` erstellt.

Folgende Optionen stehen zur Verfügung:

- `-i`, `--input_dataset`: Pfad zum Eingabeordner mit dem LangAgeDataSet (HuggingFace Dataset, erforderlich)
- `-o`, `--output_dataset`: Pfad, unter dem das vorverarbeitete Dataset gespeichert wird (erforderlich)
- `--num_cpus`: Anzahl der zu verwendenden CPU-Kerne für die Vorverarbeitung. Wenn nicht angegeben, werden alle verfügbaren Kerne genutzt.
- `--model_size`: Whisper-Modellgröße für die Feature-Extraktion (`tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3`; Standard: `large`). Modell 'large V3' benutzt 128 mel bins im Gegensatz zu den anderen Whisper Modellen, die 80 mel bins benutzen.
- `--shuffle_seed`: Zufalls-Seed für das Mischen der Datasets (Standard: 42)
- `--max_samples`: Maximale Anzahl zu verarbeitender Samples pro Split (für Tests). Wenn nicht angegeben, werden alle Samples verarbeitet.
- `--batch_size`: Batch-Größe für die Verarbeitung von Dataset-Chunks (Standard: 1000)
- `--writer_batch_size`: Batch-Größe für das Speichern auf die Festplatte (Standard: 100)
- `--max_memory_per_worker`: Maximaler Speicher pro Worker in GB (Standard: 4.0)

```bash
(.venv)$ python scripts/Dataset2LogMelSpecBatch.py -i data/LangAgeDataSet -o data/LangAgeLogMelSpec --model_size large-v3 --num_cpus 150 --batch_size 1000
```


### Grundlegendes Training

Das `finetune_whisper.py` Skript bietet umfassende Funktionen für das Training von Whisper-Modellen mit GPU-Optimierung und Checkpoint-Unterstützung.

```bash
(.venv)$ python scripts/finetune_whisper_from_LogMel.py \
    --dataset_path data/LangAgeLogMelSpec \
    --output_dir FrisperWhisper \
    --version v1 \
    --model_size large-v3 \
    --num_gpus 4 \
    --num_cpus 40 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16
```

#### Training von Checkpoints fortsetzen

Das Skript unterstützt das Fortsetzen unterbrochener Trainings von gespeicherten Checkpoints:

##### Automatisches Fortsetzen vom letzten Checkpoint

```bash
(.venv)$ python scripts/finetune_whisper_from_LogMel.py \
    --input_dataset data/LangAgeLogMelSpec \
    --output_dir FrisperWhisper \
    --version v1 \
    --resume_from_checkpoint true
```

##### Fortsetzen von einem spezifischen Checkpoint

```bash
(.venv)$ python scripts/finetune_whisper_from_LogMel.py \
    --input_dataset data/LangAgeLogMelSpec \
    --output_dir FrisperWhisper \
    --version v1 \
    --resume_from_checkpoint checkpoint-1000
```

##### Fortsetzen von absolutem Pfad

```bash
(.venv)$ scripts/finetune_whisper_from_LogMel.py \
    --input_dataset data/LangAgeLogMelSpec \
    --output_dir FrisperWhisper \
    --version v1 \
    --resume_from_checkpoint /full/path/to/checkpoint-1000
```

**Checkpoint-Verhalten:**
- Checkpoints werden automatisch alle `--save_steps` (Standard: 1000) Schritte gespeichert
- Jeder Checkpoint enthält Modellgewichte, Optimizer-Status, Scheduler-Status und Trainingsfortschritt
- Das Training wird exakt an der Stelle fortgesetzt, an der es unterbrochen wurde
- Falls der angegebene Checkpoint nicht existiert, startet das Training von vorne mit einer Warnung


## Transkribieren und Modellvergleich mit Whisper

Das Skript `run_whisper_comparison.py` ermöglicht einen systematischen Vergleich zwischen dem ursprünglichen Whisper Large V3 Modell und einem fine-tuned Modell. Es verarbeitet Audio-Segmente aus TextGrid-Dateien und berechnet verschiedene Metriken (WER, CER, BLEU) für die Bewertung der Transkriptionsqualität.

### Pipeline-Schritte

Das Skript führt folgende Schritte aus:

1. **Metadaten-Extraktion**: Extrahiert Audio-Segmente und Sprecher-Informationen aus TextGrid-Dateien
2. **Transkription**: Transkribiert mit beiden Modellen (Whisper Large V3 und Fine-tuned)
3. **Metrik-Berechnung**: Berechnet WER, CER und BLEU-Scores für beide Modelle
4. **CSV-Export**: Erstellt eine umfassende CSV-Datei mit allen Vergleichen

### Verwendung

**Basis-Kommando:**

```bash
(.venv)$ python scripts/run_whisper_comparison.py \
    --input data/LangAge16kHz \
    --output results/comparison_v1 \
    --fine_tuned_model FrisperWhisper/largeV3 \
    --checkpoint checkpoint-6000 \
    --cpus 32 \
    --gpus 4 \
    --batch_size 16 \
    --transcription_batch_processes 8 \
    --steps all
```

**Mit Train/Test-Spalten:**

Um die CSV mit Spalten zu erweitern, die anzeigen, ob ein Segment im Training oder Test-Set war, kann der Parameter `--dataset_path` verwendet werden:

```bash
(.venv)$ python scripts/run_whisper_comparison.py \
    --input data/LangAge16kHz \
    --output results/comparison_v1 \
    --fine_tuned_model FrisperWhisper/largeV3 \
    --checkpoint checkpoint-6000 \
    --dataset_path data/LangAgeDataSet \
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
- `--checkpoint`: Spezifischer Checkpoint (z.B. `checkpoint-6000`). Falls nicht angegeben, wird das finale Modell verwendet
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
- `transcript_large_v3`: Transkript von Whisper Large V3
- `transcript_fine_tuned`: Transkript vom fine-tuned Modell

**Metriken:**
- `WER_*`: Word Error Rate (niedriger ist besser)
- `CER_*`: Character Error Rate (niedriger ist besser)
- `BLEU_*`: BLEU Score (höher ist besser)

**Marker-Spalten:** Binäre Indikatoren für Interjektionen (`ah`, `euh`, etc.) und spezielle Muster (`(buzz)`, `XXX`, etc.)

### SLURM-Batch-Skripte

Für die Verarbeitung auf dem Cluster stehen SLURM-Batch-Skripte zur Verfügung:

#### LangAge-Daten vergleichen

```bash
(.venv)$ sbatch scripts/run_whisper_comparison.sbatch
```

Das Skript kann durch Bearbeiten der Variablen angepasst werden:
- `INPUT_DIR`: Eingabeverzeichnis
- `FINE_TUNED_MODEL`: Pfad zum Modell
- `CHECKPOINT`: Zu verwendender Checkpoint
- `OUTPUT_DIR`: Ausgabeverzeichnis (wird automatisch generiert)
- `NUM_CPUS`, `NUM_GPUS`: Ressourcen-Konfiguration
- `BATCH_SIZE`, `TRANSCRIPTION_PROCESSES`: Verarbeitungs-Parameter

#### ESLO-Daten vergleichen

```bash
# Für ESLO 30-39 Jahre
(.venv)$ sbatch scripts/run_whisper_comparison_eslo.sbatch data/sampleESLO30-39 results/ESLO30-39

# Für ESLO 65+ Jahre
(.venv)$ sbatch scripts/run_whisper_comparison_eslo.sbatch data/sampleESLO65plus results/ESLO65plus

# Für kombinierte ESLO-Daten
(.venv)$ sbatch scripts/run_whisper_comparison_eslo.sbatch data/ESLOcombined results/ESLOcombined
```

Das ESLO-Skript akzeptiert zwei optionale Kommandozeilen-Argumente:
1. Eingabeverzeichnis (Standard: `data/sampleESLO30-39`)
2. Ausgabeverzeichnis (Standard: `results/ESLO30-39`)



## SLURM

Für die Verarbeitung auf dem Cluster stehen verschiedene SLURM-Skripte zur Verfügung:

### Dataset-Erstellung

Das `create_dataset.sbatch` Skript erstellt ein Dataset Dictionary aus TextGrid-Dateien und Audio-Dateien. Es können Umgebungsvariablen verwendet werden, um die Verarbeitung anzupassen.

#### Standard-Verarbeitung (LangAge)
```bash
(.venv)$ sbatch ./scripts/create_dataset.sbatch
```

#### Angepasste Verarbeitung mit Umgebungsvariablen

Für kombinierte LangAge und ESLO Daten:
```bash
(.venv)$ INPUT_FOLDER=data/LangAgeESLOcombined16kHz \
         OUTPUT_FOLDER=data/LangAgeESLODataSet \
         NUM_PROCESSES=150 \
         sbatch ./scripts/create_dataset.sbatch
```

Für ESLO-Daten:
```bash
(.venv)$ INPUT_FOLDER=data/ESLOcombined16kHz \
         OUTPUT_FOLDER=data/ESLODataSet \
         NUM_PROCESSES=150 \
         sbatch ./scripts/create_dataset.sbatch
```

Mit CSV-Datei für Test-Set Definition:
```bash
(.venv)$ INPUT_FOLDER=data/LangAge16kHz \
         OUTPUT_FOLDER=data/LangAgeDataSet \
         CSV_FILE=test_intervals.csv \
         sbatch ./scripts/create_dataset.sbatch
```

**Anpassbare Parameter über Umgebungsvariablen:**
- `INPUT_FOLDER`: Pfad zum Eingabeordner mit TextGrid- und Audio-Dateien (Standard: `data/LangAge16kHz`)
- `OUTPUT_FOLDER`: Pfad zum Ausgabeordner für das Dataset Dictionary (Standard: `data/LangAgeDataSet`)
- `NUM_PROCESSES`: Anzahl paralleler Prozesse für TextGrid-Verarbeitung (Standard: `120`)
- `BATCH_SIZE`: Batch-Größe für die Verarbeitung von Einträgen (Standard: `500`)
- `AUDIO_BATCH_PROCESSES`: Anzahl der Prozesse für Audio-Batch-Verarbeitung (Standard: `8`)
- `CSV_FILE`: CSV-Datei mit Test-Set Intervallen (optional, leer = nicht verwendet)

### DataSet zu Log-Mel Spektrogrammen transformieren

Das `run_dataset_preprocess_batch.sbatch` Skript transformiert ein Dataset Dictionary zu Log-Mel Spektrogrammen. Es können Umgebungsvariablen verwendet werden, um die Verarbeitung anzupassen.

#### Standard-Verarbeitung (LangAge)
```bash
(.venv)$ sbatch ./scripts/run_dataset_preprocess_batch.sbatch
```

#### Angepasste Verarbeitung mit Umgebungsvariablen

Für kombinierte LangAge und ESLO Daten:
```bash
(.venv)$ INPUT_DATASET=data/LangAgeESLODataSet \
         OUTPUT_DATASET=data/LangAgeESLOLogMelSpec \
         MODEL_SIZE=large-v3 \
         NUM_CPUS=150 \
         BATCH_SIZE=1000 \
         sbatch ./scripts/run_dataset_preprocess_batch.sbatch
```

Für ESLO-Daten:
```bash
(.venv)$ INPUT_DATASET=data/ESLODataSet \
         OUTPUT_DATASET=data/ESLOLogMelSpec \
         MODEL_SIZE=large-v3 \
         NUM_CPUS=150 \
         BATCH_SIZE=1000 \
         sbatch ./scripts/run_dataset_preprocess_batch.sbatch
```

Mit begrenzter Anzahl Samples (für Tests):
```bash
(.venv)$ INPUT_DATASET=data/LangAgeDataSet \
         OUTPUT_DATASET=data/LangAgeLogMelSpec_test \
         MAX_SAMPLES=1000 \
         sbatch ./scripts/run_dataset_preprocess_batch.sbatch
```

**Anpassbare Parameter über Umgebungsvariablen:**
- `INPUT_DATASET`: Pfad zum Eingabe-Dataset Dictionary (Standard: `data/LangAgeDataSet`)
- `OUTPUT_DATASET`: Pfad zum Ausgabe-Dataset mit Log-Mel Spektrogrammen (Standard: `data/LangAgeLogMelSpec`)
- `NUM_CPUS`: Anzahl der CPU-Kerne für die Verarbeitung (Standard: `100`)
- `BATCH_SIZE`: Batch-Größe für die Verarbeitung von Dataset-Chunks (Standard: `400`)
- `MODEL_SIZE`: Whisper-Modellgröße für Feature-Extraktion (`tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3`; Standard: `large-v3`)
- `WRITER_BATCH_SIZE`: Batch-Größe für das Speichern auf die Festplatte (Standard: `100`)
- `MAX_MEMORY_PER_WORKER`: Maximaler Speicher pro Worker in GB (Standard: `4.0`)
- `SHUFFLE_SEED`: Zufalls-Seed für das Mischen der Datasets (Standard: `42`)
- `MAX_SAMPLES`: Maximale Anzahl zu verarbeitender Samples pro Split (optional, für Tests)

### Whisper Training

Das `train_whisper.sbatch` Skript führt das Whisper-Training auf dem Cluster durch. Die Whisper-Modelle werden mit einer Versionsangabe gespeichert. Bei Verwendung von `-o FrisperWhisper -v v1` wird das Modell in `./FrisperWhisper/v1/` gespeichert.

#### Standard-Training
```bash
(.venv)$ sbatch ./scripts/train_whisper.sbatch
```

#### Angepasstes Training mit Umgebungsvariablen

Für kombinierte LangAge und ESLO Daten:
```bash
(.venv)$ DATASET_PATH=data/LangAgeESLOLogMelSpec \
         OUTPUT_DIR=FrisperWhisper/largeV3_LangAgeESLO \
         MAX_STEPS=15000 \
         sbatch ./scripts/train_whisper.sbatch
```

Für ESLO-Daten:
```bash
(.venv)$ DATASET_PATH=data/ESLOLogMelSpec \
         OUTPUT_DIR=FrisperWhisper/largeV3_ESLO \
         MAX_STEPS=10000 \
         sbatch ./scripts/train_whisper.sbatch
```

Mit angepassten Hyperparametern:
```bash
(.venv)$ DATASET_PATH=data/LangAgeLogMelSpec \
         OUTPUT_DIR=FrisperWhisper/largeV3_custom \
         LEARNING_RATE=2e-5 \
         MAX_STEPS=12000 \
         WARMUP_STEPS=1200 \
         sbatch ./scripts/train_whisper.sbatch
```

#### Training von Checkpoint fortsetzen
```bash
# Setze RESUME_CHECKPOINT Umgebungsvariable
(.venv)$ DATASET_PATH=data/LangAgeESLOLogMelSpec \
         OUTPUT_DIR=FrisperWhisper/largeV3_LangAgeESLO \
         RESUME_CHECKPOINT=checkpoint-1000 \
         sbatch ./scripts/train_whisper.sbatch

# Oder automatisch vom letzten Checkpoint
(.venv)$ DATASET_PATH=data/LangAgeESLOLogMelSpec \
         OUTPUT_DIR=FrisperWhisper/largeV3_LangAgeESLO \
         RESUME_CHECKPOINT=true \
         sbatch ./scripts/train_whisper.sbatch
```

**Anpassbare Parameter über Umgebungsvariablen:**
- `DATASET_PATH`: Pfad zum Dataset mit Log-Mel Spektrogrammen (Standard: `data/LangAgeLogMelSpec`)
- `OUTPUT_DIR`: Ausgabeordner für das trainierte Modell (Standard: `FrisperWhisper/largeV3.2`)
- `MODEL_SIZE`: Whisper Modellgröße (Standard: `large-v3`)
- `NUM_GPUS`: Anzahl GPUs (Standard: `4`)
- `NUM_CPUS`: Anzahl CPUs für SLURM (Standard: `24`)
- `DATALOADER_WORKERS`: Anzahl Dataloader Workers (Standard: `20`)
- `TRAIN_BATCH_SIZE`: Training Batch-Größe pro GPU (Standard: `1`)
- `EVAL_BATCH_SIZE`: Evaluation Batch-Größe pro GPU (Standard: `1`)
- `GRADIENT_ACCUMULATION`: Gradient Accumulation Steps (Standard: `16`, effektive Batch-Größe = 1×4×16 = 64)
- `LEARNING_RATE`: Lernrate (Standard: `1.5e-5`)
- `MAX_STEPS`: Maximale Trainingsschritte (Standard: `10000`)
- `WARMUP_STEPS`: Warmup-Schritte (Standard: `1000`)
- `SAVE_STEPS`: Wie oft Checkpoints gespeichert werden (Standard: `500`)
- `EVAL_STEPS`: Wie oft evaluiert wird (Standard: `500`)
- `LOGGING_STEPS`: Wie oft geloggt wird (Standard: `50`)
- `WEIGHT_DECAY`: Weight Decay für Regularisierung (Standard: `0.05`)
- `LR_SCHEDULER_TYPE`: Learning Rate Scheduler Typ (Standard: `linear`)
- `RESUME_CHECKPOINT`: Checkpoint zum Fortsetzen (leer = von vorne, `true` = letzter, `checkpoint-XXXX` = spezifisch)


## Training mit ESLO-Daten

### ESLO-Daten kombinieren

Um mit ESLO-Daten zu arbeiten, müssen zunächst die verschiedenen ESLO-Unterordner kombiniert werden. Dies geschieht mit symbolischen Links, um Speicherplatz zu sparen:

```bash
(.venv)$ mkdir -p data/ESLOcombined
(.venv)$ cd data/ESLOcombined
(.venv)$ ln -s ../sampleESLO30-39/* .
(.venv)$ ln -s ../sampleESLO65plus/* .
```

### Dataset Dictionary für ESLO erstellen

Nach der Kombination kann ein Dataset Dictionary für alle ESLO-Daten erstellt werden:

```bash
(.venv)$ python scripts/Textgrids2DatasetBatch.py \
    -f data/ESLOcombined \
    -o data/ESLODataSet \
    -n 150 \
    --batch_size 500 \
    --audio_batch_processes 8
```

### Log-Mel Spektrogramme für ESLO

Anschließend werden die Log-Mel Spektrogramme erstellt:

```bash
(.venv)$ python scripts/Dataset2LogMelSpecBatch.py \
    -i data/ESLODataSet \
    -o data/ESLOLogMelSpec \
    --model_size large-v3 \
    --num_cpus 150 \
    --batch_size 1000
```

### Kombinierte LangAge und ESLO Daten

Um ein Modell auf beiden Datensätzen zu trainieren, können LangAge und ESLO kombiniert werden:

```bash
(.venv)$ mkdir -p data/LangAgeESLOcombined
(.venv)$ cd data/LangAgeESLOcombined
(.venv)$ ln -s ../LangAge16kHz/* .
(.venv)$ ln -s ../sampleESLO30-39/* .
(.venv)$ ln -s ../sampleESLO65plus/* .
```

Danach kann das Dataset Dictionary für die kombinierten Daten erstellt werden:

```bash
(.venv)$ python scripts/Textgrids2DatasetBatch.py \
    -f data/LangAgeESLOcombined \
    -o data/LangAgeESLODataSet \
    -n 150 \
    --batch_size 500 \
    --audio_batch_processes 8
```

Und die Log-Mel Spektrogramme:

```bash
(.venv)$ python scripts/Dataset2LogMelSpecBatch.py \
    -i data/LangAgeESLODataSet \
    -o data/LangAgeESLOLogMelSpec \
    --model_size large-v3 \
    --num_cpus 150 \
    --batch_size 1000
```

Das Training erfolgt dann wie gewohnt mit dem kombinierten Dataset:

```bash
(.venv)$ python scripts/finetune_whisper_from_LogMel.py \
    --dataset_path data/LangAgeESLOLogMelSpec \
    --output_dir FrisperWhisper \
    --version v2-combined \
    --model_size large-v3 \
    --num_gpus 4 \
    --num_cpus 40
```


## Transkription mit fine-tuned Modellen

Das Skript `transcribe_with_finetuned.py` ermöglicht die Transkription von Audio-Dateien mit Whisper-Modellen (lokal oder von HuggingFace). Es erstellt eine CSV-Datei mit Zeitstempeln und Transkriptionen.

### Funktionen

- Unterstützung für **lokale Modelle** und **HuggingFace Hub Modelle**
- Verarbeitung **einzelner Audio-Dateien** oder **ganzer Verzeichnisse**
- Automatische Segmentierung bei langen Audios (mit `--use_pipeline`)
- CSV-Export mit ID, Start, Stop, Transkription (und optional Dateiname)

### Verwendung

**Einzelne Audio-Datei mit lokalem Modell:**

```bash
(.venv)$ python scripts/transcribe_with_finetuned.py \
    -i data/LangAge16kHz/a001a.wav \
    -m FrisperWhisper/largeV3.2/checkpoint-2000 \
    -o transcription_result.csv \
    --language french
```

**Verzeichnis mit HuggingFace Modell:**

```bash
(.venv)$ python scripts/transcribe_with_finetuned.py \
    -i data/sampleESLO30-39 \
    -m openai/whisper-large-v3 \
    -o eslo_transcriptions.csv \
    --language french
```

**Mit automatischer Segmentierung (für lange Audios):**

```bash
(.venv)$ python scripts/transcribe_with_finetuned.py \
    -i audio_files/ \
    -m openai/whisper-large-v3 \
    -o results.csv \
    --use_pipeline \
    --device cuda
```

### Parameter

- `-i, --input`: Pfad zu Audio-Datei oder Verzeichnis (erforderlich)
- `-m, --model`: Modell-Pfad (lokal) oder HuggingFace Model-ID (z.B. `openai/whisper-large-v3`) (erforderlich)
- `-o, --output`: Pfad zur Ausgabe-CSV-Datei (erforderlich)
- `--language`: Sprache für Transkription (Standard: `french`)
- `--device`: Gerät (`cpu`, `cuda`, oder `auto`; Standard: `auto`)
- `--use_pipeline`: HuggingFace Pipeline verwenden (automatische Segmentierung für lange Audios)

### CSV-Format

**Einzelne Datei:**
```csv
ID,Start,Stop,Transcription
1,00:00.000,00:12.500,"Bonjour, comment allez-vous?"
2,00:12.500,00:25.320,"Je vais très bien, merci."
```

**Mehrere Dateien (Verzeichnis):**
```csv
ID,Filename,Start,Stop,Transcription
1,audio1.wav,00:00.000,00:12.500,"Bonjour, comment allez-vous?"
2,audio1.wav,00:12.500,00:25.320,"Je vais très bien, merci."
3,audio2.wav,00:00.000,00:08.100,"C'est magnifique."
```

**Zeitformat:** `MM:SS.mmm` (Minuten:Sekunden.Millisekunden)

### Unterstützte Audio-Formate

`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.opus`


### TODOs






## Sonstiges

### Daten-'Anomalien' erkennen

Das folgende Skript nimmt einen Ordner mit TextGrids als Input und schreibt in einen spezifizierten Outputfolder zwei CSV-Dateien. Diese Dateien listen Intervalle auf, die eine Länge von 0 ms haben sowie Intervalle, die sich überlappen.

```bash
(.venv)$ python scripts/extract_empty_intervals_and_overlaps.py -f input_folder -o output_folder
```