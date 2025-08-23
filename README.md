# KI-Transkripte als linguistische Forschungsdaten: Feintuning eines Modells für das gesprochene Französisch mit Integration prosodischer Daten

## Preambel

Dieses README beschreibt wie mithilfe des LangAge-Korpus ein ASR-Modell angepasst wird. Dafür werden die wav-Dateien des LangAge-Korpus verarbeitet (z.B. downsampling von 44,1 kHz zu 16 kHz). Um sicherzustellen, dass die Originaldateien nicht verändert werden, empfielt es sich, die Originaldateien in den Ordner `./data` zu kopieren.

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

## Datenverarbeitung

### Workflow-Übersicht

Der Datenverarbeitungsprozess erfolgt in den folgenden Schritten:

1. **TRS zu TextGrid**: `trs2tg.py` transformiert Transkripte zu TextGrids.
2. **Audio-Resampling**: `resample44k16k.py` konvertiert alle WAV-Dateien von 44,1 kHz auf 16 kHz
3. **Dataset Dictionary**: `Textgrid2DatasetBatch.py` verarbeitet TextGrid-Dateien und erstellt Train/Test-Splits mit einer Multiprocessing-Architektur. Die erstellten Splits werden in einem HuggingFace Dataset Dictionary gespeichert. Dabei wird das `clean_TextGrids.py` Modul verwendet, um bestimmte Segmente herauszufiltern (z.B. Segmente mit einer Dauer unter 100 ms)
4. **Log-Mel Spektrogramme**: Im Dataset Dictionary hinterlegte Audios werden zu Log-Mel Spektrogrammen transformiert.

### TRS zu TextGrid

Transkripte werden mit dem Skript `trs2tg.py` zu TextGrids transformiert. Dafür müssen die Transkripte in den Ordner `data/LangAge16kHz` kopiert werden, wo die TextGrids zusammen mit den 16 kHz-Versionen der Audios gespeichert werden.

```bash
(.venv)$ python scripts/trs2tg.py data/LangAge16kHz
```

### Downsampling 44,1 kHz zu 16 kHz

Zunächst werden die Original-Audioaufnahmen komprimiert, und zwar von 44,1 kHz Sampling-Rate auf 16 kHz Sampling-Rate. Dies kann einige Zeit (d.h. Tage) dauern. Das Skript überschreibt die ursprünglichen Dateien mit den resampelten Versionen. Für den folgenden Befehl gibt es zwei Optionen:

- `--input_folder`: Spezifiziert den zu verarbeitenden Ordner (sollte `./data/LangAge16kHz` sein; original Dateien müssen vorher hierher kopiert werden).
- `--processes`: Spezifiziert wie viele Aufnahmen gleichzeitig verarbeitet werden sollen; es können maximal so viele Prozesse ausgewählt werden, wie Cores zur Verfügung stehen. In der Praxis empfiehlt es sich, nicht mehr als die Hälfte der verfügbaren Cores zu verwenden.

```bash
(.venv)$ python scripts/resample44k16k.py -i data/LangAge16kHz -p 40
```

### Dataset Dictionary

Die Textgrids und Audios (16 kHz) werden als DataSet Dictionary (DataSetDict) gespeichert. Das DataSetDict besteht aus zwei Datasets, eines für Training (z.B. 80%) und eines zum Testen (z.B. 20%). Das Test-Set besteht, unter anderem, aus den händisch-kurierten Intervallen, die in einer CSV-Datei spezifiziert werden können. 

Das Skript `Textgrid2DatasetBatch.py` nutzt eine Multiprocessing-Architektur mit robusten Audio-Handling für große Dateien und für Memory-Effizienz. Zusätzlich werden die Daten automatisch via `clean_TextGrids.py` bereinigt:

**Automatische Datenbereinigung:**
- Entfernung zu kurzer Segmente (< 100ms)
- Filterung stiller Audio-Segmente (nur Nullen)
- Entfernung leerer Transkripte
- Filterung spezifischer Sprecher (z.B. "spk1")
- Entfernung von "(buzz) anon (buzz)" Mustern

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
- `--model_size`: Whisper-Modellgröße für die Feature-Extraktion (`tiny`, `base`, `small`, `medium`, `large`; Standard: `large`)
- `--shuffle_seed`: Zufalls-Seed für das Mischen der Datasets (Standard: 42)
- `--max_samples`: Maximale Anzahl zu verarbeitender Samples pro Split (für Tests). Wenn nicht angegeben, werden alle Samples verarbeitet.
- `--batch_size`: Batch-Größe für die Verarbeitung von Dataset-Chunks (Standard: 1000)
- `--writer_batch_size`: Batch-Größe für das Speichern auf die Festplatte (Standard: 100)
- `--max_memory_per_worker`: Maximaler Speicher pro Worker in GB (Standard: 4.0)

```bash
(.venv)$ python scripts/Dataset2LogMelSpecBatch.py -i data/LangAgeDataSet -o data/LangAgeLogMelSpec --num_cpus 150 --batch_size 1000
```

## Fine-tuning von Whisper

Whisper mit mehreren GPUs fine-tunen.

```bash
(.venv)$ python scripts/finetune_whisper_gpu.py -i data/LangAgeLogMelSpec -o ./FrisperWhisper -v v1 --num_gpus 2 --num_cpus 30 --model_size large --dataloader_workers 8 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 16
```

## SLURM

Für die Verarbeitung auf dem Cluster stehen verschiedene SLURM-Skripte zur Verfügung:

### Dataset-Erstellung
```bash
(.venv)$ sbatch ./scripts/create_dataset.sbatch
```

### DataSet zu Log-Mel Spektrogrammen transformieren
```bash
(.venv)$ sbatch ./scripts/run_dataset_preprocess_batch.sbatch
```

### Whisper Training

Die Whisper-Modelle werden mit einer Versionsangabe gespeichert. Bei Verwendung von `-o FrisperWhisper -v v1` wird das Modell in `./FrisperWhisper/v1/` gespeichert.

#### GPU Training (Anpassbar)
```bash
(.venv)$ sbatch ./scripts/train_whisper_gpu.sbatch
```

**Hinweis:** Das Skript `train_whisper_gpu.sbatch` kann angepasst werden für verschiedene Hardware-Konfigurationen (GPUs, CPUs, Speicher, Laufzeit). Für automatische Konfiguration siehe Wrapper-Skript (TODO).

### TODOs
- [ ] Wrapper-Skript für automatische train_whisper_gpu.sbatch Konfiguration (GPUs, CPUs, Speicher, Laufzeit, Outputfolder, Version)
- [ ] Transcription-Skript für Inferenz mit fine-tuned Modellen (`transcribe_with_finetuned.py`)
- [ ] Integriertes Training-Skript für End-to-End Pipeline
- [ ] Automatisierte Evaluation nach dem Training
- [ ] Checkpoint-Recovery für unterbrochene Jobs






## Sonstiges

### Daten-'Anomalien' erkennen

Das folgende Skript nimmt einen Ordner mit TextGrids als Input und schreibt in einen spezifizierten Outputfolder zwei CSV-Dateien. Diese Dateien listen Intervalle auf, die eine Länge von 0 ms haben sowie Intervalle, die sich überlappen.

```bash
(.venv)$ python scripts/extract_empty_intervals_and_overlaps.py -f input_folder -o output_folder
```