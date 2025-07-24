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

Anschließend kann die Virtuelle Umgebung erstellt und aktiviert werden.

```bash
$ uv venv .venv # erstellt eine virtual environment mit dem Namen ".venv"
$ source .venv/bin/activate # aktiviert die virtual environment
```

Dann werden die benötigten Pakete installiert. UV sorgt dafür, dass die exaktern Versionen installiert werden.

```bash
$ uv pip install -r requirements.txt  # installiert exakte Versionen
```

## Datenverarbeitung

### Workflow-Übersicht

Der Datenverarbeitungsprozess erfolgt in zwei Hauptschritten:

1. **Audio-Resampling**: `resample44k16k.py` konvertiert alle WAV-Dateien von 44,1 kHz auf 16 kHz
2. **Dataset-Erstellung**: `Textgrid2DatasetBatch.py` (ersetzt das alte `TextGrids2Dataset.py`) verarbeitet TextGrid-Dateien und erstellt Train/Test-Splits mit optimierter Multiprocessing-Architektur

Das `clean_TextGrids.py` Modul wird automatisch von `Textgrid2DatasetBatch.py` verwendet, um störende Elemente zu filtern (z.B. "(buzz) anon (buzz)" Muster, leere Texte, zu kurze Audio-Segmente).

### Downsampling 44,1 kHz zu 16 kHz

Zunächst werden die Original-Audioaufnahmen komprimiert, und zwar von 44,1 kHz Sampling-Rate auf 16 kHz Sampling-Rate. Dies kann einige Zeit (d.h. Tage) dauern. Das Skript überschreibt die ursprünglichen Dateien mit den resampelten Versionen. Für den folgenden Befehl gibt es zwei Optionen:

- `--input_folder`: Spezifiziert den zu verarbeitenden Ordner (sollte `./data` sein).
- `--processes`: Spezifiziert wie viele Aufnahmen gleichzeitig verarbeitet werden sollen; es können maximal so viele Prozesse ausgewählt werden, wie Cores zur Verfügung stehen. In der Praxis empfiehlt es sich, nicht mehr als die Hälfte der verfügbaren Cores zu verwenden.

```bash
(.venv)$ python scripts/resample44k16k.py -i data -p 4
```

### Erstellung des DataSetDict

Die Textgrids und Audios (16 kHz) werden in ein DataSetDict gespeichert. Das DataSetDict besteht aus zwei Datasets, eines für Training (80%) und eines zum Testen (20%). Das Test-Set besteht, unter anderem, aus den händisch-kurierten Intervallen, die in einer CSV-Datei spezifiziert werden können. 

Das Skript `Textgrid2DatasetBatch.py` nutzt eine verbesserte Multiprocessing-Architektur mit robusten Audio-Handling für große Dateien und Parquet-Dateien als Zwischenspeicher für Memory-Effizienz. Zusätzlich werden die Daten automatisch bereinigt (z.B. Entfernung von "(buzz) anon (buzz)" Mustern und leeren Intervallen) über das `clean_TextGrids.py` Modul.

Folgende Optionen stehen zur Verfügung:

- `-f`, `--folder`: Pfad zum Ordner mit TextGrid-Dateien (erforderlich)
- `-o`, `--output_folder`: Pfad für das finale DataSetDict (erforderlich)  
- `-n`, `--number_of_processes`: Anzahl paralleler Prozesse für TextGrid-Verarbeitung (Standard: 4)
- `-c`, `--csv_file`: CSV-Datei mit Test-Set Intervallen (optional, siehe unten)
- `--batch_size`: Batch-Größe für die Verarbeitung von Einträgen (Standard: 500)
- `--audio_batch_processes`: Anzahl der Prozesse für Audio-Batch-Verarbeitung (Standard: 2)

```bash
(.venv)$ python scripts/Textgrid2DatasetBatch.py -f data -o output_dataset -n 120 --batch_size 500 --audio_batch_processes 8
```

#### CSV-Format für Test-Set Definition

Falls eine CSV-Datei für die Test-Set Definition verwendet wird, muss sie folgende Spalten enthalten:
- `path`: Pfad zur TextGrid-Datei
- `speaker`: Sprecher-ID
- `interval`: Intervall-Index (beginnend bei 0)

Beispiel (`test_intervals.csv`):
```csv
path,speaker,interval
data/speaker1/file1.TextGrid,speaker1,5
data/speaker1/file2.TextGrid,speaker1,12
data/speaker2/file3.TextGrid,speaker2,3
```

#### Train/Test-Split Logik

Das Skript teilt die Daten folgendermaßen auf:

1. **Test-Set**: Falls eine CSV-Datei angegeben wird, werden die dort spezifizierten Intervalle garantiert ins Test-Set aufgenommen
2. **Verbleibendes Test-Set**: Die restlichen 20% werden zufällig aus den verbleibenden Daten ausgewählt
3. **Train-Set**: Alle übrigen Daten (ca. 80%) werden für das Training verwendet

Diese Methode stellt sicher, dass wichtige oder händisch-kuratierte Intervalle im Test-Set landen, während gleichzeitig eine ausgewogene Aufteilung gewährleistet wird.


## Sonstiges

### Daten-'Anomalien' erkennen

Das folgende Skript nimmt einen Ordner mit TextGrids als Input und schreibt in einen spezifizierten Outputfolder zwei CSV-Dateien. Diese Dateien listen Intervalle auf, die eine Länge von 0 ms haben sowie Intervalle, die sich überlappen.

```bash
(.venv)$ python extract_empty_intervals_and_overlaps.py -f input_folder -o output_folder
```