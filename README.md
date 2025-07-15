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

### Downsampling 44,1 kHz zu 16 kHz

Zunächst werden die Original-Audioaufnahmen komprimiert, und zwar von 44,4 kHz Sampling-Rate auf 16 kHz Samplin-Rate. Dies kann einiige Zeit dauern. Für den folgenden Befehl gibt es drei Optionen:

- `--input_folder`: Spezifiziert den zu verarbeitenden Ordner (sollte `./data` sein).
- `--option` (default=`keep`, alternativ `delete`): Mit dieser Option können die großen 44,1 kHz-Dateien nach dem downsampling gelöscht werden (`delete`) - das spart Speicherplatz, sollte aber nur ausgeführt werden, wenn es sich bei dem verarbeiteten Ordner um eine Kopie der Originalaufnahmen handelt.
- `--processes`: Spezifiziert wie viele Aufnahmen gleichzeitig verarbeitet werden sollen; es können maximal so viele Prozesse ausgewählt werden, wie Cores zur Verfügung stehen. In der Praxis empfiehlt es sich, nicht mehr als die Hälfte der verfügbaren Cores zu verwenden.

```bash
(.venv)$ python scripts/resample44k16k.py -i data -o keep -p 4
```



## Sonstiges

### Daten-'Anomalien' erkennen

Das folgende Skript nimmt einen Ordner mit TextGrids als Input und schreibt in einen spezifizierten Outputfolder zwei CSV-Dateien. Diese Dateien listen Intervalle auf, die eine Länge von 0 ms haben sowie Intervalle, die sich überlappen.

```bash
(.venv)$ python extract_empty_intervals_and_overlaps.py -f input_folder -o output_folder
```