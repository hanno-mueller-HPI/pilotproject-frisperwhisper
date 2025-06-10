# KI-Transkripte als linguistische Forschungsdaten: Feintuning eines Modells für das gesprochene Französisch mit Integration prosodischer Daten


## Anforderungen

Um sicherzustellen, dass über verschiedene Betriebssysteme und Maschinen hinweg die gleiche Anforderungen erfüllt sind, wird das Anlegen einer *virtual environment* empfohlen. Diese lässt sich mit Python einrichten.

```bash
python3 -m venv .venv # erstellt eine virtual environment mit dem Namen ".venv"
python3 source .venv/bin/activate # aktiviert die virtual environment
```

### Installation benötigter Pakete

```bash
pip install torch
pip install git+https://github.com/nyrahealth/transformers.git@crisper_whisper # install custom transformer for most accurate timestamps
pip install datasets
pip install transformers
pip install librosa
pip install soundfile
pip install accelerate
```


##



## TO-DOs

- automatic script requirement installation