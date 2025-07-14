# KI-Transkripte als linguistische Forschungsdaten: Feintuning eines Modells für das gesprochene Französisch mit Integration prosodischer Daten


## Anforderungen

Um sicherzustellen, dass über verschiedene Betriebssysteme und Maschinen hinweg die gleiche Anforderungen erfüllt sind, wird das Anlegen einer *virtual environment* empfohlen. Diese lässt sich mit UV einrichten.

```bash
which uv || echo "UV not found" # überprüft die UV Installation
```

Sollte UV nicht installiert sein, lässt es sich wie folgt installieren.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Anschließend kann die Virtuelle Umgebung erstellt und aktiviert werden.

```bash
uv venv .venv # erstellt eine virtual environment mit dem Namen ".venv"
source .venv/bin/activate # aktiviert die virtual environment
```

Dann werden die benötigten Pakete installiert. UV sorgt dafür, dass die exaktern Versionen installiert werden.

```bash
uv pip sync uv.lock  # installiert exakte Versionen
```


## Datenverarbeitung





## TO-DOs

- automatic script requirement installation