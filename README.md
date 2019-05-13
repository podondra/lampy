# Detekce anomálií v datech o znečištění ovzduší

Projekt detekce anomálií v datech o znečištění ovzduší ze sensorické sítě
veřejného osvětlení v okolí Karlínského náměstí na Praze 8.

## Výstupy

Výstupem projektu jsou následující body:

1. Jupyter notebook s explorací a analýzou dat,
2. [report] shrnující výsledky projektu a
3. online skript pro detekci anomálií.

## Data

Tento projekt využívá otevřená data hlavního města Prahy.
Pro více informací viz soubor `data/README.md`.

## Online monitoring

Online monitoring je implementován ve skriptu `online_detection.py`,
který se spouší po vytvoření virtuálního prostředí, například:

	$ python3 -m venv venv
	$ source venv/bin/activate
	$ pip install -U pip wheel setuptools
	$ pip install -r requirements.txt

Následně lze zobrazit nápověda:

	$ python online_detection.py --help

Monitoring se zapíní pomocí například:

	$ python online_monitoring.py nec4q4darktyq3dq6izwwhfyyhum4r44

## Reference

1. Dominique T. Shipmon, Json M. Gurevithc, Paolo M. Piselli and Steve Edwards. *Time Series Anomaly Detection*. Dostupné [online][time_series_anomaly_detection].
2. Ian Goodfellow, Yoshua Bengio and Aaron Courville. *Deep learning*. Dostupné [online][deep_learning].

[deep_learning]: https://www.deeplearningbook.org
[time_series_anomaly_detection]: https://static.googleusercontent.com/media/research.google.com/cs//pubs/archive/dfd834facc9460163438b94d53b36f51bb5ea952.pdf
[report]: https://podondra.github.io/lampy/
