# lampy

Projekt detekce anomálií v datech o znečištění ovzduší ze sensorické sítě
veřejného osvětlení v okolí Karlínského náměstí na Praze 8.

## Harmonogram

1. pondělí, 11. březen: zvolení tématu (data, metody, nástroje, vstupy, výstupy)
2. pondělí, 25. březen: první prototyp
3. pondělí, 8. duben: druhý prototyp
4. pondělí, 19. duben: třetí prototyp
5. pondělí, 6. květen: předfinální práce
6. pondělí, 13. květen: finální práce (prezentace, funkční kód, dokumentace, příklady)

## Data

Pražská datová platforma [Golemio][golemio] poskytuje data z pilotního provozu
[Senzorické sítě veřejného osvětlení][senzoricka_sit_verejneho_osvetleni],
v rámci kterého bylo nainstalováno 92 chytrých pouličních LED lamp v blízkosti
Karlínského náměstí na Praze 8. Některé z těchto lamp mají senzory pro měření a
sběr dat o hluku, prašnosti a množství dalších polutantů.

Výška senzorů je přibližně 4,5 metrů nad zemí a výčet senzoriky veřejného osvětlení je následující:

1. [Pevné částice][pevne_castice]: PM<sub>2,5</sub> a PM<sub>10</sub>, vzorkovací čas 60 sekund, vzorkovací interval 10 minut, rozsah pro PM<sub>2,5</sub>: 0–2000 mikrog/m<sup>3</sup>, rozsah pro PM<sub>10</sub>: 0-5000 µg/m<sup>3</sup>, přesnost 0,1 µg/m<sup>3</sup>.
2. Ozón: rozsah 2000 ppb, detekční limit 5 ppb, přesnost měření ± 60 ppb v typickém vnějším prostředí.
3. Oxid siřičitý (SO<sub>2</sub>): rozsah 2000 ppb, detekční limit 5 ppb, přesnost měření ± 50 ppb v typickém vnějším prostředí.
4. Oxid uhelnatý (CO): rozsah 10000 ppb, detekční limit 10 ppb, přesnost měření ± 200 ppb v typickém vnějším prostředí.
5. Oxid dusičitý (NO<sub>2</sub>): rozsah 2000 ppb, detekční limit 5 ppb, přesnost měření ± 25 ppb v typickém vnějším prostředí.

Datové zdroje z pilotního projektu v Karlíně jsou historická data polutantů z
senzorů lamp a aktualní data z lamp dostupná přes [API][api_lampy]
po získání API klíče. Z historických dat jsou dostupné dva CSV soubory s daty z
2. pololetí roku 2018 a 1. pololetí roku 2019, kdy data za 1. pololetí roku 2019
se zřejmě denně aktualizují.

## Metody

Při detekování anomálií v časových řadách se používá metoda, kdy je daná řada
předpověděna dopředu (například pomocí rekurentní neuronové sítě) a následně je
porovnána se skutečnými daty z nehož jsou určeny anomálie (například pomocí
váhování). Tímto přístupem se zabývá článek
[Time Series Anomaly Detection][time_series_anomaly_detection].
Dalším vhodným materiálem k prostudování je přehled algoritmů na detekci
anomálií [Anomaly Detection: A Survey][anomaly_detection_a_survey].

Protože data neobsahují žádná označení dřívějších anomalií, je třeba použít strojové učení bez učitele.

### Metrika úspěšnosti

Pro měření úspěšnosti predikce časové řady se jako vhodná zdá
*[střední kvadratická chyba][mse]* (mean squared error, MSE), kterou model
minimalizuje, ale
není žádoucí dosáhnout nulové chyby, protože potom by nebylo možné detekovat
anomálie. Tzn. model by se měl naučit pouze pravidelnosti v datech a nikoliv se
přeučit tak, aby si dokázal zapamatova anomálie.

V měření úspěšnosti detekce anomálií je problém absence označení anomálií
v datech. Detekce tedy musí být kontrolovány člověkem. Je možné dělat různé
statistické odhady úspěšnosti (např. odhad [matice záměň][confusion_matrix]),
ang. confusion matrix, precision, recall, PR curve nebo F-score).

### Základní (baseline) model

Základní model je pro jednoduchost předpovídat stejnou hodnotu jako je
hodnota předchozí.

### Long Short-Term Memory (LSTM)

Pokročilejší predikční model je [LSTM][lstm] rekurentní neuronová síť.
Úvod do rekurentních sítí: [The Unreasonable Effectiveness of Recurrent Neural Networks][rnn_effectiveness].

### Detekce anomálií

Jednoduché prahování.

## Nástroje

Programovací jazyk Python, jeho knihovny a Jupyter Notebook.

## Výstupy

Výstupem projektu jsou následující body:

1. natrénovaný algoritmus pro detekci anomálií,
2. vizualizace pro potřeby prezentace výsledků a
3. Jupyter notebooky s explorací a analýzou dat.

Další potenciální cíle projektu jsou:

1. vytvoření webového rozhraní,
2. srovnání dat o znečištění ovzduší z lamp s daty ČHMU (znečištění a počasí) a
3. analýza rozmístění sensorů (korelace jednotlivých časových řad).

## Reference

1. Dominique T. Shipmon, Json M. Gurevithc, Paolo M. Piselli and Steve Edwards. *Time Series Anomaly Detection*. Dostupné [online][time_series_anomaly_detection].
2. Ian Goodfellow, Yoshua Bengio and Aaron Courville. *Deep learning*. Dostupné [online][deep_learning].

[anomaly_detection_a_survey]: http://cucis.ece.northwestern.edu/projects/DMS/publications/AnomalyDetection.pdf
[api_lampy]: https://golemio.docs.apiary.io/#reference/0/lampy-v-karline/aktualni-senzoricka-data-z-lamp
[confusion_matrix]: https://en.wikipedia.org/wiki/Confusion_matrix
[deep_learning]: https://www.deeplearningbook.org
[lstm]: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
[golemio]: https://golemio.cz/
[mse]: https://en.wikipedia.org/wiki/Mean_squared_error
[pevne_castice]: https://cs.wikipedia.org/wiki/Pevn%C3%A9_%C4%8D%C3%A1stice
[rnn_effectiveness]: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
[senzoricka_sit_verejneho_osvetleni]: https://golemio.cz/cs/node/622
[time_series_anomaly_detection]: https://static.googleusercontent.com/media/research.google.com/cs//pubs/archive/dfd834facc9460163438b94d53b36f51bb5ea952.pdf
