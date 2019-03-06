# lampy

Projekt detekce anomálií v datech o znečištění ovzduší ze sensorické sítě veřejného osvětlení v okolí Karlínského náměstí na Praze 8.

## Harmonogram

- 11. březen: zvolení tématu (data, metody, nástroje, vstupy, výstupy)
- 25. březen: první prototyp
- 8. duben: druhý prototyp
- 19. duben: třetí prototyp
- 6. květen: předfinální práce
- 13. květen: finální práce (prezentace, funkční kód, dokumentace, příklady)

## Data

Pražská datová platforma [Golemio](https://golemio.cz/) poskytuje data z pilotního provozu [Senzorické sítě veřejného osvětlení](https://golemio.cz/cs/node/622), v rámci kterého bylo nainstalováno 92 chytrých pouličních LED lamp v blízkosti Karlínského náměstí na Praze 8. Některé z těchto lamp mají senzory pro měření a sběr dat o hluku, prašnosti a množství dalších polutantů.

Výška senzorů je přibližně 4,5 metrů nad zemí a výčet senzoriky veřejného osvětlení je následující:

1. [Pevné částice](https://cs.wikipedia.org/wiki/Pevn%C3%A9_%C4%8D%C3%A1stice): PM<sub>2,5</sub> a PM<sub>10</sub>, vzorkovací čas 60 sekund, vzorkovací interval 10 minut, rozsah pro PM<sub>2,5</sub>: 0–2000 mikrog/m<sup>3</sup>, rozsah pro PM<sub>10</sub>: 0-5000 µg/m<sup>3</sup>, přesnost 0,1 µg/m<sup>3</sup>.
2. Ozón: rozsah 2000 ppb, detekční limit 5 ppb, přesnost měření ± 60 ppb v typickém vnějším prostředí.
3. Oxid siřičitý (SO<sub>2</sub>): rozsah 2000 ppb, detekční limit 5 ppb, přesnost měření ± 50 ppb v typickém vnějším prostředí.
4. Oxid uhelnatý (CO): rozsah 10000 ppb, detekční limit 10 ppb, přesnost měření ± 200 ppb v typickém vnějším prostředí.
5. Oxid dusičitý (NO<sub>2</sub>): rozsah 2000 ppb, detekční limit 5 ppb, přesnost měření ± 25 ppb v typickém vnějším prostředí.

Datové zdroje z pilotního projektu v Karlíně jsou historická data polutantů z senzorů lamp a aktualní data z lamp dostupná přes [API](https://golemio.docs.apiary.io/#reference/0/lampy-v-karline/aktualni-senzoricka-data-z-lamp) po získání API klíče. Z historických dat jsou dostupné dva CSV soubory s daty z 2. pololetí roku 2018 a 1. pololetí roku 2019, kdy data za 1. pololetí roku 2019 se zřejmě denně aktualizují.

## Metody

Při detekování anomálií v časových řadách se používá metoda, kdy je daná řada předpověděna dopředu (například pomocí rekurentní neuronové sítě) a následně je porovnána se skutečnými daty z nehož jsou určeny anomálie (například pomocí váhování). Tímto přístupem se zabývá článek [Time Series Anomaly Detection](https://static.googleusercontent.com/media/research.google.com/cs//pubs/archive/dfd834facc9460163438b94d53b36f51bb5ea952.pdf). Dalším vhodným materiálem k prostudování je přehled algoritmů na detekci anomálií [Anomaly Detection: A Survey](http://cucis.ece.northwestern.edu/projects/DMS/publications/AnomalyDetection.pdf).

Protože data neobsahují žádná označení dřívějších anomalií je třeba použít strojové učení bez učitele.

## Nástroje

Programovací jazyk Python a jeho knihovny.

## Výstupy

Výstupy projektu jsou natrénovaný algoritmus pro detekci anomálií v datech o znečištění životního prostředí z chytrých lamp v okolí Karlínského náměstí a script nebo webové rozhraní online detekující anomálie online v aktuální datec API (případně algoritmus, který se z nich online také učí).
