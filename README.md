# lampy

Detekce anomálií v datech o znečištění ze sensorické sítě veřejného osvětlení v okolí Karlínského náměstí na Praze 8.

## Harmonogram

1. 11. březen: zvolení tématu (data, metody, nástroje, vstupy, výstupy)
2. 25. březen: první prototyp
3. 8. duben: druhý prototyp
4. 19. duben: třetí prototyp
5. 6. květen: předfinální práce
6. 13. květen: finální práce (prezentace, funkční kód, dokumentace, příklady)

## Data

Pražská datová platforma [Golemio](https://golemio.cz/) poskytuje data z pilotního provozu [Senzorické sítě veřejného osvětlení](https://golemio.cz/cs/node/622), v rámci kterého bylo nainstalováno 92 chytrých pouličních LED lamp v blízkosti Karlínského náměstí na Praze 8. Některé z těchto lamp mají senzory pro měření a sběr dat o hluk, prašnosti a množství dalších polutatntů.

Výška senzorů je přibližně 4,5 metrů nad zemí a výčet senzoriky veřejného osvětlení je následující:

1. Prach: PM<sub>2,5</sub> a PM<sub>10</sub>, vzorkovací čas 60 sekund, vzorkovací interval 10 minut, rozsah pro PM<sub>2,5</sub>: 0–2000 mikrog/m<sup>3</sup>, rozsah pro PM<sub>10</sub>: 0-5000 µg/m<sup>3</sup>, přesnost 0,1 µg/m<sup>3</sup>.
2. Ozón: rozsah 2000 ppb, detekční limit 5 ppb, přesnost měření ± 60 ppb v typickém vnějším prostředí.
3. Oxid siřičitý (SO<sub>2</sub>): rozsah 2000 ppb, detekční limit 5 ppb, přesnost měření ± 50 ppb v typickém vnějším prostředí.
4. Oxid uhelnatý (CO): rozsah 10000 ppb, detekční limit 10 ppb, přesnost měření ± 200 ppb v typickém vnějším prostředí.
5. Oxid dusičitý (NO<sub>2</sub>): rozsah 2000 ppb, detekční limit 5 ppb, přesnost měření ± 25 ppb v typickém vnějším prostředí.

Datové zdroje z pilotního projektu v Karlíně jsou historická data polutatntů z senzorů Karlínských lamp a aktualní data z lamp v Karlíně dostupná přes [API](https://golemio.docs.apiary.io/#reference/0/lampy-v-karline/aktualni-senzoricka-data-z-lamp) po získání API klíče. Z historických dat jsou dostupné dva CSV soubory s daty z 2. pololetí 2018 a 1. pololetí 2019, kdy data za 1. pololetí 2019 se zřejmě denně aktualizují.

## Metody

## Nástroje

Programovací jazyk Python a jeho knihovny.

## Výstupy

Výstupy projektu jsou následující:

1. natrénovaný algoritmus pro detekci anomálií v datech o znečištění životního prostředí z chytrých lamp v okolí Karlínského náměstí,
2. script nebo webové rozhraní monitorující online aktuální data z API a případně algoritmus, který se z nich online učí.
