# TODO list

Úkoly, které je potřeba dodělat nebo by bylo dobré v budoucnu implementovat
(řazené podle priority):

- jeden Jupyter notebook s detailní analýzou všeho,
    - rozmyslet zda udělat optimalizace hyperparametrů LSTM
        (random search, viz `optimalizace-hyperparametru.ipynb`, pro zrychlení přesunout na GPU),
    - udělat závěreční odhad střední kvadratické odchylky (RMSE) na testovacích datech,
- vyvinout verzi, která bude hledat anomálie online pomocí Golemio API,
- sestavit HTML report (jako GitHub page s distill šablonou),
- napsat dokumentaci (v podobě docstring u funkcí nebo vygenerovat pomocí Read the Docs).

Možné úkoly, které nesplním: 

- testovaní je pouze z hlediska MSE a nikoliv podle detekovaných anomálií
    (to může udělat pouze člověk prohlédnutím detekcí),
- detekuju pouze pro jednu lampu, ale lepší by bylo spojit více lamp.