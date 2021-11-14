# RwAttribution
Default and custom attribution mode &amp; pipeline for them

## Модуль tools
В данном модуле находятся различные методы для работы с данными, на основе которых будет строиться атрибуция.

Основная предобработка находится в функции `prep_data` модуля `tools.prep`.
Функция принимает pandas.DataFrame следующей структуры: 
- client_id_col - это может быть id конверсии, client_id из Google Analytics или другой системы и т.д.;
- channel_col - столбец с каналом(может быть и другая детализация/группировка) точки касания;
- interaction_type_col - столбец с типом взаимодействия - "Click" или "Impression";
- conversion_col - столбец с флагом наличия/отсутствия конверсии. Может быть задан как в boolean, так и в int;
- order_col - столбец, по которому можно определить номер касания в конверсионной цепочке;
Пример данных: 
<img width="707" alt="Снимок экрана 2021-04-20 в 10 09 29" src="https://user-images.githubusercontent.com/60659176/115352860-801b5600-a1c0-11eb-8b09-e5f17aac1791.png">

В модуле `tools.influence` находятся методы для "предиктивной атрибуции". 
С их помощью возможно изменять цепочки в зависимости от перераспределения 
бюджета, охвата и т.д. Подробности читайте в документации методов.

## Модуль attribution
В данном модуле реализованы различные модели атрибуции, от простейших, до
кастомных.

В `attribution.basic_attribution` находятся базовые модели атрибуции:
- last click
- first click
- uniform(linear)
- last non-direct click
- time decay
- position based

Каждый метод принимает на вход данные, полученные после предобработки при помощи
`tools.prep.prep_data`

В `RwShapley` реализован класс RwShapley, который сочетает в себе базовую 
модель атрибуции Шэпли, и кастомную, реализуемую при помощи `tools.prep.compute_FIC`

В модуле `markov` реализован класс `RwMarkov`. Эта модель атрибуции требует
других данных - если в данных до этого используются **только** конверсионные 
цепочки, то для этого метода в данных также должны быть цепочки, которы не 
привели к конверсии.

Пример загрузки и обработки данных:
```python
from tools import prep_data
import pandas as pd
import plotly.express as px
from attribution import uniform

markov_data = pd.read_csv('../data/attribution data.csv', sep=',')
markov_data['interaction'] = (
    markov_data['interaction']
    .apply(lambda x: 'click' if x == 'conversion' else 'view'))

# Data preprocessing
markov_full, markov_cl, markov_v = prep_data(markov_data, 'channel',
                                             'cookie', 'interaction')
markov_data.head()
```
Uniform атрибуция 
```python
uniform_results = uniform(markov_full, markov_data['channel'].unique())
```
Атрибуция, основанная на цепях Маркова 
```python
markov = RwMarkov(markov_data, 'channel', 'conversion', 'cookie', 'time', verbose=0)
m_res = markov.make_markov() 

fig = px.bar(y = m_res.keys(), x = m_res.values(), title = 'Markov', orientation = 'h')
fig.update_yaxes(title_text = 'Группа каналов')
fig.update_xaxes(title_text = 'Конверсии')
fig.show()
```

## Spark
Для больших датасетов производительности питона уже не хватает. Для обработки
больших датасетов используйте BQ, шаблоны сможете найти в папке `sql`.

Также, большие датасеты можно локально обрабатывать в Apache spark. Такой 
пайплайн реализован в `Attribution spark.ipynb`

Для того чтобы работать с этим ноубуком необходимо установить ядро Scala.

Следуйте инструкции https://almond.sh/docs/quick-start-install

**ВАЖНО!** обязательно устанавливайте scala версии 2.12 или ранее, т.к. Spark
не работает с более поздними.