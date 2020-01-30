---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```

```python
df = pd.read_csv('data/persistent_network.csv')
df.head()
```

```python
len(set(df['Target']))
```

```python
import json

with open('data/vevo_full_series.json') as f:
    corpus = json.load(f)
```

```python
# Let's find the spikes
for k, series in corpus.items():
    series = np.array(series)
    series += 1 # Avoid division by zero
    diff = series[1:] / series[:-1]
    if np.max(diff) > 100:
        print(k, np.max(diff), np.argmax(diff))
```

```python
plt.plot(corpus['5393'])
```

```python
plt.plot(corpus['48365'])
```

```python
plt.plot(corpus['55559'])
```

```python

```
