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
```

```python
df = pd.read_csv('data/persistent_network.csv')
df.head()
```

```python
len(set(df['Target']))
```

```python

```
