# 자극제

## 자극제의 import templates

#### 일반적인(대개 공통)

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as ticker
import IPython.display
```

#### out 패스를 포함하는

```python
import os

NOTEBOOK_ID = "TEST"
OUTPUT_PATH = f"out/{NOTEBOOK_ID}/"

if not os.path.isdir(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
```

#### 음성(*.wav)
```python
import scipy.io.wavfile
import scipy.fftpack
```

#### 시뮬레이션(pygame)
```python
import pygame

pygame.init()
```


# 목록
## gavity
중력과 한계를 시험


## wave simul
[참조](https://angeloyeo.github.io/2019/08/29/Heat_Wave_Equation.html)


## soft interpolation
