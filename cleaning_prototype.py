"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

import pandas as pd
import numpy as np
import pickle

wdf = pd.read_pickle('src/EWRweather.pickle')

wdf.to_pickle('src/EWRweather.pickle')