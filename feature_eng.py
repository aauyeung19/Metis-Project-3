import sys
print(sys.version)

import pandas as pd
import numpy as np
import pickle

wdf = pd.read_pickle('src/EWRweather.pickle')

humdf = wdf[['date', 'humid_avg']]
humdf = humdf.set_index('date')
humdf.shift(-1)
humdf.reset_index(inplace=True)
humdf.rename(columns={'humid_avg': 'humid_avg_lag1'}, inplace=True)

wdf.merge(humdf, on='date')

def set_precip_level(thresh):
    """
    Sets the threshold on precipitation level to be considered rain
    args:
        df : (DataFrame) df with precipitation level
        thresh : (float) Sets the decimal threshold for how many inches of precipitation to consider it raining

    returns:
        df : (DataFrame) with new column with 1 or 0 for rain or no
    """

    wdf['raining'] = wdf.precip.map(lambda x: int(x > thresh))

set_precip_level(0.5)


