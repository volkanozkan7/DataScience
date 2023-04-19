# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 16:43:21 2023

@author: volka
"""

from sklearn.model_selection import train_test_split
import mglearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

demo_df = pd.DataFrame({'Integer Feature': [0, 1, 2, 1],'Categorical Feature': ['socks', 'fox', 'socks', 'box']})
display(demo_df)
ddemo_df = pd.get_dummies(demo_df)
display(ddemo_df)

demo_df['Integer Feature'] = demo_df['Integer Feature'].astype(str)
pd.get_dummies(demo_df, columns=['Integer Feature', 'Categorical Feature'])