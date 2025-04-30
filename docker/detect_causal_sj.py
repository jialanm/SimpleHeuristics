"""
Created a set of heuristics to recover causal splice junctions in the muscle truth set.
"""
import os

import numpy as np
import pandas as pd
import pyBigWig
import matplotlib.pyplot as plt


def main():
    test = pd.DataFrame(np.array([[1, 2], [2, 4]]))
    print(test)

