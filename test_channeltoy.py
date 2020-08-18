#!/usr/bin/env python



import channeltoy as ct
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

def run_tests():
    yo = ct.channeltoy(spacing=100)

    yo.plot_ss_channel()


if __name__ == "__main__":
    run_tests()

