#!/usr/bin/env python



import channeltoy as ct
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

def run_tests():
    yo = ct.channeltoy(spacing=100, U = 0.00001,n=1.2)
    yo.plot_ss_channel(filename = "profile_Usmall.png")

    yo.set_U_values(U = 0.0001)
    #yo.plot_ss_channel(filename = "profile_Ubig.png")




    #yo.solve_timestep(base_level= 0,dt = 10)
    yo.transient_simulation(base_level = 0, dt = 10, start_time = 0, end_time = 5000, print_interval = 1000)




if __name__ == "__main__":
    run_tests()

