#!/usr/bin/env python



import channeltoy as ct
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

def run_tests():
    yo = ct.channeltoy(spacing=250, U = 0.0002,n=1.2)
    yo.plot_ss_channel(filename = "profile_Usmall.png")

    yo.set_U_values(U = 0.0001)
    yo.plot_ss_channel(filename = "profile_Ubig.png")



    yo.set_U_values(U = 0.0002)
    #yo.solve_timestep(base_level= 0,dt = 10)
    times, elevations = yo.transient_simulation(base_level = 0, dt = 200, start_time = 0, end_time = 500001, print_interval = 250000)

    yo.set_U_values(U = 0.0001)
    initial_z = yo.solve_steady_state_elevation()
    print("Initial z:")
    print(initial_z)

    yo.set_U_values(U = 0.0002)
    final_z = yo.solve_steady_state_elevation()
    print("final z:")
    print(final_z)

    print("Now initial z again")
    print(initial_z)

    yo.plot_transient_channel(times = times, elevations = elevations, initial_elevation = initial_z, final_elevation =final_z)

def run_tests2():
    yo = ct.channeltoy(spacing=500, U = 0.0002,K = 0.00005,n=1.2)

    yo.plot_channel(filename = "pre_channel_profile.png",show_area=True)

    yo.create_drainage_capture_channel(new_K = 0.00005, new_U = 0.0002, new_max_x = 10001,new_spacing = 500, new_X_0 = 10501, new_rho = 1.8, capture_location_fraction = 0.5)

    yo.plot_channel(filename = "capture_channel_profile.png",show_area=True)

    # Now run the transient channel
    #times, elevations = yo.transient_simulation(base_level = 0, dt = 1, start_time = 0, end_time = 500001, print_interval = 250000)



    #yo.plot_transient_channel(times = times, elevations = elevations, initial_elevation = initial_z, final_elevation =final_z)


if __name__ == "__main__":
    run_tests2()

