"""Main module."""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import optimize
import statistics as st


def solve_timestep_differencer(z_future,z_downstream,z_past,dt,dx,U,K,A,m,n):
    """Solves the transient equations
        E = K A^m S^n
        dz/dt = U - E
        dz/dt = U - K A^m S^n
        (z^j+1-z^j)/dt = U - K A^m ((z_i^j+1-z_i-1^j+1)/dx)^n
        We move all terms to one side:
        0 = U - K A^m ((z_i^j+1-z_i-1^j+1)/dx)^n - (z^j+1-z^j)/dt
        we then assume this is a function
        z_predict = U - K A^m ((z_i^j+1-z_i-1^j+1)/dx)^n - (z^j+1-z^j)/dt
        and use a root finding algorithm to solve.
        We use Newton's method if n = 1 and the toms748 algorithm if n != 1.
        tom's is faster and guaranteed to converge, but does not work if function is not differentiable 4 times.
        we use upslope values of K, U, A for the discretization

    Args:
        uses all data members, no args

    Returns:
        Overwrites the elevation

    Author:
        Simon M Mudd

    Date:
        18/08/2020
    """

    difference = U - K*A**m * ( (z_future - z_downstream)/dx )**n - (z_future-z_past)/dt
    return difference




def axis_styler(ax,axis_style="Normal"):
    """This sets the line width and fonts on the axes.

    Args:
        ax_list (axes objects): the list of axis objects
        axis_style (string): The syle of the axis. See options below.

    Author: SMM
    """

    if axis_style == "Normal":
        lw = 1              # line width
        ftsz = 10           # Size of tick label font
        tpd = 2             # Tick padding
        label_ftsz = 12     # Fontsize of axis label
    elif axis_style == "Thick":
        lw = 2
        ftsz = 10
        tpd = 2
        label_ftsz = 12
    elif axis_style == "Thin":
        lw = 0.5
        ftsz = 8
        tpd = 1
        label_ftsz = 10
    elif axis_style == "Ultra_Thin":
        lw = 0.4
        ftsz = 4
        tpd = 0.3
        label_ftsz = 6
    elif axis_style == "Big":
        lw = 2
        ftsz = 12
        tpd = 3
        label_ftsz = 14
    elif axis_style == "Madhouse":
        # This is just a crazy style to test if the figure is actually recieving these instructions
        lw = 4
        ftsz = 20
        tpd = 3
        label_ftsz = 6
    else:
        print("Using the default axis styling")
        lw = 1
        ftsz = 10
        tpd = 2
        label_ftsz = 12


    # Now to fix up the axes
    ax.spines['top'].set_linewidth(lw)
    ax.spines['left'].set_linewidth(lw)
    ax.spines['right'].set_linewidth(lw)
    ax.spines['bottom'].set_linewidth(lw)

    ax.xaxis.label.set_size(label_ftsz)
    ax.yaxis.label.set_size(label_ftsz)

    # This gets all the ticks, and pads them away from the axis so that the corners don't overlap
    ax.tick_params(axis='both', width=lw, pad = tpd, labelsize = ftsz )

    return ax


class channeltoy():
    """
    This is the channeltoy object.

    Args:
        m (float): the area exponent
        n (float): the slope exponent
        minimum_x (float): The minimum value of x in metres (divide is at 0)
        maximum_x (float): The maximum value of x in metres  (divide is at 0)
        spacing (float): The spacing of x  in metres
        X_0 (float): Length of basin (in m)
        rho (float): Exponent from this modified version of hacks law
        K (float): bedrock erodibility. Units depend on m and n

    Returns:
        Creates a channeltoy object

    Author: SMM

    Date: 18/08/2020
    """
    def __init__(self, m= 0.45,  n = 1, minimim_x = 0, maximum_x = 9001,spacing = 1000,
                 X_0 = 10000, rho = 1.8,
                 K = 0.000005,
                 U = 0.0001):

        self.m_exponent = m
        self.n_exponent = n
        self.x_data = self.set_profile_locations_constant(minimum_x = minimim_x,
                                                     maximum_x = maximum_x,
                                                     spacing = spacing)
        self.z_data = np.zeros_like(self.x_data,dtype=float)
        self.A_data = self.set_hack_area(X_0 = X_0,
                                         rho = rho)

        self.K_data = self.set_K_values(K = K)
        self.U_data = self.set_U_values(U = U)

        print("m is: "+str(self.m_exponent)+" and n is: "+str(self.n_exponent))
        print("x locations are (m):")
        print(self.x_data)
        print("elevations are (m):")
        print(self.z_data)
        print("Areas are (m^2):")
        print(self.A_data)
        print("K values are:")
        print(self.K_data)
        print("Uplift rates are (m/yr):")
        print(self.U_data)

        self.solve_steady_state_elevation()

        print("elevation at steady is")
        print(self.z_data)


    def set_profile_locations_constant(self,minimum_x = 1000, maximum_x = 100000,spacing = 1000):
        """This is the most basic function for setting up the profile locations.
        Locations are distance from outlet

        Args:
            minimum_x (float): The minimum value of x in metres (divide is at 0)
            maximum_x (float): The maximum value of x in metres  (divide is at 0)
            spacing (float): The spacing of x  in metres

        Returns:
            A numpy array with the channel locations in metres

        Author:
            Simon M Mudd

        Date:
            14/05/2020
        """
        x_locs = np.arange(minimum_x,maximum_x,spacing)
        return x_locs

    def set_hack_area(self,X_0 = 10000, rho = 1.8):
        """This sets the area on the basis of Hack's law

        Args:
            X_0 (float): Length of basin (in m)
            rho (float): Exponent from this modified version of hacks law

        Returns:
            A numpy array with the drainage area metres^2

        Author:
            Simon M Mudd

        Date:
            18/08/2020
        """
        A_locs = np.power(np.subtract(X_0,self.x_data),rho)
        self.A_data = A_locs
        return A_locs

    def set_K_values(self,K = 0.000005):
        """This sets the erodibility values

        Args:
            K(float): bedrock erodibility. Units depend on m and n

        Returns:
            A numpy array with the K values

        Author:
            Simon M Mudd

        Date:
            18/08/2020
        """
        print("K is: "+str(K))
        K_vals = np.full_like(self.x_data,K,dtype=float)
        self.K_data = K_vals
        return K_vals

    def set_U_values(self,U = 0.0001):
        """This sets the uplift values

        Args:
            U (float): uplift rates in m/yr

        Returns:
            A numpy array with the U values

        Author:
            Simon M Mudd

        Date:
            18/08/2020
        """
        print("U is: "+str(U))
        U_vals = np.full_like(self.x_data,U,dtype=float)
        self.U_data = U_vals
        return U_vals

    def solve_steady_state_elevation(self,base_level = 0):
        """Solves the steady state equations
        E = K A^m S^n
        U = E
        U = K A^m S^n
        S^n = U / (K A^m)
        S = (U / (K A^m))^1/n
        z_i+1 = z_i + dz((U / (K A^m))^1/n)
        we use upslope values of K, U, A for the discretization

        Args:
            uses all data members, no args

        Returns:
            the elevations. Also overwrites the elevation data member

        Author:
            Simon M Mudd

        Date:
            18/08/2020
        """

        #print("The U in the first cell is:")
        #print(self.U_data[0])

        # Please forgive me, but I am going to do this in a
        yoyoma = range(1,self.x_data.size)
        #print("Range is:")
        #print(yoyoma)

        Apow = np.power(self.A_data,self.m_exponent)
        term1 = np.multiply(self.K_data,Apow)
        term2 = np.divide(self.U_data,term1)
        term3 = np.power(term2,(1/self.n_exponent))
        #print("Slope is:")
        #print(term3)

        z = np.copy(self.z_data)
        z[0]= base_level
        for i in range(1,self.x_data.size):
            z[i] = z[i-1]+(self.x_data[i]-self.x_data[i-1])*term3[i]
        self.z_data = np.copy(z)
        return z



    def solve_timestep_point(self,z_min,z_max,z_downstream,z_past,dt,dx,U,K,A,m,n):
        """Solves the transient equations
        E = K A^m S^n
        dz/dt = U - E
        dz/dt = U - K A^m S^n
        (z^j+1-z^j)/dt = U - K A^m ((z_i^j+1-z_i-1^j+1)/dx)^n
        We move all terms to one side:
        0 = U - K A^m ((z_i^j+1-z_i-1^j+1)/dx)^n - (z^j+1-z^j)/dt
        we then assume this is a function
        z_predict = U - K A^m ((z_i^j+1-z_i-1^j+1)/dx)^n - (z^j+1-z^j)/dt
        and use a root finding algorithm to solve.
        We use Newton's method if n = 1 and the toms748 algorithm if n != 1.
        tom's is faster and guaranteed to converge, but does not work if function is not differentiable 4 times.
        we use upslope values of K, U, A for the discretization
        Args:
            uses all data members, no args

        Returns:
            Overwrites the elevation

        Author:
            Simon M Mudd

        Date:
            18/08/2020
        """

        if n == 1:
            z_future = optimize.newton(solve_timestep_differencer,z_past,args=(z_downstream,z_past,dt,dx,U,K,A,m,n))
        else:
            z_future = optimize.toms748(solve_timestep_differencer,z_min,z_max,args=(z_downstream,z_past,dt,dx,U,K,A,m,n))

        #print("Z_future is: ")
        #print(z_future)

        return z_future


    def solve_timestep(self, base_level= 0,dt = 1):
        """Solves the transient equations
        E = K A^m S^n
        dz/dt = U - E
        dz/dt = U - K A^m S^n
        (z^j+1-z^j)/dt = U - K A^m ((z_i^j+1-z_i-1^j+1)/dx)^n
        We move all terms to one side:
        0 = U - K A^m ((z_i^j+1-z_i-1^j+1)/dx)^n - (z^j+1-z^j)/dt
        we then assume this is a function
        z_predict = U - K A^m ((z_i^j+1-z_i-1^j+1)/dx)^n - (z^j+1-z^j)/dt
        and use a root finding algorithm to solve.
        We use Newton's method if n = 1 and the toms748 algorithm if n != 1.
        tom's is faster and guaranteed to converge, but does not work if function is not differentiable 4 times.
        we use upslope values of K, U, A for the discretization



        Args:
            base_level (float): The elevation at the outlet
            dt (float): the timestep

        Returns:
            Overwrites the elevation, also returns the elevation array

        Author:
            Simon M Mudd

        Date:
            19/08/2020
        """

        z = np.copy(self.z_data)
        z[0]= base_level

        m = self.m_exponent
        n = self.n_exponent

        for i in range(1,self.x_data.size):
        #for i in range(1,2):
            A = self.A_data[i]
            K = self.K_data[i]
            U = self.U_data[i]
            z_past = z[i]
            z_downstream = z[i-1]
            dx = self.x_data[i]-self.x_data[i-1]
            z_min = z_past-dt*U*100
            z_max = z_past+dt*U*100
            z_future=self.solve_timestep_point(z_min,z_max,z_downstream,z_past,dt,dx,U,K,A,m,n)
            z[i] = z_future


        self.z_data = np.copy(z)
        return z


    def transient_simulation(self,base_level = 0, dt = 1, start_time = 0, end_time = 100000, print_interval = 1000):
        """Solves the transient evolution of a channel over a time period

        Args:
            base_level (float): The elevation at the outlet
            dt (float): the timestep
            start_time (float): the starting time
            end_time (float): the ending time

        Returns:
            times (list, float): a list of the times when the elevation is recorded
            elevations (list, float array): a list of the elevations at the recorded times
            Overwrites the elevation as it goes

        Author:
            Simon M Mudd

        Date:
            19/08/2020
        """

        t_ime = np.arange(start_time, end_time+0.5*dt, dt, dtype=float)

        times = []
        elevations = []

        for t in t_ime:
            elev = self.solve_timestep(base_level= base_level,dt = dt)
            if t%print_interval == 0:
                print("Time is: "+str(t))
                print("Elevation is ")
                print(elev)
                times.append(t)
                elevations.append(elev)
            if t%5000 == 0:
                print("Time is: "+str(t))

        return times,elevations


    def plot_transient_channel(self,show_figure = False, print_to_file = True, filename = "transient_channel_profile.png", times = [], elevations = [], initial_elevation = [], final_elevation =[]):
        """This prints the channel profile
        Args:
            show_figure (bool): If true, show figure
            print_to_file (bool): If true, print to file
            filename (string): Name of file to which the function prints

        Returns:
            Either a shown figure, a printed figure, or both

        Author:
            Simon M Mudd

        Date:
            18/08/2020
        """

        x_loc = self.x_data

        fig, ax = plt.subplots()
        if not final_elevation == []:
            #print("Plotting final elevation")
            #print(final_elevation)
            ax.plot(self.x_data, final_elevation, label="Final steady state profile")

        if not initial_elevation == []:
            #print("Plotting initial elevation")
            #print(initial_elevation)
            ax.plot(self.x_data, initial_elevation, label="Initial steady state profile")

        for i, t in enumerate(times):
            print("Time is: "+str(t))
            #print("Elevation is:")
            #print(elevations[i])
            ax.plot(self.x_data, elevations[i], label="Time = "+str(t))



        plt.legend(loc='upper left')
        ax.set(xlabel='distance from outlet (m)', ylabel='elevation (m)',
           title='Channel profile')
        ax.grid()
        ax = axis_styler(ax,axis_style="Normal")

        if print_to_file:
            fig.savefig(filename)

        if show_figure:
            plt.show()


    def plot_ss_channel(self,show_figure = False, print_to_file = True, filename = "channel_profile.png"):
        """This prints the channel profile
        Args:
            show_figure (bool): If true, show figure
            print_to_file (bool): If true, print to file
            filename (string): Name of file to which the function prints

        Returns:
            Either a shown figure, a printed figure, or both

        Author:
            Simon M Mudd

        Date:
            18/08/2020
        """

        self.solve_steady_state_elevation()

        fig, ax = plt.subplots()
        ax.plot(self.x_data, self.z_data)

        ax.set(xlabel='distance from outlet (m)', ylabel='elevation (m)',
           title='Channel profile')
        ax.grid()
        ax = axis_styler(ax,axis_style="Normal")

        if print_to_file:
            fig.savefig(filename)

        if show_figure:
            plt.show()
