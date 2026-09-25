import numpy as np
from diffusion_functions import updated_concentration, check_stability
from plots import concentration_profile_over_time

# user inputs
Length = float(input("Enter the length of material in meters:"))
initial_concentration = float(
    input("Enter the initial concentration in the material in mol/m^3:"))
surface_concentration = float(
    input("Enter the surface concentration in mol/m^3:"))


D = float(input("Enter the diffusion coefficient in m^2/s:"))
dx = float(input("Enter the spatial step size in meters:"))
dt = float(input("Enter the time step size in seconds:"))
total_time = float(input("Enter the total simulation time in seconds:"))

# material grid

x = np.arange(0, Length + dx, dx)
# x is the length of the material paritioned into steps of dx, from 0 to Length inputted
C = np.ones(len(x))*initial_concentration
# C is the the concentration array of the initial material

# boundary conditions of both ends of the material (the surface and the end of the material)
C[0] = surface_concentration
C[-1] = C[-2]

# Stability check
stability_value = check_stability(D, dt, dx)
if stability_value > 0.5:
    print("Simulation stopped because it is unstable."
          "Try reducing dt or increasing dx.")
else:
    print("Simulation is stable. Running simulation...")

    # running of simulation

    number_of_steps = int(total_time/dt)

    snapshot_times = [0, total_time/4,
                      total_time/2, 3*total_time/4, total_time]

    snapshots = {}

    # Data storage for further plots will be inputted here
    profiles = []
    times = []
    save_every = 10

    for step in range(number_of_steps + 1):
        current_time = step * dt

        for snap_time in snapshot_times:  # For numerical output and snapshot plot
            if abs(current_time - snap_time) < dt/2:
                snapshots[snap_time] = C.copy()
            if step % save_every == 0:  # for concentration map
                profiles.append(C.copy())
                times.append(current_time)
        C = updated_concentration(C, D, dt, dx)

# Reapply boundary conditions after each update
        C[0] = surface_concentration
        C[-1] = initial_concentration

    print("\n Simulation Complete")
    print("Saved Snapshots:")

    for time, concentration in snapshots.items():
        print(f"\ntime = {time:.4f} seconds")
        print(concentration)
concentration_profile_over_time(x, profiles, times)
