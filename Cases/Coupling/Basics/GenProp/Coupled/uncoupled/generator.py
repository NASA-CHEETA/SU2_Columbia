import numpy as np
import time


# Define computational domain
n = 20
dn = 1/ n


# Generate Mesh
y = np.linspace(0, 1, n+1)

# Time parameters
dt = 0.01
t = 0

while True:
    print("Generating data...")
    time.sleep(0.2)
    u = 1 -2 * np.random.rand(n)

    # Advance time
    t = t + dt

    if (t > 0.1):
        print("Time window reached..Exiting!\n")
        break