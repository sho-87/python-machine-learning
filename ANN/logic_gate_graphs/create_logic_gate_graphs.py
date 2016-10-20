import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D  

# AND gate
and_data = pd.DataFrame(
    np.array([[0,0,0], [0,1,0], [1,0,0], [1,1,1]]),
    columns=["Input 1","Input 2", "Result"]
    )

groups = and_data.groupby('Result')

# Plot
fig, ax = plt.subplots()
ax.margins(0.05)
for name, group in groups:
    if name == 0:
        cur_label = "Off"
    else:
        cur_label = "On"
    ax.plot(group["Input 1"], group["Input 2"], marker='o', linestyle='',
            ms=12, label=cur_label)
    
ax.legend(numpoints=1, loc='best')
l = Line2D([0, 1.5], [1.5, 0], linestyle='--', color='red')                                    
ax.add_line(l)

plt.xticks([0,1])
plt.yticks([0,1])
plt.title("AND gate")
plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.show()

# OR gate
or_data = pd.DataFrame(
    np.array([[0,0,0], [0,1,1], [1,0,1], [1,1,1]]),
    columns=["Input 1","Input 2", "Result"]
    )

groups = or_data.groupby('Result')

# Plot
fig, ax = plt.subplots()
ax.margins(0.05)
for name, group in groups:
    if name == 0:
        cur_label = "Off"
    else:
        cur_label = "On"
    ax.plot(group["Input 1"], group["Input 2"], marker='o', linestyle='', 
            ms=12, label=cur_label)
    
ax.legend(numpoints=1, loc='best')
l = Line2D([-1, 1.5], [1.5, -1], linestyle='--', color='red')                                    
ax.add_line(l)

l2 = Line2D([-1, 1], [1, -1], linestyle=':', color='blue')                                    
ax.add_line(l2)

plt.xticks([0,1])
plt.yticks([0,1])
plt.title("OR gate")
plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.show()

# XOR gate
or_data = pd.DataFrame(
    np.array([[0,0,0], [0,1,1], [1,0,1], [1,1,0]]),
    columns=["Input 1","Input 2", "Result"]
    )

groups = or_data.groupby('Result')

# Plot
fig, ax = plt.subplots()
ax.margins(0.05)
for name, group in groups:
    if name == 0:
        cur_label = "Off"
    else:
        cur_label = "On"
    ax.plot(group["Input 1"], group["Input 2"], marker='o', linestyle='', 
            ms=12, label=cur_label)
    
ax.legend(numpoints=1, loc='best')

plt.xticks([0,1])
plt.yticks([0,1])
plt.title("XOR gate")
plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.show()