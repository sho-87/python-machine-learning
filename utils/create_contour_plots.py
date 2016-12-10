import numpy as np
import matplotlib.pyplot as plt

f, (ax1, ax2) = plt.subplots(1, 2, sharey=False,figsize=(8,4))

# Subplot 1 (left)
xlist = np.linspace(-4.0, 4.0, 100)
ylist = np.linspace(-2.0, 2.0, 100)
X, Y = np.meshgrid(xlist, ylist)
Z = np.sqrt(X ** 2 + Y ** 2 )
levels = [0.0, 0.2, 0.5, 0.9, 1.5, 2.5, 3.5]
contour1 = ax1.contour(X, Y, Z, levels)

# Create legend items
lines = []
for i in range(len(levels)):
    lines.append(contour1.collections[i])

# Subplot 2 (right)
xlist = np.linspace(-4.0, 4.0, 100)
ylist = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(xlist, ylist)
Z = np.sqrt(X ** 2 + Y ** 2 )
levels = [0.0, 0.2, 0.5, 0.9, 1.5, 2.5, 3.5]
contour2 = ax2.contour(X, Y, Z, levels)

# set titles
f.suptitle('Contour Plots', fontweight="bold", size=14)
ax1.set_title('Raw', fontweight="bold")
ax2.set_title('Standardized', fontweight="bold")

ax1.grid(True)
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_xlabel('Feature 2')
ax1.set_ylabel('Feature 1')

ax2.grid(True)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_xlabel('Feature 2')
ax2.set_ylabel('Feature 1')

# Adjust layout
plt.figlegend(lines, levels, title='Loss', loc="center left", bbox_to_anchor=(0,0.64))
plt.tight_layout()
plt.subplots_adjust(top=0.86)
