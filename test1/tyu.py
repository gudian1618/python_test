import numpy as np
import matplotlib.pyplot as plt

b = np.sin(12)
print(b)

x = np.linspace(-np.pi, np.pi, 201)
y = np.sin(x)

plt.plot(x, y)
plt.show()