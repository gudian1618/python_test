# -*- coding:utf-8 -*-

import numpy as np
import timeit
import matplotlib.pyplot as plt
import matplotlib as mpi
mpi.rcParams['font.sans-serif']=['SimHei']

integers = []

def dosort():
	integers.sort()


def meature():
	timer = timeit.Timer('dosort()', 'from __main__ import dosort')
	return timer.timeit(10 **2)

powersOf2 = np.arange(0, 19)

sizes = 2**powersOf2

times = np.array([])

for size in sizes:
	integers = np.random.random_integers(1, 10**6, size)
	times = np.append(times, meature())
	
fit = np.polyfit(sizes*powersOf2, times, 1)
print('fit')
plt.title('排序时间')
plt.xlabel('尺寸')
plt.ylabel('时间')
plt.semilogx(sizes, times, 'ro')
plt.semilogx(sizes, np.polyval(fit, sizes*powersOf2))
plt.legend(loc='best')
plt.grid()
plt.show()










