#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

if len(sys.argv) > 1 and os.path.exists(sys.argv[1]) and os.path.isfile(sys.argv[1]):
    res = np.loadtxt(sys.argv[1], delimiter=',')
    fig, ax = plt.subplots()

    ax.imshow(res, interpolation='bicubic', cmap=plt.get_cmap("terrain"))
    plt.axis("off")
    plt.tight_layout()

    plt.show()
