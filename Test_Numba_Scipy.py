import numpy as np
from numba import jit, prange
import scipy.integrate as spi
import scipy.interpolate as spip
import scipy.special as sps
import math as math
import time
import tqdm
import sys
import json

@jit(nopython=True)
def main(argv):
    print(sps.jv(1.,1.))

if __name__ == '__main__':
    main(sys.argv)