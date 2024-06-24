# src/cubicspline.py

# This file contains the implementation of Cubic Spline interpolation using CuPy and Numba.
# The CubicHermiteSpline class is a subclass of the PPoly class, which is a piecewise polynomial class.
# The CpCubicSpline class is a subclass of the CubicHermiteSpline class.
#
# This code was adapted from the more general Interpolation_CUPY project found at:
# https://github.com/GavinJiacheng/Interpolation_CUPY

import cupy as cp
import functools
from numba import cuda
from numpy.linalg import LinAlgError
import operator
import scipy.linalg as scl


@cuda.jit('void(float64[:,:,:], float64[:], float64[:], float64[:,:])')
def evaluate(c, x, xp, out):
    dx = 0
    extrapolate = True
    interval = 0
    start = cuda.grid(1)
    #start = 0
    stride = cuda.gridsize(1)
    #stride =1
    length = len(xp)
    cshape2 = c.shape[2]
    for ip in range(start, length, stride):
        xval = xp[ip]
        # Find correct interval
        # funciton start: -------------------------------------------------------
        a = x[0]
        b = x[x.shape[0]-1]

        it = 0
        if it < 0 or it >= x.shape[0]:
            it = 0

        if not (a <= xval <= b):
        # Out-of-bounds (or nan)
            if xval < a and extrapolate:
            # below
                it = 0
            elif xval > b and extrapolate:
            # above
                it = x.shape[0] - 2
            else:
            # nan or no extrapolation
                it = -1
        elif xval == b:
        # Make the interval closed from the right
            it = x.shape[0] - 2
        else:
        # Find the interval the coordinate is in
        # (binary search with locality)
            if xval >= x[it]:
                low = it
                high = x.shape[0]-2
            else:
                low = 0
                high = it

            if xval < x[low+1]:
                high = low
            while low < high:
                mid = (high + low)//2
                if xval < x[mid]:
                # mid < high
                    high = mid
                elif xval >= x[mid + 1]:
                    low = mid + 1
                else:
                # x[mid] <= xval < x[mid+1]
                    low = mid
                    break

            it = low
        # function end -----------------------------------------------------------------
        i = it
        if i < 0:
            for jp in range(0, cshape2, 1):
                out[ip, jp] = 0
            continue
        else:
            interval = i

        ci = interval
        for jp in range(0, cshape2, 1):
            ss = xval - x[interval]
            cj = jp
            # function start: ----------------------------------------------------------------------
            res = 0.0
            z = 1.0
            cshape1 = c.shape[0]

            for kp in range(0, cshape1, 1):
                # prefactor of term after differentiation
                if dx == 0:
                    prefactor = 1.0
                elif dx > 0:
                    # derivative
                    if kp < dx:
                        continue
                    else:
                        prefactor = 1.0
                        for k in range(kp, kp - dx, -1):
                            prefactor *= k
                else:
                    # antiderivative
                    prefactor = 1.0
                    for k in range(kp, kp - dx):
                        prefactor /= k + 1

                res = res + c[c.shape[0] - kp - 1, ci, cj] * z * prefactor

                if kp < c.shape[0] - 1 and kp >= dx:
                    z *= ss
            # function end ----------------------------------------------------------------------
            out[ip][jp] = res
    #cuda.defer_cleanup()
    # out[1], out[2]
    # out[2], out[1]
    # f(x) = a + bx + cx^2 .....

def prod(x):
    """Product of a list of numbers; ~40x faster vs np.prod for Python tuples"""
    if len(x) == 0:
        return 1
    return functools.reduce(operator.mul, x)

def solve_banded(l_and_u, ab, b, overwrite_ab=True, overwrite_b=True, debug=None, check_finite=False):
    a1 = cp.asnumpy(ab)
    b1 = cp.asnumpy(b)
    if a1.shape[-1] != b1.shape[0]:
        raise ValueError("shapes of ab and b are not compatible.")
    (nlower, nupper) = l_and_u

    if a1.shape[-1] == 1:
        b2 = cp.array(b1, copy=(not overwrite_b))
        b2 /= a1[1, 0]
        return b2
    if nlower == nupper == 1:
        gtsv, = scl.get_lapack_funcs(('gtsv',), (a1, b1))  # get_lapack_funcs
        du = a1[0, 1:]
        d = a1[1, :]
        dl = a1[2, :-1]
        du2, d, du, x, info = gtsv(dl, d, du, b1, overwrite_ab, overwrite_ab, overwrite_ab, overwrite_b)  # gtsv
    if info == 0:
        return x
    if info > 0:
        raise LinAlgError("singular matrix")
    raise ValueError('illegal value in %d-th argument of internal gbsv/gtsv' % -info)

def prepare_input(x, y, axis, dydx=None):
    x, y = map(cp.asarray, (x, y))
    if cp.issubdtype(x.dtype, cp.complexfloating):
        raise ValueError("`x` must contain real values.")
    x = x.astype(float)

    if cp.issubdtype(y.dtype, cp.complexfloating):
        dtype = complex
    else:
        dtype = float

    if dydx is not None:
        dydx = cp.asarray(dydx)
        if y.shape != dydx.shape:
            raise ValueError("The shapes of `y` and `dydx` must be identical.")
        if cp.issubdtype(dydx.dtype, cp.complexfloating):
            dtype = complex
        dydx = dydx.astype(dtype, copy=False)

    y = y.astype(dtype, copy=False)
    axis = axis % y.ndim
    if x.ndim != 1:
        raise ValueError("`x` must be 1-dimensional.")
    if x.shape[0] < 2:
        raise ValueError("`x` must contain at least 2 elements.")
    if x.shape[0] != y.shape[axis]:
        raise ValueError("The length of `y` along `axis`={0} doesn't match the length of `x`".format(axis))

    if not cp.all(cp.isfinite(x)):
        raise ValueError("`x` must contain only finite values.")
    if not cp.all(cp.isfinite(y)):
        raise ValueError("`y` must contain only finite values.")

    if dydx is not None and not cp.all(cp.isfinite(dydx)):
        raise ValueError("`dydx` must contain only finite values.")

    dx = cp.diff(x)
    if cp.any(dx <= 0):
        raise ValueError("`x` must be strictly increasing sequence.")

    y = cp.rollaxis(y, axis)
    if dydx is not None:
        dydx = cp.rollaxis(dydx, axis)

    return x, dx, y, axis, dydx

class PPoly(object):
    __slots__ = ('c', 'x', 'extrapolate', 'axis')

    def __init__(self, c, x, extrapolate=None, axis=0):
        self.c = cp.asarray(c)
        self.x = cp.ascontiguousarray(x, dtype=cp.float64)
        if extrapolate is None:
            extrapolate = True
        self.extrapolate = bool(extrapolate)

        if self.c.ndim < 2:
            raise ValueError("Coefficients array must be at least "
                             "2-dimensional.")

        if not (0 <= axis < self.c.ndim - 1):
            raise ValueError("axis=%s must be between 0 and %s" %
                             (axis, self.c.ndim-1))

        self.axis = axis
        if axis != 0:
            self.c = cp.rollaxis(self.c, axis+1)
            self.c = cp.rollaxis(self.c, axis+1)

        if self.x.ndim != 1:
            raise ValueError("x must be 1-dimensional")
        if self.x.size < 2:
            raise ValueError("at least 2 breakpoints are needed")
        if self.c.ndim < 2:
            raise ValueError("c must have at least 2 dimensions")
        if self.c.shape[0] == 0:
            raise ValueError("polynomial must be at least of order 0")
        if self.c.shape[1] != self.x.size-1:
            raise ValueError("number of coefficients != len(x)-1")
        dx = cp.diff(self.x)
        if not (cp.all(dx >= 0) or cp.all(dx <= 0)):
            raise ValueError("`x` must be strictly increasing or decreasing.")

        dtype = self._get_dtype(self.c.dtype)
        self.c = cp.ascontiguousarray(self.c, dtype=dtype)

    def __call__(self, x, nu=0, extrapolate=None):
        if extrapolate is None:
            extrapolate = self.extrapolate
        x = cp.asarray(x)
        x_shape, x_ndim = x.shape, x.ndim
        x = cp.ascontiguousarray(x.ravel(), dtype=cp.float_)
        out = cp.empty((len(x), prod(self.c.shape[2:])), dtype=self.c.dtype)
        self._ensure_c_contiguous()
        self._evaluate(x, nu, extrapolate, out)
        out = out.reshape(x_shape + self.c.shape[2:])
        if self.axis != 0:
            # transpose to move the calculated values to the interpolation axis
            l = list(range(out.ndim))
            l = l[x_ndim:x_ndim+self.axis] + l[:x_ndim] + l[x_ndim+self.axis:]
            out = out.transpose(l)
        return out

    def _ensure_c_contiguous(self):
        if not self.x.flags.c_contiguous:
            self.x = self.x.copy()
        if not self.c.flags.c_contiguous:
            self.c = self.c.copy()

    def _evaluate(self, x, nu, extrapolate, out):
        evaluate[2048,256](self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
                        self.x, x, out)
        cuda.synchronize()
        #threads, cores
        #1,2,3,4
        #256 cores, each core run 1024 threads
        #256 * 1024 functions running at the same time

    def _get_dtype(self, dtype):
        if cp.issubdtype(dtype, cp.complexfloating) \
               or cp.issubdtype(self.c.dtype, cp.complexfloating):
            return cp.complex_
        else:
            return cp.float_

class CubicHermiteSpline(PPoly):
    def __init__(self, x, y, dydx, axis=0):
        x, dx, y, axis, dydx = prepare_input(x, y, axis, dydx)
        dxr = dx.reshape([dx.shape[0]] + [1] * (y.ndim - 1))
        slope = cp.diff(y, axis=0) / dxr
        t = (dydx[:-1] + dydx[1:] - 2 * slope) / dxr
        c = cp.empty((4, len(x) - 1) + y.shape[1:], dtype=t.dtype)
        c[0] = t / dxr
        c[1] = (slope - dydx[:-1]) / dxr - t
        c[2] = dydx[:-1]
        c[3] = y[:-1]
        super(CubicHermiteSpline, self).__init__(c, x,)
        self.axis = axis

class CpCubicSpline(CubicHermiteSpline):
    def __init__(self, x, y, axis=0):
        x, dx, y, axis, _ = prepare_input(x, y, axis)
        n = len(x)
        dxr = dx.reshape([dx.shape[0]] + [1] * (y.ndim - 1))
        slope = cp.diff(y, axis=0) / dxr

        # Find derivative values at each x[i] by solving a tridiagonal system.
        A = cp.zeros((3, n))  # This is a banded matrix representation.
        b = cp.empty((n,) + y.shape[1:], dtype=y.dtype)

        A[1, 1:-1] = 2 * (dx[:-1] + dx[1:])  # The diagonal
        A[0, 2:] = dx[:-1]                   # The upper diagonal
        A[-1, :-2] = dx[1:]                  # The lower diagonal

        b[1:-1] = 3 * (dxr[1:] * slope[:-1] + dxr[:-1] * slope[1:])

        A[1, 0] = dx[1]
        A[0, 1] = x[2] - x[0]
        d = x[2] - x[0]
        b[0] = ((dxr[0] + 2*d) * dxr[1] * slope[0] + dxr[0]**2 * slope[1]) / d

        A[1, -1] = dx[-2]
        A[-1, -2] = x[-1] - x[-3]
        d = x[-1] - x[-3]
        b[-1] = ((dxr[-1]**2*slope[-2] + (2*d + dxr[-1])*dxr[-2]*slope[-1]) / d)

        s = solve_banded((1, 1), A, b)

        super(CpCubicSpline, self).__init__(x, y, s, axis=0)
        self.axis = axis