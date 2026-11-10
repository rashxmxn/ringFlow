"""Optional Numba acceleration.

The numerical kernels are written as plain loops (so they can be read side by
side with the original Fortran).  If Numba is installed they are compiled to
machine code and run at roughly Fortran speed; without Numba they still work,
just much more slowly (fine for small tests, too slow for full runs).
"""

try:  # pragma: no cover - depends on environment
    from numba import njit as _njit

    HAVE_NUMBA = True

    def njit(func):
        # On-disk caching makes the 2nd start fast.  In a frozen .exe the
        # cache goes to NUMBA_CACHE_DIR (set by the GUI); if caching is not
        # possible we simply compile at every start.
        try:
            return _njit(cache=True, nogil=True)(func)
        except Exception:
            return _njit(nogil=True)(func)

except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(func):
        return func
