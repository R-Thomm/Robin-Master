#!python
#cython: language_level=3
# This file is generated automatically by QuTiP.

import numpy as np
cimport numpy as np
import scipy.special as spe
cimport cython
np.import_array()
cdef extern from "numpy/arrayobject.h" nogil:
    void PyDataMem_NEW_ZEROED(size_t size, size_t elsize)
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)
from qutip.cy.spmatfuncs cimport spmvpy
from qutip.cy.inter cimport _spline_complex_t_second, _spline_complex_cte_second
from qutip.cy.inter cimport _spline_float_t_second, _spline_float_cte_second
from qutip.cy.inter cimport _step_float_cte, _step_complex_cte
from qutip.cy.inter cimport _step_float_t, _step_complex_t
from qutip.cy.interpolate cimport (interp, zinterp)
from qutip.cy.cqobjevo_factor cimport StrCoeff
from qutip.cy.cqobjevo cimport CQobjEvo
from qutip.cy.math cimport erf, zerf
from qutip.qobj import Qobj
cdef double pi = 3.14159265358979323

include 'C:/Users/Robin/AppData/Local/Programs/Python/Python37-32/lib/site-packages/qutip-4.5.0-py3.7-win32.egg/qutip/cy/complex_math.pxi'

cdef class CompiledStrCoeff(StrCoeff):
    cdef double n
    cdef double w0
    cdef double dwQ
    cdef double dtQ
    cdef double dwP
    cdef double dtP
    cdef double delay
    cdef double f0

    def set_args(self, args):
        self.n=args['n']
        self.w0=args['w0']
        self.dwQ=args['dwQ']
        self.dtQ=args['dtQ']
        self.dwP=args['dwP']
        self.dtP=args['dtP']
        self.delay=args['delay']
        self.f0=args['f0']

    cdef void _call_core(self, double t, complex * coeff):
        cdef double n = self.n
        cdef double w0 = self.w0
        cdef double dwQ = self.dwQ
        cdef double dtQ = self.dtQ
        cdef double dwP = self.dwP
        cdef double dtP = self.dtP
        cdef double delay = self.delay
        cdef double f0 = self.f0

        coeff[0] = w0 + dwQ*exp(-0.5*(t/dtQ)**2) + (dwP*sin(2*w0*(t-delay)) if t > delay and t < delay+dtP else 0)
        coeff[1] = 1j/4*(- dwQ*exp(-0.5*(t/dtQ)**2) * t/(dtQ**2) + (2*w0*dwP*cos(2*w0*(t-delay)) if t > delay and t < delay+dtP else 0))/(w0 + dwQ*exp(-0.5*(t/dtQ)**2) + (dwP*sin(2*w0*(t-delay)) if t > delay and t < delay+dtP else 0))
        coeff[2] = 9*(f0/((w0 + dwQ*exp(-0.5*(t/dtQ)**2) + (dwP*sin(2*w0*(t-delay)) if t > delay and t < delay+dtP else 0))**2) - f0/(w0**2))
