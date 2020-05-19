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

include '/usr/local/lib/python3.6/dist-packages/qutip/cy/complex_math.pxi'

cdef class CompiledStrCoeff(StrCoeff):
    cdef double n
    cdef double n_LD
    cdef double w0
    cdef double wz
    cdef double Omega

    def set_args(self, args):
        self.n=args['n']
        self.n_LD=args['n_LD']
        self.w0=args['w0']
        self.wz=args['wz']
        self.Omega=args['Omega']

    cdef void _call_core(self, double t, complex * coeff):
        cdef double n = self.n
        cdef double n_LD = self.n_LD
        cdef double w0 = self.w0
        cdef double wz = self.wz
        cdef double Omega = self.Omega

        coeff[0] = 1*0.5*w0
        coeff[1] = 1*wz
        coeff[2] = 1*0.5*Omega
