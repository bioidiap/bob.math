/**
 * @date Tue Jan 10 21:14 2016 +0100
 * @author Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
 *
 * @brief Binds the Generalized SVD
 */

#include "gsvd.h"
#include <bob.blitz/cppapi.h>
#include <bob.blitz/cleanup.h>
#include <bob.math/gsvd.h>
#include <bob.math/svd.h>
#include <bob.math/linear.h>

PyObject* py_gsvd (PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = { "A", "B", 0 /* Sentinel */ };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* A = 0;
  PyBlitzArrayObject* B = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&", kwlist, 
                                   &PyBlitzArray_Converter, &A,
                                   &PyBlitzArray_Converter, &B
      ))
      return 0;

   auto A_ = make_safe(A);
   auto B_ = make_safe(B);

  if (A->ndim != 2 || A->type_num != NPY_FLOAT64) {
    PyErr_Format(PyExc_TypeError, "`A` matrix only supports 2D 64-bit float array");
    return 0;
  }

  if (B->ndim != 2 || B->type_num != NPY_FLOAT64) {
    PyErr_Format(PyExc_TypeError, "`B` matrix only supports 2D 64-bit float array");
    return 0;
  }

  auto A_bz = PyBlitzArrayCxx_AsBlitz<double,2>(A);
  auto B_bz = PyBlitzArrayCxx_AsBlitz<double,2>(B);

  const int M = A_bz->extent(0);
  const int N = A_bz->extent(1);
  const int P = B_bz->extent(0);

  // Creating the output matrices
  blitz::Array<double,2> U(M, M); U=0;
  blitz::Array<double,2> V(P, P); V=0;
  blitz::Array<double,2> Q(N, N); Q=0;
  blitz::Array<double,2> zeroR(N, N); zeroR=0;
  blitz::Array<double,2> X(N, N); X=0;
  blitz::Array<double,2> C(N, N); C=0;
  blitz::Array<double,2> S(N, N); S=0;
  

  try {
    bob::math::gsvd(*A_bz,*B_bz,U,V,zeroR,Q,X,C,S);
    return Py_BuildValue("NNNNN",
                         PyBlitzArrayCxx_AsConstNumpy(U),
                         PyBlitzArrayCxx_AsConstNumpy(V),
                         PyBlitzArrayCxx_AsConstNumpy(X),
                         PyBlitzArrayCxx_AsConstNumpy(C),
                         PyBlitzArrayCxx_AsConstNumpy(S));
  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
  }
  catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "gsvd failed: unknown exception caught");
  }


  return 0;

}


PyObject* py_svd (PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = { "A",  0 /* Sentinel */ };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* A = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&", kwlist, 
                                   &PyBlitzArray_Converter, &A
      ))
      return 0;

   auto A_ = make_safe(A);

  if (A->ndim != 2 || A->type_num != NPY_FLOAT64) {
    PyErr_Format(PyExc_TypeError, "`A` matrix only supports 2D 64-bit float array");
    return 0;
  }


  auto A_bz = PyBlitzArrayCxx_AsBlitz<double,2>(A);

  int M = A_bz->extent(0);
  int N = A_bz->extent(1); 

  // Creating the output matrices
  // Prepares to call LAPACK function:
  // We recall that FORTRAN/LAPACK is column-major order whereas blitz arrays
  // are row-major order by default.
  // If A = U.S.V^T, then A^T = V.S.U^T

  blitz::Array<double,2> V(M, M); V=0;
  blitz::Array<double,1> S(std::min(M,N)); S=0;  
  blitz::Array<double,2> U(N, N); U=0;   
  

  try {
    bob::math::svd(*A_bz,V,S,U, true);

    // S for the python output.
    // LAPACK returns an 1d matrix of size n.
    // In order to nicelly fit A = U S V,
    // I need to create a matrix S_output of size MxN and fits the diagonal of S (NxN) on it.    
    blitz::Array<double,2> S_output(M, N); S_output=0;
    blitz::Array<double,2> S_diag(N, N); S_diag=0;
    bob::math::diag(S, S_diag);
    S_output(blitz::Range(0,N-1), blitz::Range(0,N-1)) = S_diag;
    //bob::math::diag(S, (reinterpret_cast<blitz::Array<double,2>>)S_output(blitz::Range(0,N-1), blitz::Range(0,N-1)));    
    return Py_BuildValue("NNN",
                         PyBlitzArrayCxx_AsConstNumpy(U),
                         PyBlitzArrayCxx_AsConstNumpy(S_output),
                         PyBlitzArrayCxx_AsConstNumpy(V));
  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
  }
  catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "svd failed: unknown exception caught");
  }


  return 0;

}



