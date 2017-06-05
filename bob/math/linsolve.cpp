/**
 * @file math/python/linsolve.cc
 * @date Sat Mar 19 19:49:51 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds the Linear System solver based on LAPACK to python.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include "histogram.h"
#include <bob.blitz/cppapi.h>
#include <bob.blitz/cleanup.h>
#include <bob.math/linsolve.h>
#include <bob.extension/documentation.h>


bob::extension::FunctionDoc s_linsolve = bob::extension::FunctionDoc(
  "linsolve",
  "Solves the linear system :math:`Ax=b` and returns the result in :math:`x`.",
  "This method uses LAPACK's ``dgesv`` generic solver. "
  "You can use this method in two different formats. "
  "The first interface accepts the matrices :math:`A` and :math:`b` returning :math:`x`. "
  "The second one accepts a pre-allocated :math:`x` vector and sets it with the linear system solution."
  )
  .add_prototype("A, b", "x")
  .add_prototype("A, b, x")
  .add_parameter("A", "array_like (2D)", "The matrix :math:`A` of the linear system")
  .add_parameter("b", "array_like (1D)", "The vector :math:`b` of the linear system")
  .add_parameter("x", "array_like (1D)", "The result vector :math:`x`, as parameter")
  .add_return("x", "array_like (1D)", "The result vector :math:`x`, as return value")
;

static PyObject* py_linsolve_1(PyObject*, PyObject* args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = s_linsolve.kwlist(1);

  PyBlitzArrayObject* A = 0;
  PyBlitzArrayObject* b = 0;
  PyBlitzArrayObject* x = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&O&", kwlist,
        &PyBlitzArray_Converter, &A,
        &PyBlitzArray_Converter, &b,
        &PyBlitzArray_OutputConverter, &x
        )) return 0;


  //protects acquired resources through this scope
  auto A_ = make_safe(A);
  auto x_ = make_safe(x);
  auto b_ = make_safe(b);

  if (A->type_num != NPY_FLOAT64 ||
      x->type_num != NPY_FLOAT64 ||
      b->type_num != NPY_FLOAT64) {
    PyErr_Format(PyExc_TypeError, "linear solver only supports float type (i.e., `numpy.float64' equivalents) - make sure all your input conforms");
    return 0;
  }

  if (A->ndim != 2) {
    PyErr_Format(PyExc_TypeError, "A matrix should be two-dimensional");
    return 0;
  }

  if (b->ndim != x->ndim) {
    PyErr_Format(PyExc_TypeError, "mismatch between the number of dimensions of x and b (respectively %" PY_FORMAT_SIZE_T "d and %" PY_FORMAT_SIZE_T "d)", x->ndim, b->ndim);
    return 0;
  }

  switch(b->ndim) {
    case 1:
      bob::math::linsolve(
          *PyBlitzArrayCxx_AsBlitz<double,2>(A),
          *PyBlitzArrayCxx_AsBlitz<double,1>(b),
          *PyBlitzArrayCxx_AsBlitz<double,1>(x)
          );
      break;

    case 2:
      bob::math::linsolve(
          *PyBlitzArrayCxx_AsBlitz<double,2>(A),
          *PyBlitzArrayCxx_AsBlitz<double,2>(b),
          *PyBlitzArrayCxx_AsBlitz<double,2>(x)
          );
      break;

    default:
      PyErr_Format(PyExc_TypeError, "linear solver can only work with 1D or 2D problems, but your b array has %" PY_FORMAT_SIZE_T "d dimensions", b->ndim);
      return 0;
  }

  Py_RETURN_NONE;

BOB_CATCH_FUNCTION("linsolve", 0)
}

static PyObject* py_linsolve_2(PyObject*, PyObject* args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = s_linsolve.kwlist(0);

  PyBlitzArrayObject* A = 0;
  PyBlitzArrayObject* b = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&", kwlist,
        &PyBlitzArray_Converter, &A,
        &PyBlitzArray_Converter, &b
        )) return 0;

  //protects acquired resources through this scope
  auto A_ = make_safe(A);
  auto b_ = make_safe(b);

  if (A->type_num != NPY_FLOAT64 || b->type_num != NPY_FLOAT64) {
    PyErr_Format(PyExc_TypeError, "linear solver only supports float type (i.e., `numpy.float64' equivalents) - make sure all your input conforms");
    return 0;
  }

  if (A->ndim != 2) {
    PyErr_Format(PyExc_TypeError, "A matrix should be two-dimensional");
    return 0;
  }

  PyBlitzArrayObject* retval = 0;
  auto retval_ = make_xsafe(retval);

  switch(b->ndim) {
    case 1:
      retval = (PyBlitzArrayObject*)PyBlitzArray_SimpleNew(NPY_FLOAT64, b->ndim, b->shape);
      if (!retval) return 0;
      retval_ = make_safe(retval);
      bob::math::linsolve(
          *PyBlitzArrayCxx_AsBlitz<double,2>(A),
          *PyBlitzArrayCxx_AsBlitz<double,1>(b),
          *PyBlitzArrayCxx_AsBlitz<double,1>(retval)
          );
      break;

    case 2:
      retval = (PyBlitzArrayObject*)PyBlitzArray_SimpleNew(NPY_FLOAT64, b->ndim, b->shape);
      if (!retval) return 0;
      retval_ = make_safe(retval);
      bob::math::linsolve(
          *PyBlitzArrayCxx_AsBlitz<double,2>(A),
          *PyBlitzArrayCxx_AsBlitz<double,2>(b),
          *PyBlitzArrayCxx_AsBlitz<double,2>(retval)
          );
      break;

    default:
      PyErr_Format(PyExc_TypeError, "linear solver can only work with 1D or 2D arrays, but your b array has %" PY_FORMAT_SIZE_T "d dimensions", b->ndim);
      return 0;
  }

  return PyBlitzArray_AsNumpyArray(retval, 0);

BOB_CATCH_FUNCTION("linsolve", 0)
}

/**
 * Note: Dispatcher function.
 */
PyObject* py_linsolve(PyObject*, PyObject* args, PyObject* kwargs) {

  Py_ssize_t nargs = args?PyTuple_Size(args):0 + kwargs?PyDict_Size(kwargs):0;

  switch (nargs) {

    case 3:
      return py_linsolve_1(0, args, kwargs);

    case 2:
      return py_linsolve_2(0, args, kwargs);

    default:

      PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - linsolve requires 2 or 3 arguments, but you provided %" PY_FORMAT_SIZE_T "d (see help)", nargs);
  }

  return 0;
}

bob::extension::FunctionDoc s_linsolve_sympos = bob::extension::FunctionDoc(
  "linsolve_sympos",
  "Solves the linear system :math:`Ax=b` and returns the result in :math:`x` for symmetric :math:`A` matrix.",
  "This method uses LAPACK's ``dposv`` solver, assuming :math:`A` is a symmetric positive definite matrix. "
  "You can use this method in two different formats. "
  "The first interface accepts the matrices :math:`A` and :math:`b` returning :math:`x`. "
  "The second one accepts a pre-allocated :math:`x` vector and sets it with the linear system solution."
  )
  .add_prototype("A, b", "x")
  .add_prototype("A, b, x")
  .add_parameter("A", "array_like (2D)", "The matrix :math:`A` of the linear system")
  .add_parameter("b", "array_like (1D)", "The vector :math:`b` of the linear system")
  .add_parameter("x", "array_like (1D)", "The result vector :math:`x`, as parameter")
  .add_return("x", "array_like (1D)", "The result vector :math:`x`, as return value")
;


static PyObject* py_linsolve_sympos_1(PyObject*, PyObject* args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = s_linsolve_sympos.kwlist(1);

  PyBlitzArrayObject* A = 0;
  PyBlitzArrayObject* b = 0;
  PyBlitzArrayObject* x = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&O&", kwlist,
        &PyBlitzArray_Converter, &A,
        &PyBlitzArray_Converter, &b,
        &PyBlitzArray_OutputConverter, &x
        )) return 0;

  //protects acquired resources through this scope
  auto A_ = make_safe(A);
  auto x_ = make_safe(x);
  auto b_ = make_safe(b);

  if (A->type_num != NPY_FLOAT64 ||
      x->type_num != NPY_FLOAT64 ||
      b->type_num != NPY_FLOAT64) {
    PyErr_Format(PyExc_TypeError, "linear solver only supports float type (i.e., `numpy.float64' equivalents) - make sure all your input conforms");
    return 0;
  }

  if (A->ndim != 2) {
    PyErr_Format(PyExc_TypeError, "A matrix should be two-dimensional");
    return 0;
  }

  if (b->ndim != x->ndim) {
    PyErr_Format(PyExc_TypeError, "mismatch between the number of dimensions of x and b (respectively %" PY_FORMAT_SIZE_T "d and %" PY_FORMAT_SIZE_T "d)", x->ndim, b->ndim);
    return 0;
  }

  switch(b->ndim) {
    case 1:
      bob::math::linsolveSympos(
          *PyBlitzArrayCxx_AsBlitz<double,2>(A),
          *PyBlitzArrayCxx_AsBlitz<double,1>(b),
          *PyBlitzArrayCxx_AsBlitz<double,1>(x)
          );
      break;

    case 2:
      bob::math::linsolveSympos(
          *PyBlitzArrayCxx_AsBlitz<double,2>(A),
          *PyBlitzArrayCxx_AsBlitz<double,2>(b),
          *PyBlitzArrayCxx_AsBlitz<double,2>(x)
          );
      break;

    default:
      PyErr_Format(PyExc_TypeError, "linear solver can only work with 1D or 2D problems, but your b array has %" PY_FORMAT_SIZE_T "d dimensions", b->ndim);
      return 0;
  }

  Py_RETURN_NONE;
BOB_CATCH_FUNCTION("linsolve_sympos", 0)
}

static PyObject* py_linsolve_sympos_2(PyObject*, PyObject* args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = s_linsolve_sympos.kwlist(0);

  PyBlitzArrayObject* A = 0;
  PyBlitzArrayObject* b = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&", kwlist,
        &PyBlitzArray_Converter, &A,
        &PyBlitzArray_Converter, &b
        )) return 0;

  //protects acquired resources through this scope
  auto A_ = make_safe(A);
  auto b_ = make_safe(b);

  if (A->type_num != NPY_FLOAT64 || b->type_num != NPY_FLOAT64) {
    PyErr_Format(PyExc_TypeError, "linear solver only supports float type (i.e., `numpy.float64' equivalents) - make sure all your input conforms");
    return 0;
  }

  if (A->ndim != 2) {
    PyErr_Format(PyExc_TypeError, "A matrix should be two-dimensional");
    return 0;
  }

  PyBlitzArrayObject* retval = 0;
  auto retval_ = make_xsafe(retval);

  switch(b->ndim) {
    case 1:
      retval = (PyBlitzArrayObject*)PyBlitzArray_SimpleNew(NPY_FLOAT64, b->ndim, b->shape);
      if (!retval) return 0;
      retval_ = make_safe(retval);
      bob::math::linsolveSympos(
          *PyBlitzArrayCxx_AsBlitz<double,2>(A),
          *PyBlitzArrayCxx_AsBlitz<double,1>(b),
          *PyBlitzArrayCxx_AsBlitz<double,1>(retval)
          );
      break;

    case 2:
      retval = (PyBlitzArrayObject*)PyBlitzArray_SimpleNew(NPY_FLOAT64, b->ndim, b->shape);
      if (!retval) return 0;
      retval_ = make_safe(retval);
      bob::math::linsolveSympos(
          *PyBlitzArrayCxx_AsBlitz<double,2>(A),
          *PyBlitzArrayCxx_AsBlitz<double,2>(b),
          *PyBlitzArrayCxx_AsBlitz<double,2>(retval)
          );
      break;

    default:
      PyErr_Format(PyExc_TypeError, "linear solver can only work with 1D or 2D arrays, but your b array has %" PY_FORMAT_SIZE_T "d dimensions", b->ndim);
  }

  return PyBlitzArray_AsNumpyArray(retval, 0);
BOB_CATCH_FUNCTION("linsolve_sympos", 0)
}

/**
 * Note: Dispatcher function.
 */
PyObject* py_linsolve_sympos(PyObject*, PyObject* args, PyObject* kwargs) {

  Py_ssize_t nargs = args?PyTuple_Size(args):0 + kwargs?PyDict_Size(kwargs):0;

  switch (nargs) {

    case 3:
      return py_linsolve_sympos_1(0, args, kwargs);

    case 2:
      return py_linsolve_sympos_2(0, args, kwargs);

    default:

      PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - linsolve_sympos requires 2 or 3 arguments, but you provided %" PY_FORMAT_SIZE_T "d (see help)", nargs);
  }

  return 0;
}


bob::extension::FunctionDoc s_linsolve_cg_sympos = bob::extension::FunctionDoc(
  "linsolve_cg_sympos",
  "Solves the linear system :math:`Ax=b` using conjugate gradients and returns the result in :math:`x` for symmetric :math:`A` matrix.",
  "This method uses the conjugate gradient solver, assuming :math:`A` is a symmetric positive definite matrix. "
  "You can use this method in two different formats. "
  "The first interface accepts the matrices :math:`A` and :math:`b` returning :math:`x`. "
  "The second one accepts a pre-allocated :math:`x` vector and sets it with the linear system solution."
  )
  .add_prototype("A, b, [acc], [max_iter]", "x")
  .add_prototype("A, b, x, [acc], [max_iter]")
  .add_parameter("A", "array_like (2D)", "The matrix :math:`A` of the linear system")
  .add_parameter("b", "array_like (1D)", "The vector :math:`b` of the linear system")
  .add_parameter("x", "array_like (1D)", "The result vector :math:`x`, as parameter")
  .add_parameter("acc", "float", "[Default: 0] The desired accuracy. The algorithm terminates when norm(Ax-b)/norm(b) < acc")
  .add_parameter("max_iter", "int", "[Default: 0] The maximum number of iterations")
  .add_return("x", "array_like (1D)", "The result vector :math:`x`, as return value")
;

static PyObject* py_linsolve_cg_sympos_1(PyObject*, PyObject* args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = s_linsolve_cg_sympos.kwlist(1);

  PyBlitzArrayObject* A = 0;
  PyBlitzArrayObject* b = 0;
  PyBlitzArrayObject* x = 0;
  double acc = 0.;
  int max_iter = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&O&di", kwlist,
        &PyBlitzArray_Converter, &A,
        &PyBlitzArray_Converter, &b,
        &PyBlitzArray_OutputConverter, &x,
        &acc, &max_iter
        )) return 0;

  //protects acquired resources through this scope
  auto A_ = make_safe(A);
  auto x_ = make_safe(x);
  auto b_ = make_safe(b);

  if (A->type_num != NPY_FLOAT64 ||
      x->type_num != NPY_FLOAT64 ||
      b->type_num != NPY_FLOAT64) {
    PyErr_Format(PyExc_TypeError, "linear solver only supports float type (i.e., `numpy.float64' equivalents) - make sure all your input conforms");
    return 0;
  }

  if (A->ndim != 2) {
    PyErr_Format(PyExc_TypeError, "A matrix should be two-dimensional");
    return 0;
  }

  if (b->ndim != x->ndim) {
    PyErr_Format(PyExc_TypeError, "mismatch between the number of dimensions of x and b (respectively %" PY_FORMAT_SIZE_T "d and %" PY_FORMAT_SIZE_T "d)", x->ndim, b->ndim);
    return 0;
  }

  switch(b->ndim) {
    case 1:
      bob::math::linsolveCGSympos(
          *PyBlitzArrayCxx_AsBlitz<double,2>(A),
          *PyBlitzArrayCxx_AsBlitz<double,1>(b),
          *PyBlitzArrayCxx_AsBlitz<double,1>(x),
          acc, max_iter
          );
      break;

    default:
      PyErr_Format(PyExc_TypeError, "linear solver with conjugate gradients can only work with 1D problems, but your b array has %" PY_FORMAT_SIZE_T "d dimensions", b->ndim);
      return 0;
  }

  Py_RETURN_NONE;
BOB_CATCH_FUNCTION("linsolve_cg_sympos", 0)
}

static PyObject* py_linsolve_cg_sympos_2(PyObject*, PyObject* args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = s_linsolve_cg_sympos.kwlist(0);

  PyBlitzArrayObject* A = 0;
  PyBlitzArrayObject* b = 0;
  double acc = 0.;
  int max_iter = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&di", kwlist,
        &PyBlitzArray_Converter, &A,
        &PyBlitzArray_Converter, &b,
        &acc, &max_iter
        )) return 0;

  //protects acquired resources through this scope
  auto A_ = make_safe(A);
  auto b_ = make_safe(b);

  if (A->type_num != NPY_FLOAT64 || b->type_num != NPY_FLOAT64) {
    PyErr_Format(PyExc_TypeError, "linear solver only supports float type (i.e., `numpy.float64' equivalents) - make sure all your input conforms");
    return 0;
  }

  if (A->ndim != 2) {
    PyErr_Format(PyExc_TypeError, "A matrix should be two-dimensional");
    return 0;
  }

  PyBlitzArrayObject* retval = 0;
  auto retval_ = make_xsafe(retval);

  switch(b->ndim) {
    case 1:
      retval = (PyBlitzArrayObject*)PyBlitzArray_SimpleNew(NPY_FLOAT64, b->ndim, b->shape);
      if (!retval) return 0;
      retval_ = make_safe(retval);
      bob::math::linsolveCGSympos(
          *PyBlitzArrayCxx_AsBlitz<double,2>(A),
          *PyBlitzArrayCxx_AsBlitz<double,1>(b),
          *PyBlitzArrayCxx_AsBlitz<double,1>(retval),
          acc, max_iter
          );
      break;

    default:
      PyErr_Format(PyExc_TypeError, "linear solver can only work with 1D arrays, but your b array has %" PY_FORMAT_SIZE_T "d dimensions", b->ndim);
      return 0;
  }

  return PyBlitzArray_AsNumpyArray(retval, 0);
BOB_CATCH_FUNCTION("linsolve_cg_sympos", 0)
}

/**
 * Note: Dispatcher function.
 */
PyObject* py_linsolve_cg_sympos(PyObject*, PyObject* args, PyObject* kwargs) {

  Py_ssize_t nargs = args?PyTuple_Size(args):0 + kwargs?PyDict_Size(kwargs):0;

  switch (nargs) {

    case 5:
      return py_linsolve_cg_sympos_1(0, args, kwargs);
      break;

    case 4:
      return py_linsolve_cg_sympos_2(0, args, kwargs);
      break;

    default:

      PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - linsolve_cg_sympos requires 4 or 5 arguments, but you provided %" PY_FORMAT_SIZE_T "d (see help)", nargs);
  }

  return 0;
}
