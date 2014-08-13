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

static PyObject* py_linsolve_1(PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = { "A", "x", "b", 0 /* Sentinel */ };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* A = 0;
  PyBlitzArrayObject* x = 0;
  PyBlitzArrayObject* b = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&O&", kwlist,
        &PyBlitzArray_Converter, &A,
        &PyBlitzArray_OutputConverter, &x,
        &PyBlitzArray_Converter, &b
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

  try {

    switch(b->ndim) {
      case 1:
        bob::math::linsolve(
            *PyBlitzArrayCxx_AsBlitz<double,2>(A),
            *PyBlitzArrayCxx_AsBlitz<double,1>(x),
            *PyBlitzArrayCxx_AsBlitz<double,1>(b)
            );
        break;

      case 2:
        bob::math::linsolve(
            *PyBlitzArrayCxx_AsBlitz<double,2>(A),
            *PyBlitzArrayCxx_AsBlitz<double,2>(x),
            *PyBlitzArrayCxx_AsBlitz<double,2>(b)
            );
        break;

      default:
        PyErr_Format(PyExc_TypeError, "linear solver can only work with 1D or 2D problems, but your b array has %" PY_FORMAT_SIZE_T "d dimensions", b->ndim);
        return 0;
    }

  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return 0;
  }
  catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "linsolve failed: unknown exception caught");
    return 0;
  }

  Py_RETURN_NONE;

}

static PyObject* py_linsolve_2(PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = { "A", "b", 0 /* Sentinel */ };
  static char** kwlist = const_cast<char**>(const_kwlist);

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

  PyObject* retval = 0;

  try {

    switch(b->ndim) {
      case 1:
        retval = PyBlitzArray_SimpleNew(NPY_FLOAT64, b->ndim, b->shape);
        if (!retval) return 0;
        bob::math::linsolve(
            *PyBlitzArrayCxx_AsBlitz<double,2>(A),
            *PyBlitzArrayCxx_AsBlitz<double,1>((PyBlitzArrayObject*)retval),
            *PyBlitzArrayCxx_AsBlitz<double,1>(b)
            );
        break;

      case 2:
        retval = PyBlitzArray_SimpleNew(NPY_FLOAT64, b->ndim, b->shape);
        if (!retval) return 0;
        bob::math::linsolve(
            *PyBlitzArrayCxx_AsBlitz<double,2>(A),
            *PyBlitzArrayCxx_AsBlitz<double,2>((PyBlitzArrayObject*)retval),
            *PyBlitzArrayCxx_AsBlitz<double,2>(b)
            );
        break;

      default:
        PyErr_Format(PyExc_TypeError, "linear solver can only work with 1D or 2D arrays, but your b array has %" PY_FORMAT_SIZE_T "d dimensions", b->ndim);
        return 0;

    }

  }
  catch (std::exception& e) {
    Py_DECREF(retval);
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return 0;
  }
  catch (...) {
    Py_DECREF(retval);
    PyErr_SetString(PyExc_RuntimeError, "linsolve failed: unknown exception caught");
    return 0;
  }

  return PyBlitzArray_NUMPY_WRAP(retval);

}

/**
 * Note: Dispatcher function.
 */
PyObject* py_linsolve(PyObject*, PyObject* args, PyObject* kwargs) {

  Py_ssize_t nargs = args?PyTuple_Size(args):0 + kwargs?PyDict_Size(kwargs):0;

  PyObject* retval = 0;

  switch (nargs) {

    case 3:
      retval = py_linsolve_1(0, args, kwargs);
      break;

    case 2:
      retval = py_linsolve_2(0, args, kwargs);
      break;

    default:

      PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - linsolve requires 2 or 3 arguments, but you provided %" PY_FORMAT_SIZE_T "d (see help)", nargs);

  }

  return retval;

}

PyObject* py_linsolve_nocheck(PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = { "A", "x", "b", 0 /* Sentinel */ };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* A = 0;
  PyBlitzArrayObject* x = 0;
  PyBlitzArrayObject* b = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&O&", kwlist,
        &PyBlitzArray_Converter, &A,
        &PyBlitzArray_OutputConverter, &x,
        &PyBlitzArray_Converter, &b
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

  if (A->ndim != x->ndim || A->ndim != b->ndim) {
    PyErr_Format(PyExc_TypeError, "mismatch between the number of dimensions of A, x and b (respectively %" PY_FORMAT_SIZE_T "d, %" PY_FORMAT_SIZE_T "d and %" PY_FORMAT_SIZE_T "d) - all input must have the same number of dimensions", A->ndim, x->ndim, b->ndim);
    return 0;
  }

  try {

    switch(b->ndim) {
      case 1:
        bob::math::linsolve_(
            *PyBlitzArrayCxx_AsBlitz<double,2>(A),
            *PyBlitzArrayCxx_AsBlitz<double,1>(x),
            *PyBlitzArrayCxx_AsBlitz<double,1>(b)
            );
        break;

      case 2:
        bob::math::linsolve_(
            *PyBlitzArrayCxx_AsBlitz<double,2>(A),
            *PyBlitzArrayCxx_AsBlitz<double,2>(x),
            *PyBlitzArrayCxx_AsBlitz<double,2>(b)
            );
        break;

      default:
        PyErr_Format(PyExc_TypeError, "linear solver can only work with 1D or 2D arrays, but your b array has %" PY_FORMAT_SIZE_T "d dimensions", b->ndim);
        return 0;
    }

  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return 0;
  }
  catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "linsolve_ failed: unknown exception caught");
    return 0;
  }

  Py_RETURN_NONE;

}

static PyObject* py_linsolve_sympos_1(PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = { "A", "x", "b", 0 /* Sentinel */ };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* A = 0;
  PyBlitzArrayObject* x = 0;
  PyBlitzArrayObject* b = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&O&", kwlist,
        &PyBlitzArray_Converter, &A,
        &PyBlitzArray_OutputConverter, &x,
        &PyBlitzArray_Converter, &b
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

  try {

    switch(b->ndim) {
      case 1:
        bob::math::linsolveSympos(
            *PyBlitzArrayCxx_AsBlitz<double,2>(A),
            *PyBlitzArrayCxx_AsBlitz<double,1>(x),
            *PyBlitzArrayCxx_AsBlitz<double,1>(b)
            );
        break;

      case 2:
        bob::math::linsolveSympos(
            *PyBlitzArrayCxx_AsBlitz<double,2>(A),
            *PyBlitzArrayCxx_AsBlitz<double,2>(x),
            *PyBlitzArrayCxx_AsBlitz<double,2>(b)
            );
        break;

      default:
        PyErr_Format(PyExc_TypeError, "linear solver can only work with 1D or 2D problems, but your b array has %" PY_FORMAT_SIZE_T "d dimensions", b->ndim);
        return 0;
    }

  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return 0;
  }
  catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "linsolve_sympos failed: unknown exception caught");
    return 0;
  }

  Py_RETURN_NONE;

}

static PyObject* py_linsolve_sympos_2(PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = { "A", "b", 0 /* Sentinel */ };
  static char** kwlist = const_cast<char**>(const_kwlist);

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

  PyObject* retval = 0;

  try {

    switch(b->ndim) {
      case 1:
        retval = PyBlitzArray_SimpleNew(NPY_FLOAT64, b->ndim, b->shape);
        if (!retval) return 0;
        bob::math::linsolveSympos(
            *PyBlitzArrayCxx_AsBlitz<double,2>(A),
            *PyBlitzArrayCxx_AsBlitz<double,1>((PyBlitzArrayObject*)retval),
            *PyBlitzArrayCxx_AsBlitz<double,1>(b)
            );
        break;

      case 2:
        retval = PyBlitzArray_SimpleNew(NPY_FLOAT64, b->ndim, b->shape);
        if (!retval) return 0;
        bob::math::linsolveSympos(
            *PyBlitzArrayCxx_AsBlitz<double,2>(A),
            *PyBlitzArrayCxx_AsBlitz<double,2>((PyBlitzArrayObject*)retval),
            *PyBlitzArrayCxx_AsBlitz<double,2>(b)
            );
        break;

      default:
        PyErr_Format(PyExc_TypeError, "linear solver can only work with 1D or 2D arrays, but your b array has %" PY_FORMAT_SIZE_T "d dimensions", b->ndim);

    }

  }
  catch (std::exception& e) {
    Py_DECREF(retval);
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return 0;
  }
  catch (...) {
    Py_DECREF(retval);
    PyErr_SetString(PyExc_RuntimeError, "linsolve_sympos failed: unknown exception caught");
    return 0;
  }

  return PyBlitzArray_NUMPY_WRAP(retval);

}

/**
 * Note: Dispatcher function.
 */
PyObject* py_linsolve_sympos(PyObject*, PyObject* args, PyObject* kwargs) {

  Py_ssize_t nargs = args?PyTuple_Size(args):0 + kwargs?PyDict_Size(kwargs):0;

  PyObject* retval = 0;

  switch (nargs) {

    case 3:
      retval = py_linsolve_sympos_1(0, args, kwargs);
      break;

    case 2:
      retval = py_linsolve_sympos_2(0, args, kwargs);
      break;

    default:

      PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - linsolve_sympos requires 2 or 3 arguments, but you provided %" PY_FORMAT_SIZE_T "d (see help)", nargs);

  }

  return retval;

}

PyObject* py_linsolve_sympos_nocheck(PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = { "A", "x", "b", 0 /* Sentinel */ };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* A = 0;
  PyBlitzArrayObject* x = 0;
  PyBlitzArrayObject* b = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&O&", kwlist,
        &PyBlitzArray_Converter, &A,
        &PyBlitzArray_OutputConverter, &x,
        &PyBlitzArray_Converter, &b
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

  if (A->ndim != x->ndim || A->ndim != b->ndim) {
    PyErr_Format(PyExc_TypeError, "mismatch between the number of dimensions of A, x and b (respectively %" PY_FORMAT_SIZE_T "d, %" PY_FORMAT_SIZE_T "d and %" PY_FORMAT_SIZE_T "d) - all input must have the same number of dimensions", A->ndim, x->ndim, b->ndim);
    return 0;
  }

  try {

    switch(b->ndim) {
      case 1:
        bob::math::linsolveSympos_(
            *PyBlitzArrayCxx_AsBlitz<double,2>(A),
            *PyBlitzArrayCxx_AsBlitz<double,1>(x),
            *PyBlitzArrayCxx_AsBlitz<double,1>(b)
            );
        break;

      case 2:
        bob::math::linsolveSympos_(
            *PyBlitzArrayCxx_AsBlitz<double,2>(A),
            *PyBlitzArrayCxx_AsBlitz<double,2>(x),
            *PyBlitzArrayCxx_AsBlitz<double,2>(b)
            );
        break;

      default:
        PyErr_Format(PyExc_TypeError, "linear solver can only work with 1D or 2D arrays, but your b array has %" PY_FORMAT_SIZE_T "d dimensions", b->ndim);
        return 0;
    }

  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return 0;
  }
  catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "linsolve_sympos_ failed: unknown exception caught");
    return 0;
  }

  Py_RETURN_NONE;

}

static PyObject* py_linsolve_cg_sympos_1(PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {
    "A", "x", "b", "acc", "max_iter",
    0 /* Sentinel */ };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* A = 0;
  PyBlitzArrayObject* x = 0;
  PyBlitzArrayObject* b = 0;
  double acc = 0.;
  int max_iter = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&O&di", kwlist,
        &PyBlitzArray_Converter, &A,
        &PyBlitzArray_OutputConverter, &x,
        &PyBlitzArray_Converter, &b,
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

  try {

    switch(b->ndim) {
      case 1:
        bob::math::linsolveCGSympos(
            *PyBlitzArrayCxx_AsBlitz<double,2>(A),
            *PyBlitzArrayCxx_AsBlitz<double,1>(x),
            *PyBlitzArrayCxx_AsBlitz<double,1>(b),
            acc, max_iter
            );
        break;

      default:
        PyErr_Format(PyExc_TypeError, "linear solver with conjugate gradients can only work with 1D problems, but your b array has %" PY_FORMAT_SIZE_T "d dimensions", b->ndim);
        return 0;
    }

  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return 0;
  }
  catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "linsolve_cg_sympos failed: unknown exception caught");
    return 0;
  }

  Py_RETURN_NONE;

}

static PyObject* py_linsolve_cg_sympos_2(PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = { "A", "b", "acc", "max_iter", 0 /* Sentinel */ };
  static char** kwlist = const_cast<char**>(const_kwlist);

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

  PyObject* retval = 0;

  try {

    switch(b->ndim) {
      case 1:
        retval = PyBlitzArray_SimpleNew(NPY_FLOAT64, b->ndim, b->shape);
        if (!retval) return 0;
        bob::math::linsolveCGSympos(
            *PyBlitzArrayCxx_AsBlitz<double,2>(A),
            *PyBlitzArrayCxx_AsBlitz<double,1>((PyBlitzArrayObject*)retval),
            *PyBlitzArrayCxx_AsBlitz<double,1>(b),
            acc, max_iter
            );
        break;

      default:
        PyErr_Format(PyExc_TypeError, "linear solver can only work with 1D arrays, but your b array has %" PY_FORMAT_SIZE_T "d dimensions", b->ndim);
        return 0;

    }

  }
  catch (std::exception& e) {
    Py_DECREF(retval);
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return 0;
  }
  catch (...) {
    Py_DECREF(retval);
    PyErr_SetString(PyExc_RuntimeError, "linsolve_cg_sympos failed: unknown exception caught");
    return 0;
  }

  return PyBlitzArray_NUMPY_WRAP(retval);

}

/**
 * Note: Dispatcher function.
 */
PyObject* py_linsolve_cg_sympos(PyObject*, PyObject* args, PyObject* kwargs) {

  Py_ssize_t nargs = args?PyTuple_Size(args):0 + kwargs?PyDict_Size(kwargs):0;

  PyObject* retval = 0;

  switch (nargs) {

    case 5:
      retval = py_linsolve_cg_sympos_1(0, args, kwargs);
      break;

    case 4:
      retval = py_linsolve_cg_sympos_2(0, args, kwargs);
      break;

    default:

      PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - linsolve_cg_sympos requires 4 or 5 arguments, but you provided %" PY_FORMAT_SIZE_T "d (see help)", nargs);

  }

  return retval;

}

PyObject* py_linsolve_cg_sympos_nocheck(PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {
    "A", "x", "b", "acc", "max_iter",
    0 /* Sentinel */ };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* A = 0;
  PyBlitzArrayObject* x = 0;
  PyBlitzArrayObject* b = 0;
  double acc = 0.;
  int max_iter = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&O&di", kwlist,
        &PyBlitzArray_Converter, &A,
        &PyBlitzArray_OutputConverter, &x,
        &PyBlitzArray_Converter, &b,
        acc, max_iter
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

  if (A->ndim != x->ndim || A->ndim != b->ndim) {
    PyErr_Format(PyExc_TypeError, "mismatch between the number of dimensions of A, x and b (respectively %" PY_FORMAT_SIZE_T "d, %" PY_FORMAT_SIZE_T "d and %" PY_FORMAT_SIZE_T "d) - all input must have the same number of dimensions", A->ndim, x->ndim, b->ndim);
    return 0;
  }

  try {

    switch(b->ndim) {
      case 1:
        bob::math::linsolveCGSympos_(
            *PyBlitzArrayCxx_AsBlitz<double,2>(A),
            *PyBlitzArrayCxx_AsBlitz<double,1>(x),
            *PyBlitzArrayCxx_AsBlitz<double,1>(b),
            acc, max_iter
            );
        break;

      default:
        PyErr_Format(PyExc_TypeError, "linear solver with conjugate gradients can only work with 1D arrays, but your b array has %" PY_FORMAT_SIZE_T "d dimensions", b->ndim);
        return 0;
    }

  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return 0;
  }
  catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "linsolve_cg_sympos_ failed: unknown exception caught");
    return 0;
  }

  Py_RETURN_NONE;

}
