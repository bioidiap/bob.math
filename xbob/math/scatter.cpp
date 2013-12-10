/**
 * @date Mon Jun 20 11:47:58 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Python bindings to statistical methods
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include "scatter.h"
#include <xbob.blitz/cppapi.h>
#include <bob/math/stats.h>

PyObject* py_scatter (PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = { "a", "s", "m", 0 /* Sentinel */ };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* a = 0;
  PyBlitzArrayObject* s = 0;
  PyBlitzArrayObject* m = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O&O&",
        kwlist,
        &PyBlitzArray_Converter, &a,
        &PyBlitzArray_OutputConverter, &s,
        &PyBlitzArray_OutputConverter, &m
        )) return 0;

  // basic checks
  if (a->ndim != 2 || (a->type_num != NPY_FLOAT32 && a->type_num != NPY_FLOAT64)) {
    PyErr_SetString(PyExc_TypeError, "input data matrix `a' should be either a 32 or 64-bit float 2D array");
    Py_DECREF(a);
    Py_XDECREF(s);
    Py_XDECREF(m);
    return 0;
  }

  if (s && (s->ndim != 2 || (s->type_num != a->type_num))) {
    PyErr_SetString(PyExc_TypeError, "output data matrix `s' should be either a 32 or 64-bit float 2D array, matching the data type of `a'");
    Py_DECREF(a);
    Py_DECREF(s);
    Py_XDECREF(m);
  }

  if (m && (m->ndim != 1 || (m->type_num != a->type_num))) {
    PyErr_SetString(PyExc_TypeError, "output data vector `m' should be either a 32 or 64-bit float 1D array, matching the data type of `a'");
    Py_DECREF(a);
    Py_XDECREF(s);
    Py_DECREF(m);
  }

  // allocates data not passed by the user
  bool user_s = s;
  if (!s) {
    Py_ssize_t sshape[2] = {a->shape[1], a->shape[1]};
    s = (PyBlitzArrayObject*)PyBlitzArray_SimpleNew(a->type_num, 2, sshape);
  }

  bool user_m = m;
  if (!m) m = (PyBlitzArrayObject*)PyBlitzArray_SimpleNew(a->type_num, 1, &a->shape[1]);

  try {
    switch (a->type_num) {
      case NPY_FLOAT32:
        bob::math::scatter(
            *PyBlitzArrayCxx_AsBlitz<float,2>(a),
            *PyBlitzArrayCxx_AsBlitz<float,2>(s),
            *PyBlitzArrayCxx_AsBlitz<float,1>(m)
            );
        break;

      case NPY_FLOAT64:
        bob::math::scatter(
            *PyBlitzArrayCxx_AsBlitz<double,2>(a),
            *PyBlitzArrayCxx_AsBlitz<double,2>(s),
            *PyBlitzArrayCxx_AsBlitz<double,1>(m)
            );
        break;

      default:
        PyErr_Format(PyExc_TypeError, "scatter calculation currently not implemented for type '%s'", PyBlitzArray_TypenumAsString(a->type_num));
    }
  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
  }
  catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "scatter calculation failed: unknown exception caught");
  }

  Py_DECREF(a);
  if (PyErr_Occurred()) {
    Py_DECREF(s);
    Py_DECREF(m);
    return 0;
  }

  int returns = 2 - (user_s + user_m);

  PyObject* retval = PyTuple_New(returns);

  // fill from the back
  if (!user_m) PyTuple_SET_ITEM(retval, --returns, (PyObject*)m);
  else Py_DECREF(m);
  if (!user_s) PyTuple_SET_ITEM(retval, --returns, (PyObject*)s);
  else Py_DECREF(s);

  return retval;

}

PyObject* py_scatter_nocheck (PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = { "a", "s", "m", 0 /* Sentinel */ };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* a = 0;
  PyBlitzArrayObject* s = 0;
  PyBlitzArrayObject* m = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&O&",
        kwlist, 
        &PyBlitzArray_Converter, &a,
        &PyBlitzArray_OutputConverter, &s,
        &PyBlitzArray_OutputConverter, &m
        )) return 0;

  // basic checks
  if (a->ndim != 2 || (a->type_num != NPY_FLOAT32 && a->type_num != NPY_FLOAT64)) {
    PyErr_SetString(PyExc_TypeError, "input data matrix `a' should be either a 32 or 64-bit float 2D array");
    Py_DECREF(a);
    Py_DECREF(s);
    Py_DECREF(m);
    return 0;
  }

  if (s->ndim != 2 || (s->type_num != a->type_num)) {
    PyErr_SetString(PyExc_TypeError, "output data matrix `s' should be either a 32 or 64-bit float 2D array, matching the data type of `a'");
    Py_DECREF(a);
    Py_DECREF(s);
    Py_DECREF(m);
  }

  if (m->ndim != 1 || (m->type_num != a->type_num)) {
    PyErr_SetString(PyExc_TypeError, "output data vector `m' should be either a 32 or 64-bit float 1D array, matching the data type of `a'");
    Py_DECREF(a);
    Py_DECREF(s);
    Py_DECREF(m);
  }

  try {
    switch (a->type_num) {
      case NPY_FLOAT32:
        bob::math::scatter(
            *PyBlitzArrayCxx_AsBlitz<float,2>(a),
            *PyBlitzArrayCxx_AsBlitz<float,2>(s),
            *PyBlitzArrayCxx_AsBlitz<float,1>(m)
            );
        break;

      case NPY_FLOAT64:
        bob::math::scatter(
            *PyBlitzArrayCxx_AsBlitz<double,2>(a),
            *PyBlitzArrayCxx_AsBlitz<double,2>(s),
            *PyBlitzArrayCxx_AsBlitz<double,1>(m)
            );
        break;

      default:
        PyErr_Format(PyExc_TypeError, "(no-check) scatter calculation currently not implemented for type '%s'", PyBlitzArray_TypenumAsString(a->type_num));
    }
  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
  }
  catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "(no-check) scatter calculation failed: unknown exception caught");
  }

  Py_DECREF(a);
  Py_DECREF(s);
  Py_DECREF(m);

  if (PyErr_Occurred()) return 0;

  Py_RETURN_NONE; 

}

/**
 * Converts the input iterable d into a tuple of PyBlitzArrayObject's. Checks
 * each array is 2D and of type NPY_FLOAT32 or NPY_FLOAT64, consistently.
 * Returns 0 if a problem occurs, 1 otherwise.
 */
int BzTuple_Converter(PyObject* o, PyObject** a) {

  PyObject* tmp = PySequence_Tuple(o);
  if (!tmp) return 0;

  if (PyTuple_GET_SIZE(o) < 2) {
    PyErr_SetString(PyExc_TypeError, "input data object must be a sequence or iterable with at least 2 2D arrays with 32 or 64-bit floats");
    Py_DECREF(tmp);
    return 0;
  }

  PyBlitzArrayObject* first = 0;
  int status = PyBlitzArray_Converter(PyTuple_GET_ITEM(tmp, 0), &first);
  if (!status) {
    Py_DECREF(tmp);
    return 0;
  }

  if (first->ndim != 2 || 
      (first->type_num != NPY_FLOAT32 && first->type_num != NPY_FLOAT64)) {
    PyErr_SetString(PyExc_TypeError, "input data object must be a sequence or iterable with at least 2 2D arrays with 32 or 64-bit floats - the first array does not conform");
    Py_DECREF(first);
    Py_DECREF(tmp);
  }

  PyObject* retval = PyTuple_New(PyTuple_GET_SIZE(tmp));
  if (!retval) {
    Py_DECREF(tmp);
    Py_DECREF(first);
    return 0;
  }

  PyTuple_SET_ITEM(retval, 0, (PyObject*)first);

  for (Py_ssize_t i=1; i<PyTuple_GET_SIZE(tmp); ++i) {

    PyBlitzArrayObject* next = 0;
    PyObject* item = PyTuple_GET_ITEM(tmp, i); //borrowed
    int status = PyBlitzArray_Converter(item, &next);
    if (!status) {
      Py_DECREF(retval);
      Py_DECREF(tmp);
      return 0;
    }
    if (next->type_num != first->type_num) {
        PyErr_Format(PyExc_TypeError, "array at data[%" PY_FORMAT_SIZE_T "d] does not have the same data type as the first array on the sequence (%s != %s)", i, PyBlitzArray_TypenumAsString(next->type_num), PyBlitzArray_TypenumAsString(first->type_num));
        Py_DECREF(next);
        Py_DECREF(retval);
        Py_DECREF(tmp);
        return 0;
    }
    if (next->ndim != 2) {
        PyErr_Format(PyExc_TypeError, "array at data[%" PY_FORMAT_SIZE_T "d] does not have two dimensions, but %" PY_FORMAT_SIZE_T "d", i, next->ndim);
        Py_DECREF(next);
        Py_DECREF(retval);
        Py_DECREF(tmp);
        return 0;
    }

    PyTuple_SET_ITEM(retval, i, (PyObject*)next); //steals `next'

  }

  Py_DECREF(tmp);
  *a = retval;

  return 1;

}

PyObject* py_scatters (PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = { "data", "sw", "sb", "m", 0 /* Sentinel */ };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* data = 0;
  PyBlitzArrayObject* sw = 0;
  PyBlitzArrayObject* sb = 0;
  PyBlitzArrayObject* m = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O&O&O&", kwlist, 
        &BzTuple_Converter, &data,
        &PyBlitzArray_OutputConverter, &sw,
        &PyBlitzArray_OutputConverter, &sb,
        &PyBlitzArray_OutputConverter, &m
        )) return 0;

  PyBlitzArrayObject* first = (PyBlitzArrayObject*)PyTuple_GET_ITEM(data, 0);

  if (sw && (sw->ndim != 2 || (sw->type_num != first->type_num))) {
    PyErr_SetString(PyExc_TypeError, "output data matrix `sw' should be either a 32 or 64-bit float 2D array, matching the data type of `data'");
    Py_DECREF(data);
    Py_DECREF(sw);
    Py_XDECREF(sb);
    Py_XDECREF(m);
  }

  if (sb && (sb->ndim != 2 || (sb->type_num != first->type_num))) {
    PyErr_SetString(PyExc_TypeError, "output data matrix `sb' should be either a 32 or 64-bit float 2D array, matching the data type of `data'");
    Py_DECREF(data);
    Py_XDECREF(sw);
    Py_DECREF(sb);
    Py_XDECREF(m);
  }

  if (m && (m->ndim != 1 || (m->type_num != first->type_num))) {
    PyErr_SetString(PyExc_TypeError, "output data vector `m' should be either a 32 or 64-bit float 1D array, matching the data type of `data'");
    Py_DECREF(data);
    Py_XDECREF(sw);
    Py_XDECREF(sb);
    Py_DECREF(m);
  }

  // allocates data not passed by the user
  bool user_sw = sw;
  if (!sw) {
    Py_ssize_t sshape[2] = {first->shape[1], first->shape[1]};
    sw = (PyBlitzArrayObject*)PyBlitzArray_SimpleNew(first->type_num, 2, sshape);
  }

  bool user_sb = sb;
  if (!sb) {
    Py_ssize_t sshape[2] = {first->shape[1], first->shape[1]};
    sb = (PyBlitzArrayObject*)PyBlitzArray_SimpleNew(first->type_num, 2, sshape);
  }

  bool user_m = m;
  if (!m) m = (PyBlitzArrayObject*)PyBlitzArray_SimpleNew(first->type_num, 1, &first->shape[1]);

  try {
    switch (first->type_num) {
      case NPY_FLOAT32:
        {
          std::vector<blitz::Array<float,2>> cxxdata;
          for (Py_ssize_t i=0; i<PyTuple_GET_SIZE(data); ++i) {
            cxxdata.push_back(*PyBlitzArrayCxx_AsBlitz<float,2>
                ((PyBlitzArrayObject*)PyTuple_GET_ITEM(data,i)));
            bob::math::scatters(cxxdata,
                *PyBlitzArrayCxx_AsBlitz<float,2>(sw),
                *PyBlitzArrayCxx_AsBlitz<float,2>(sb),
                *PyBlitzArrayCxx_AsBlitz<float,1>(m)
                );
          }
        }
        break;

      case NPY_FLOAT64:
        {
          std::vector<blitz::Array<double,2>> cxxdata;
          for (Py_ssize_t i=0; i<PyTuple_GET_SIZE(data); ++i) {
            cxxdata.push_back(*PyBlitzArrayCxx_AsBlitz<double,2>
                ((PyBlitzArrayObject*)PyTuple_GET_ITEM(data,i)));
            bob::math::scatters(cxxdata,
                *PyBlitzArrayCxx_AsBlitz<double,2>(sw),
                *PyBlitzArrayCxx_AsBlitz<double,2>(sb),
                *PyBlitzArrayCxx_AsBlitz<double,1>(m)
                );
          }
        }
        break;

      default:
        PyErr_Format(PyExc_TypeError, "scatters calculation currently not implemented for type '%s'", PyBlitzArray_TypenumAsString(first->type_num));
    }
  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
  }
  catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "scatters calculation failed: unknown exception caught");
  }

  Py_DECREF(data);
  if (PyErr_Occurred()) {
    Py_DECREF(sw);
    Py_DECREF(sb);
    Py_DECREF(m);
    return 0;
  }

  int returns = 3 - (user_sw + user_sb + user_m);

  PyObject* retval = PyTuple_New(returns);

  // fill from the back
  if (!user_m) PyTuple_SET_ITEM(retval, --returns, (PyObject*)m);
  else Py_DECREF(m);
  if (!user_sb) PyTuple_SET_ITEM(retval, --returns, (PyObject*)sb);
  else Py_DECREF(sb);
  if (!user_sw) PyTuple_SET_ITEM(retval, --returns, (PyObject*)sw);
  else Py_DECREF(sw);

  return retval;

}

PyObject* py_scatters_nocheck (PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = { "data", "sw", "sb", "m", 0 /* Sentinel */ };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* data = 0;
  PyBlitzArrayObject* sw = 0;
  PyBlitzArrayObject* sb = 0;
  PyBlitzArrayObject* m = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&O&O&", kwlist, 
        &BzTuple_Converter, &data,
        &PyBlitzArray_OutputConverter, &sw,
        &PyBlitzArray_OutputConverter, &sb,
        &PyBlitzArray_OutputConverter, &m
        )) return 0;

  PyBlitzArrayObject* first = (PyBlitzArrayObject*)PyTuple_GET_ITEM(data, 0);

  if (sw->ndim != 2 || (sw->type_num != first->type_num)) {
    PyErr_SetString(PyExc_TypeError, "output data matrix `sw' should be either a 32 or 64-bit float 2D array, matching the data type of `data'");
    Py_DECREF(data);
    Py_DECREF(sw);
    Py_DECREF(sb);
    Py_DECREF(m);
  }

  if (sb->ndim != 2 || (sb->type_num != first->type_num)) {
    PyErr_SetString(PyExc_TypeError, "output data matrix `sb' should be either a 32 or 64-bit float 2D array, matching the data type of `data'");
    Py_DECREF(data);
    Py_DECREF(sw);
    Py_DECREF(sb);
    Py_DECREF(m);
  }

  if (m->ndim != 1 || (m->type_num != first->type_num)) {
    PyErr_SetString(PyExc_TypeError, "output data vector `m' should be either a 32 or 64-bit float 1D array, matching the data type of `data'");
    Py_DECREF(data);
    Py_DECREF(sw);
    Py_DECREF(sb);
    Py_DECREF(m);
  }

  try {
    switch (first->type_num) {
      case NPY_FLOAT32:
        {
          std::vector<blitz::Array<float,2>> cxxdata;
          for (Py_ssize_t i=0; i<PyTuple_GET_SIZE(data); ++i) {
            cxxdata.push_back(*PyBlitzArrayCxx_AsBlitz<float,2>
                ((PyBlitzArrayObject*)PyTuple_GET_ITEM(data,i)));
            bob::math::scatters_(cxxdata,
                *PyBlitzArrayCxx_AsBlitz<float,2>(sw),
                *PyBlitzArrayCxx_AsBlitz<float,2>(sb),
                *PyBlitzArrayCxx_AsBlitz<float,1>(m)
                );
          }
        }
        break;

      case NPY_FLOAT64:
        {
          std::vector<blitz::Array<double,2>> cxxdata;
          for (Py_ssize_t i=0; i<PyTuple_GET_SIZE(data); ++i) {
            cxxdata.push_back(*PyBlitzArrayCxx_AsBlitz<double,2>
                ((PyBlitzArrayObject*)PyTuple_GET_ITEM(data,i)));
            bob::math::scatters_(cxxdata,
                *PyBlitzArrayCxx_AsBlitz<double,2>(sw),
                *PyBlitzArrayCxx_AsBlitz<double,2>(sb),
                *PyBlitzArrayCxx_AsBlitz<double,1>(m)
                );
          }
        }
        break;

      default:
        PyErr_Format(PyExc_TypeError, "(no-check) scatters calculation currently not implemented for type '%s'", PyBlitzArray_TypenumAsString(first->type_num));
    }
  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
  }
  catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "(no-check) scatters calculation failed: unknown exception caught");
  }

  Py_DECREF(data);
  Py_DECREF(sw);
  Py_DECREF(sb);
  Py_DECREF(m);

  if (PyErr_Occurred()) return 0;

  Py_RETURN_NONE;

}