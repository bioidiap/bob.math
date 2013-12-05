/**
 * @file math/python/pavx.cc
 * @date Sat Dec 8 20:53:50 2012 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds the Pool-Adjacent-Violators Algorithm
 */

#include "pavx.h"
#include <xbob.blitz/cppapi.h>
#include "bob/math/pavx.h"
#include "bob/core/cast.h"

PyObject* py_pavx (PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = { "input", "output", 0 /* Sentinel */ };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* input = 0;
  PyBlitzArrayObject* output = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O&",
        kwlist, 
        &PyBlitzArray_Converter, &input, 
        &PyBlitzArray_OutputConverter, &output
        )) return 0;

  // can only handle 1D arrays
  if (input->ndim != 1 || (output && output->ndim != 1)) {
    PyErr_SetString(PyExc_TypeError, "input and output arrays should be one-dimensional");
    Py_DECREF(input);
    Py_XDECREF(output);
    return 0;
  }

  // can only handle float arrays
  if (input->type_num != NPY_FLOAT64 || (output && output->type_num != NPY_FLOAT64)) {
    PyErr_SetString(PyExc_TypeError, "input and output arrays data types should be float (i.e. `numpy.float64' equivalents)");
    Py_DECREF(input);
    Py_XDECREF(output);
    return 0;
  }

  // if output was not provided by user
  bool returns_output = false;
  if (!output) {
    output = (PyBlitzArrayObject*)PyBlitzArray_SimpleNew(NPY_FLOAT64, input->ndim, input->shape);
    if (!output) {
      Py_DECREF(input);
      return 0;
    }
    returns_output = true;
  }

  try {
    bob::math::pavx(*PyBlitzArrayCxx_AsBlitz<double,1>(input),
        *PyBlitzArrayCxx_AsBlitz<double,1>(output));
  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
  }
  catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "pavx failed: unknown exception caught");
  }

  Py_DECREF(input);
  if (returns_output) return (PyObject*)output;
  Py_DECREF(output);
  Py_RETURN_NONE;

}

PyObject* py_pavx_nocheck (PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = { "input", "output", 0 /* Sentinel */ };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* input = 0;
  PyBlitzArrayObject* output = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&",
        kwlist, 
        &PyBlitzArray_Converter, &input, 
        &PyBlitzArray_OutputConverter, &output
        )) return 0;

  // can only handle 1D arrays
  if (input->ndim != 1 || output->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "input and output arrays should be one-dimensional");
    Py_DECREF(input);
    Py_DECREF(output);
    return 0;
  }

  // can only handle float arrays
  if (input->type_num != NPY_FLOAT64 || output->type_num != NPY_FLOAT64) {
    PyErr_SetString(PyExc_TypeError, "input and output arrays data types should be float (i.e. `numpy.float64' equivalents)");
    Py_DECREF(input);
    Py_DECREF(output);
    return 0;
  }

  try {
    bob::math::pavx_(*PyBlitzArrayCxx_AsBlitz<double,1>(input),
        *PyBlitzArrayCxx_AsBlitz<double,1>(output));
  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
  }
  catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "pavx failed: unknown exception caught");
  }

  Py_DECREF(input);
  Py_DECREF(output);
  Py_RETURN_NONE;
}

PyObject* py_pavx_width (PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = { "input", "output", 0 /* Sentinel */ };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* input = 0;
  PyBlitzArrayObject* output = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&",
        kwlist,
        &PyBlitzArray_Converter, &input,
        &PyBlitzArray_OutputConverter, &output
        )) return 0;

  // can only handle 1D arrays
  if (input->ndim != 1 || output->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "input and output arrays should be one-dimensional");
    Py_DECREF(input);
    Py_DECREF(output);
    return 0;
  }

  // can only handle float arrays
  if (input->type_num != NPY_FLOAT64 || output->type_num != NPY_FLOAT64) {
    PyErr_SetString(PyExc_TypeError, "input and output arrays data types should be float (i.e. `numpy.float64' equivalents)");
    Py_DECREF(input);
    Py_DECREF(output);
    return 0;
  }

  PyObject* retval = 0;

  try {
    blitz::Array<size_t,1> width =
      bob::math::pavxWidth(*PyBlitzArrayCxx_AsBlitz<double,1>(input),
          *PyBlitzArrayCxx_AsBlitz<double,1>(output));
    blitz::Array<uint64_t,1> uwidth = bob::core::array::cast<uint64_t>(width);
    retval = PyBlitzArrayCxx_NewFromArray(uwidth);
  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
  }
  catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "pavx failed: unknown exception caught");
  }

  Py_DECREF(input);
  Py_DECREF(output);
  return retval;
}

PyObject* py_pavx_width_height (PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = { "input", "output", 0 /* Sentinel */ };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* input = 0;
  PyBlitzArrayObject* output = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&",
        kwlist, 
        &PyBlitzArray_Converter, &input, 
        &PyBlitzArray_OutputConverter, &output
        )) return 0;

  // can only handle 1D arrays
  if (input->ndim != 1 || output->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "input and output arrays should be one-dimensional");
    Py_DECREF(input);
    Py_DECREF(output);
    return 0;
  }

  // can only handle float arrays
  if (input->type_num != NPY_FLOAT64 || output->type_num != NPY_FLOAT64) {
    PyErr_SetString(PyExc_TypeError, "input and output arrays data types should be float (i.e. `numpy.float64' equivalents)");
    Py_DECREF(input);
    Py_DECREF(output);
    return 0;
  }

  PyObject* width = 0;
  PyObject* height = 0;

  try {
    std::pair<blitz::Array<size_t,1>,blitz::Array<double,1>> width_height =
      bob::math::pavxWidthHeight(*PyBlitzArrayCxx_AsBlitz<double,1>(input),
          *PyBlitzArrayCxx_AsBlitz<double,1>(output));
    blitz::Array<uint64_t,1> uwidth = bob::core::array::cast<uint64_t>(width_height.first);
    width = PyBlitzArrayCxx_NewFromArray(uwidth);
    if (width) {
      height = PyBlitzArrayCxx_NewFromArray(width_height.second);
      if (!height) Py_CLEAR(width);
    }
  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
  }
  catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "pavx failed: unknown exception caught");
  }

  Py_DECREF(input);
  Py_DECREF(output);

  if (!height) return 0;

  // creates the return pair and returns
  PyObject* retval = PyTuple_New(2);
  PyTuple_SET_ITEM(retval, 0, width);
  PyTuple_SET_ITEM(retval, 1, height);
  return retval;
}
