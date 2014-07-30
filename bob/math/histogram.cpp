/**
 * @author Manuel Guenther <Manuel.Guenther@idiap.ch>
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue  3 Dec 14:23:42 2013 CET
 *
 * @brief Binds fast versions of some histogram measures
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include "histogram.h"
#include <bob.blitz/cppapi.h>
#include <bob.blitz/cleanup.h>
#include <bob.math/histogram.h>

static PyObject* py_histogram_intersection_1
(PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = { "h1", "h2", 0 /* Sentinel */ };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* h1 = 0;
  PyBlitzArrayObject* h2 = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&",
        kwlist, &PyBlitzArray_Converter, &h1, &PyBlitzArray_Converter, &h2)) return 0;

  //protects acquired resources through this scope
  auto h1_ = make_safe(h1);
  auto h2_ = make_safe(h2);

  // checks for type mismatch
  if (h1->type_num != h2->type_num) {
    PyErr_Format(PyExc_TypeError, "data type mismatch between `h1' and `h2' (%s != %s)", PyBlitzArray_TypenumAsString(h1->type_num), PyBlitzArray_TypenumAsString(h2->type_num));
    return 0;
  }

  // input arrays must be 1d
  if (h1->ndim != 1 || h2->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "both `h1' and `h2' must be 1D arrays");
    return 0;
  }

  PyObject* retval = 0;

  try {

    switch(h1->type_num) {

      case NPY_UINT8:
        retval = PyBlitzArrayCxx_FromCScalar(bob::math::histogram_intersection(*PyBlitzArrayCxx_AsBlitz<uint8_t,1>(h1), *PyBlitzArrayCxx_AsBlitz<uint8_t,1>(h2)));
        break;

      case NPY_UINT16:
        retval = PyBlitzArrayCxx_FromCScalar(bob::math::histogram_intersection(*PyBlitzArrayCxx_AsBlitz<uint16_t,1>(h1), *PyBlitzArrayCxx_AsBlitz<uint16_t,1>(h2)));
        break;

      case NPY_INT32:
        retval = PyBlitzArrayCxx_FromCScalar(bob::math::histogram_intersection(*PyBlitzArrayCxx_AsBlitz<int32_t,1>(h1), *PyBlitzArrayCxx_AsBlitz<int32_t,1>(h2)));
        break;

      case NPY_INT64:
        retval = PyBlitzArrayCxx_FromCScalar(bob::math::histogram_intersection(*PyBlitzArrayCxx_AsBlitz<int64_t,1>(h1), *PyBlitzArrayCxx_AsBlitz<int64_t,1>(h2)));
        break;

      case NPY_FLOAT64:
        retval = PyBlitzArrayCxx_FromCScalar(bob::math::histogram_intersection(*PyBlitzArrayCxx_AsBlitz<double,1>(h1), *PyBlitzArrayCxx_AsBlitz<double,1>(h2)));
        break;

      default:
        PyErr_Format(PyExc_TypeError, "Histogram intersection currently not implemented for type '%s'", PyBlitzArray_TypenumAsString(h1->type_num));
        return 0;

    }

  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return 0;
  }
  catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "histogram intersection failed: unknown exception caught");
    return 0;
  }

  return retval;

}

template <typename T1> PyObject* py_histogram_intersection_2_inner(
    PyBlitzArrayObject* index1, PyBlitzArrayObject* value1,
    PyBlitzArrayObject* index2, PyBlitzArrayObject* value2) {

  switch(value1->type_num) {

    case NPY_UINT8:
      return PyBlitzArrayCxx_FromCScalar(bob::math::histogram_intersection(
            *PyBlitzArrayCxx_AsBlitz<T1,1>(index1),
            *PyBlitzArrayCxx_AsBlitz<uint8_t,1>(value1),
            *PyBlitzArrayCxx_AsBlitz<T1,1>(index2),
            *PyBlitzArrayCxx_AsBlitz<uint8_t,1>(value2)));

    case NPY_UINT16:
      return PyBlitzArrayCxx_FromCScalar(bob::math::histogram_intersection(
            *PyBlitzArrayCxx_AsBlitz<T1,1>(index1),
            *PyBlitzArrayCxx_AsBlitz<uint16_t,1>(value1),
            *PyBlitzArrayCxx_AsBlitz<T1,1>(index2),
            *PyBlitzArrayCxx_AsBlitz<uint16_t,1>(value2)));

    case NPY_INT32:
      return PyBlitzArrayCxx_FromCScalar(bob::math::histogram_intersection(
            *PyBlitzArrayCxx_AsBlitz<T1,1>(index1),
            *PyBlitzArrayCxx_AsBlitz<int32_t,1>(value1),
            *PyBlitzArrayCxx_AsBlitz<T1,1>(index2),
            *PyBlitzArrayCxx_AsBlitz<int32_t,1>(value2)));

    case NPY_INT64:
      return PyBlitzArrayCxx_FromCScalar(bob::math::histogram_intersection(
            *PyBlitzArrayCxx_AsBlitz<T1,1>(index1),
            *PyBlitzArrayCxx_AsBlitz<int64_t,1>(value1),
            *PyBlitzArrayCxx_AsBlitz<T1,1>(index2),
            *PyBlitzArrayCxx_AsBlitz<int64_t,1>(value2)));

    case NPY_FLOAT64:
      return PyBlitzArrayCxx_FromCScalar(bob::math::histogram_intersection(
            *PyBlitzArrayCxx_AsBlitz<T1,1>(index1),
            *PyBlitzArrayCxx_AsBlitz<double,1>(value1),
            *PyBlitzArrayCxx_AsBlitz<T1,1>(index2),
            *PyBlitzArrayCxx_AsBlitz<double,1>(value2)));

    default:
      break;

  }

  PyErr_Format(PyExc_TypeError, "Histogram intersection currently not implemented for value type '%s'", PyBlitzArray_TypenumAsString(value1->type_num));
  return 0;

}

static PyObject* py_histogram_intersection_2(PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = { "index_1", "value_1", "index_2", "value_2", 0 /* Sentinel */ };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* index1 = 0;
  PyBlitzArrayObject* value1 = 0;
  PyBlitzArrayObject* index2 = 0;
  PyBlitzArrayObject* value2 = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&O&O&",
        kwlist,
        &PyBlitzArray_Converter, &index1,
        &PyBlitzArray_Converter, &value1,
        &PyBlitzArray_Converter, &index2,
        &PyBlitzArray_Converter, &value2
        )) return 0;

  //protects acquired resources through this scope
  auto index1_ = make_safe(index1);
  auto value1_ = make_safe(value1);
  auto index2_ = make_safe(index2);
  auto value2_ = make_safe(value2);

  // checks for type mismatch
  if (index1->type_num != index2->type_num) {
    PyErr_Format(PyExc_TypeError, "data type mismatch between `index_1' and `index_2' (%s != %s)", PyBlitzArray_TypenumAsString(index1->type_num), PyBlitzArray_TypenumAsString(index2->type_num));
    return 0;
  }

  if (value1->type_num != value2->type_num) {
    PyErr_Format(PyExc_TypeError, "data type mismatch between `value_1' and `value_2' (%s != %s)", PyBlitzArray_TypenumAsString(value1->type_num), PyBlitzArray_TypenumAsString(value2->type_num));
    return 0;
  }

  // input arrays must be 1d
  if (index1->ndim != 1 || index2->ndim != 1 ||
      value1->ndim != 1 || value2->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "all input arrays must be 1D");
    return 0;
  }

  PyObject* retval = 0;

  try {

    switch(index1->type_num) {

      case NPY_UINT8:
        retval = py_histogram_intersection_2_inner<uint8_t>(index1, value1,
            index2, value2);
        break;

      case NPY_UINT16:
        retval = py_histogram_intersection_2_inner<uint16_t>(index1, value1,
            index2, value2);
        break;

      case NPY_INT32:
        retval = py_histogram_intersection_2_inner<int32_t>(index1, value1,
            index2, value2);
        break;

      case NPY_INT64:
        retval = py_histogram_intersection_2_inner<int64_t>(index1, value1,
            index2, value2);
        break;

      case NPY_FLOAT64:
        retval = py_histogram_intersection_2_inner<double>(index1, value1,
            index2, value2);
        break;

      default:
        PyErr_Format(PyExc_TypeError, "Histogram intersection currently not implemented for index type '%s'", PyBlitzArray_TypenumAsString(index1->type_num));
        return 0;

    }

  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return 0;
  }
  catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "histogram intersection failed: unknown exception caught");
    return 0;
  }

  return retval;

}

/**
 * Note: Dispatcher function.
 */
PyObject* py_histogram_intersection (PyObject*, PyObject* args, PyObject* kwargs) {

  Py_ssize_t nargs = args?PyTuple_Size(args):0 + kwargs?PyDict_Size(kwargs):0;

  PyObject* retval = 0;

  switch (nargs) {

    case 2:
      retval = py_histogram_intersection_1(0, args, kwargs);
      break;

    case 4:
      retval = py_histogram_intersection_2(0, args, kwargs);
      break;

    default:

      PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - kullback_leibler requires 2 or 4 arguments, but you provided %" PY_FORMAT_SIZE_T "d (see help)", nargs);

  }

  return retval;

}

static PyObject* py_chi_square_1
(PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = { "h1", "h2", 0 /* Sentinel */ };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* h1 = 0;
  PyBlitzArrayObject* h2 = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&",
        kwlist, &PyBlitzArray_Converter, &h1, &PyBlitzArray_Converter, &h2)) return 0;

  //protects acquired resources through this scope
  auto h1_ = make_safe(h1);
  auto h2_ = make_safe(h2);

  // checks for type mismatch
  if (h1->type_num != h2->type_num) {
    PyErr_Format(PyExc_TypeError, "data type mismatch between `h1' and `h2' (%s != %s)", PyBlitzArray_TypenumAsString(h1->type_num), PyBlitzArray_TypenumAsString(h2->type_num));
    return 0;
  }

  // input arrays must be 1d
  if (h1->ndim != 1 || h2->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "both `h1' and `h2' must be 1D arrays");
    return 0;
  }

  PyObject* retval = 0;

  switch(h1->type_num) {

    case NPY_UINT8:
      retval = PyBlitzArrayCxx_FromCScalar(bob::math::chi_square(*PyBlitzArrayCxx_AsBlitz<uint8_t,1>(h1), *PyBlitzArrayCxx_AsBlitz<uint8_t,1>(h2)));
      break;

    case NPY_UINT16:
      retval = PyBlitzArrayCxx_FromCScalar(bob::math::chi_square(*PyBlitzArrayCxx_AsBlitz<uint16_t,1>(h1), *PyBlitzArrayCxx_AsBlitz<uint16_t,1>(h2)));
      break;

    case NPY_INT32:
      retval = PyBlitzArrayCxx_FromCScalar(bob::math::chi_square(*PyBlitzArrayCxx_AsBlitz<int32_t,1>(h1), *PyBlitzArrayCxx_AsBlitz<int32_t,1>(h2)));
      break;

    case NPY_INT64:
      retval = PyBlitzArrayCxx_FromCScalar(bob::math::chi_square(*PyBlitzArrayCxx_AsBlitz<int64_t,1>(h1), *PyBlitzArrayCxx_AsBlitz<int64_t,1>(h2)));
      break;

    case NPY_FLOAT64:
      retval = PyBlitzArrayCxx_FromCScalar(bob::math::chi_square(*PyBlitzArrayCxx_AsBlitz<double,1>(h1), *PyBlitzArrayCxx_AsBlitz<double,1>(h2)));
      break;

    default:
      PyErr_Format(PyExc_TypeError, "Chi^2 distance between histograms currently not implemented for type '%s'", PyBlitzArray_TypenumAsString(h1->type_num));

  }

  return retval;

}

template <typename T1> PyObject* py_chi_square_2_inner(
    PyBlitzArrayObject* index1, PyBlitzArrayObject* value1,
    PyBlitzArrayObject* index2, PyBlitzArrayObject* value2) {

  switch(value1->type_num) {

    case NPY_UINT8:
      return PyBlitzArrayCxx_FromCScalar(bob::math::chi_square(
            *PyBlitzArrayCxx_AsBlitz<T1,1>(index1),
            *PyBlitzArrayCxx_AsBlitz<uint8_t,1>(value1),
            *PyBlitzArrayCxx_AsBlitz<T1,1>(index2),
            *PyBlitzArrayCxx_AsBlitz<uint8_t,1>(value2)));

    case NPY_UINT16:
      return PyBlitzArrayCxx_FromCScalar(bob::math::chi_square(
            *PyBlitzArrayCxx_AsBlitz<T1,1>(index1),
            *PyBlitzArrayCxx_AsBlitz<uint16_t,1>(value1),
            *PyBlitzArrayCxx_AsBlitz<T1,1>(index2),
            *PyBlitzArrayCxx_AsBlitz<uint16_t,1>(value2)));

    case NPY_INT32:
      return PyBlitzArrayCxx_FromCScalar(bob::math::chi_square(
            *PyBlitzArrayCxx_AsBlitz<T1,1>(index1),
            *PyBlitzArrayCxx_AsBlitz<int32_t,1>(value1),
            *PyBlitzArrayCxx_AsBlitz<T1,1>(index2),
            *PyBlitzArrayCxx_AsBlitz<int32_t,1>(value2)));

    case NPY_INT64:
      return PyBlitzArrayCxx_FromCScalar(bob::math::chi_square(
            *PyBlitzArrayCxx_AsBlitz<T1,1>(index1),
            *PyBlitzArrayCxx_AsBlitz<int64_t,1>(value1),
            *PyBlitzArrayCxx_AsBlitz<T1,1>(index2),
            *PyBlitzArrayCxx_AsBlitz<int64_t,1>(value2)));

    case NPY_FLOAT64:
      return PyBlitzArrayCxx_FromCScalar(bob::math::chi_square(
            *PyBlitzArrayCxx_AsBlitz<T1,1>(index1),
            *PyBlitzArrayCxx_AsBlitz<double,1>(value1),
            *PyBlitzArrayCxx_AsBlitz<T1,1>(index2),
            *PyBlitzArrayCxx_AsBlitz<double,1>(value2)));

    default:
      break;

  }

  PyErr_Format(PyExc_TypeError, "Chi^2 distance between histograms currently not implemented for value type '%s'", PyBlitzArray_TypenumAsString(value1->type_num));
  return 0;

}

static PyObject* py_chi_square_2(PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = { "index_1", "value_1", "index_2", "value_2", 0 /* Sentinel */ };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* index1 = 0;
  PyBlitzArrayObject* value1 = 0;
  PyBlitzArrayObject* index2 = 0;
  PyBlitzArrayObject* value2 = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&O&O&",
        kwlist,
        &PyBlitzArray_Converter, &index1,
        &PyBlitzArray_Converter, &value1,
        &PyBlitzArray_Converter, &index2,
        &PyBlitzArray_Converter, &value2
        )) return 0;

  //protects acquired resources through this scope
  auto index1_ = make_safe(index1);
  auto value1_ = make_safe(value1);
  auto index2_ = make_safe(index2);
  auto value2_ = make_safe(value2);

  // checks for type mismatch
  if (index1->type_num != index2->type_num) {
    PyErr_Format(PyExc_TypeError, "data type mismatch between `index_1' and `index_2' (%s != %s)", PyBlitzArray_TypenumAsString(index1->type_num), PyBlitzArray_TypenumAsString(index2->type_num));
    return 0;
  }

  if (value1->type_num != value2->type_num) {
    PyErr_Format(PyExc_TypeError, "data type mismatch between `value_1' and `value_2' (%s != %s)", PyBlitzArray_TypenumAsString(value1->type_num), PyBlitzArray_TypenumAsString(value2->type_num));
    return 0;
  }

  // input arrays must be 1d
  if (index1->ndim != 1 || index2->ndim != 1 ||
      value1->ndim != 1 || value2->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "all input arrays must be 1D");
    return 0;
  }

  PyObject* retval = 0;

  switch(index1->type_num) {

    case NPY_UINT8:
      retval = py_chi_square_2_inner<uint8_t>(index1, value1,
          index2, value2);
      break;

    case NPY_UINT16:
      retval = py_chi_square_2_inner<uint16_t>(index1, value1,
          index2, value2);
      break;

    case NPY_INT32:
      retval = py_chi_square_2_inner<int32_t>(index1, value1,
          index2, value2);
      break;

    case NPY_INT64:
      retval = py_chi_square_2_inner<int64_t>(index1, value1,
          index2, value2);
      break;

    case NPY_FLOAT64:
      retval = py_chi_square_2_inner<double>(index1, value1,
          index2, value2);
      break;

    default:
      PyErr_Format(PyExc_TypeError, "Histogram intersection currently not implemented for index type '%s'", PyBlitzArray_TypenumAsString(index1->type_num));

  }

  return retval;

}

/**
 * Note: Dispatcher function.
 */
PyObject* py_chi_square (PyObject*, PyObject* args, PyObject* kwargs) {

  Py_ssize_t nargs = args?PyTuple_Size(args):0 + kwargs?PyDict_Size(kwargs):0;

  PyObject* retval = 0;

  switch (nargs) {

    case 2:
      retval = py_chi_square_1(0, args, kwargs);
      break;

    case 4:
      retval = py_chi_square_2(0, args, kwargs);
      break;

    default:

      PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - kullback_leibler requires 2 or 4 arguments, but you provided %" PY_FORMAT_SIZE_T "d (see help)", nargs);


  }

  return retval;

}

static PyObject* py_kullback_leibler_1
(PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = { "h1", "h2", 0 /* Sentinel */ };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* h1 = 0;
  PyBlitzArrayObject* h2 = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&",
        kwlist, &PyBlitzArray_Converter, &h1, &PyBlitzArray_Converter, &h2)) return 0;

  //protects acquired resources through this scope
  auto h1_ = make_safe(h1);
  auto h2_ = make_safe(h2);

  // checks for type mismatch
  if (h1->type_num != h2->type_num) {
    PyErr_Format(PyExc_TypeError, "data type mismatch between `h1' and `h2' (%s != %s)", PyBlitzArray_TypenumAsString(h1->type_num), PyBlitzArray_TypenumAsString(h2->type_num));
    return 0;
  }

  // input arrays must be 1d
  if (h1->ndim != 1 || h2->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "both `h1' and `h2' must be 1D arrays");
    return 0;
  }

  PyObject* retval = 0;

  switch(h1->type_num) {

    case NPY_UINT8:
      retval = PyBlitzArrayCxx_FromCScalar(bob::math::kullback_leibler(*PyBlitzArrayCxx_AsBlitz<uint8_t,1>(h1), *PyBlitzArrayCxx_AsBlitz<uint8_t,1>(h2)));
      break;

    case NPY_UINT16:
      retval = PyBlitzArrayCxx_FromCScalar(bob::math::kullback_leibler(*PyBlitzArrayCxx_AsBlitz<uint16_t,1>(h1), *PyBlitzArrayCxx_AsBlitz<uint16_t,1>(h2)));
      break;

    case NPY_INT32:
      retval = PyBlitzArrayCxx_FromCScalar(bob::math::kullback_leibler(*PyBlitzArrayCxx_AsBlitz<int32_t,1>(h1), *PyBlitzArrayCxx_AsBlitz<int32_t,1>(h2)));
      break;

    case NPY_INT64:
      retval = PyBlitzArrayCxx_FromCScalar(bob::math::kullback_leibler(*PyBlitzArrayCxx_AsBlitz<int64_t,1>(h1), *PyBlitzArrayCxx_AsBlitz<int64_t,1>(h2)));
      break;

    case NPY_FLOAT64:
      retval = PyBlitzArrayCxx_FromCScalar(bob::math::kullback_leibler(*PyBlitzArrayCxx_AsBlitz<double,1>(h1), *PyBlitzArrayCxx_AsBlitz<double,1>(h2)));
      break;

    default:
      PyErr_Format(PyExc_TypeError, "Histogram intersection currently not implemented for type '%s'", PyBlitzArray_TypenumAsString(h1->type_num));

  }

  return retval;

}

template <typename T1> PyObject* py_kullback_leibler_2_inner(
    PyBlitzArrayObject* index1, PyBlitzArrayObject* value1,
    PyBlitzArrayObject* index2, PyBlitzArrayObject* value2) {

  switch(value1->type_num) {

    case NPY_UINT8:
      return PyBlitzArrayCxx_FromCScalar(bob::math::kullback_leibler(
            *PyBlitzArrayCxx_AsBlitz<T1,1>(index1),
            *PyBlitzArrayCxx_AsBlitz<uint8_t,1>(value1),
            *PyBlitzArrayCxx_AsBlitz<T1,1>(index2),
            *PyBlitzArrayCxx_AsBlitz<uint8_t,1>(value2)));

    case NPY_UINT16:
      return PyBlitzArrayCxx_FromCScalar(bob::math::kullback_leibler(
            *PyBlitzArrayCxx_AsBlitz<T1,1>(index1),
            *PyBlitzArrayCxx_AsBlitz<uint16_t,1>(value1),
            *PyBlitzArrayCxx_AsBlitz<T1,1>(index2),
            *PyBlitzArrayCxx_AsBlitz<uint16_t,1>(value2)));

    case NPY_INT32:
      return PyBlitzArrayCxx_FromCScalar(bob::math::kullback_leibler(
            *PyBlitzArrayCxx_AsBlitz<T1,1>(index1),
            *PyBlitzArrayCxx_AsBlitz<int32_t,1>(value1),
            *PyBlitzArrayCxx_AsBlitz<T1,1>(index2),
            *PyBlitzArrayCxx_AsBlitz<int32_t,1>(value2)));

    case NPY_INT64:
      return PyBlitzArrayCxx_FromCScalar(bob::math::kullback_leibler(
            *PyBlitzArrayCxx_AsBlitz<T1,1>(index1),
            *PyBlitzArrayCxx_AsBlitz<int64_t,1>(value1),
            *PyBlitzArrayCxx_AsBlitz<T1,1>(index2),
            *PyBlitzArrayCxx_AsBlitz<int64_t,1>(value2)));

    case NPY_FLOAT64:
      return PyBlitzArrayCxx_FromCScalar(bob::math::kullback_leibler(
            *PyBlitzArrayCxx_AsBlitz<T1,1>(index1),
            *PyBlitzArrayCxx_AsBlitz<double,1>(value1),
            *PyBlitzArrayCxx_AsBlitz<T1,1>(index2),
            *PyBlitzArrayCxx_AsBlitz<double,1>(value2)));

    default:
      break;

  }

  PyErr_Format(PyExc_TypeError, "Histogram intersection currently not implemented for value type '%s'", PyBlitzArray_TypenumAsString(value1->type_num));
  return 0;

}

static PyObject* py_kullback_leibler_2(PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = { "index_1", "value_1", "index_2", "value_2", 0 /* Sentinel */ };
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* index1 = 0;
  PyBlitzArrayObject* value1 = 0;
  PyBlitzArrayObject* index2 = 0;
  PyBlitzArrayObject* value2 = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&O&O&",
        kwlist,
        &PyBlitzArray_Converter, &index1,
        &PyBlitzArray_Converter, &value1,
        &PyBlitzArray_Converter, &index2,
        &PyBlitzArray_Converter, &value2
        )) return 0;

  //protects acquired resources through this scope
  auto index1_ = make_safe(index1);
  auto value1_ = make_safe(value1);
  auto index2_ = make_safe(index2);
  auto value2_ = make_safe(value2);

  // checks for type mismatch
  if (index1->type_num != index2->type_num) {
    PyErr_Format(PyExc_TypeError, "data type mismatch between `index_1' and `index_2' (%s != %s)", PyBlitzArray_TypenumAsString(index1->type_num), PyBlitzArray_TypenumAsString(index2->type_num));
    return 0;
  }

  if (value1->type_num != value2->type_num) {
    PyErr_Format(PyExc_TypeError, "data type mismatch between `value_1' and `value_2' (%s != %s)", PyBlitzArray_TypenumAsString(value1->type_num), PyBlitzArray_TypenumAsString(value2->type_num));
    return 0;
  }

  // input arrays must be 1d
  if (index1->ndim != 1 || index2->ndim != 1 ||
      value1->ndim != 1 || value2->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "all input arrays must be 1D");
    return 0;
  }

  PyObject* retval = 0;

  switch(index1->type_num) {

    case NPY_UINT8:
      retval = py_kullback_leibler_2_inner<uint8_t>(index1, value1,
          index2, value2);
      break;

    case NPY_UINT16:
      retval = py_kullback_leibler_2_inner<uint16_t>(index1, value1,
          index2, value2);
      break;

    case NPY_INT32:
      retval = py_kullback_leibler_2_inner<int32_t>(index1, value1,
          index2, value2);
      break;

    case NPY_INT64:
      retval = py_kullback_leibler_2_inner<int64_t>(index1, value1,
          index2, value2);
      break;

    case NPY_FLOAT64:
      retval = py_kullback_leibler_2_inner<double>(index1, value1,
          index2, value2);
      break;

    default:
      PyErr_Format(PyExc_TypeError, "Histogram intersection currently not implemented for index type '%s'", PyBlitzArray_TypenumAsString(index1->type_num));

  }

  return retval;

}

/**
 * Note: Dispatcher function.
 */
PyObject* py_kullback_leibler (PyObject*, PyObject* args, PyObject* kwargs) {

  Py_ssize_t nargs = args?PyTuple_Size(args):0 + kwargs?PyDict_Size(kwargs):0;

  PyObject* retval = 0;

  switch (nargs) {

    case 2:
      retval = py_kullback_leibler_1(0, args, kwargs);
      break;

    case 4:
      retval = py_kullback_leibler_2(0, args, kwargs);
      break;

    default:

      PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - kullback_leibler requires 2 or 4 arguments, but you provided %" PY_FORMAT_SIZE_T "d (see help)", nargs);

  }

  return retval;

}
