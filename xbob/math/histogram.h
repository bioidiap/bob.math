/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed  4 Dec 13:18:26 2013 
 *
 * @brief Declaration of components relevant for main.cpp
 */

#include <Python.h>

PyObject* py_histogram_intersection(PyObject*, PyObject* args, PyObject* kwds);
PyObject* py_chi_square(PyObject*, PyObject* args, PyObject* kwds);
PyObject* py_kullback_leibler(PyObject*, PyObject* args, PyObject* kwds);
