/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu  5 Dec 12:10:18 2013 
 *
 * @brief Declaration of components relevant for main.cpp
 */

#include <Python.h>

PyObject* py_scatter(PyObject*, PyObject* args, PyObject* kwds);
PyObject* py_scatter_nocheck(PyObject*, PyObject* args, PyObject* kwds);
PyObject* py_scatters(PyObject*, PyObject* args, PyObject* kwds);
PyObject* py_scatters_nocheck(PyObject*, PyObject* args, PyObject* kwds);
