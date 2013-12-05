/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed  4 Dec 17:46:18 2013 
 *
 * @brief Declaration of components relevant for main.cpp
 */

#include <Python.h>

PyObject* py_pavx(PyObject*, PyObject* args, PyObject* kwds);
PyObject* py_pavx_nocheck(PyObject*, PyObject* args, PyObject* kwds);
PyObject* py_pavx_width(PyObject*, PyObject* args, PyObject* kwds);
PyObject* py_pavx_width_height(PyObject*, PyObject* args, PyObject* kwds);
