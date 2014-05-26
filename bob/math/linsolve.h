/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed  4 Dec 15:26:54 2013 
 *
 * @brief Declaration of components relevant for main.cpp
 */

#include <Python.h>

PyObject* py_linsolve(PyObject*, PyObject* args, PyObject* kwargs);
PyObject* py_linsolve_nocheck(PyObject*, PyObject* args, PyObject* kwargs);
PyObject* py_linsolve_sympos(PyObject*, PyObject* args, PyObject* kwargs);
PyObject* py_linsolve_sympos_nocheck(PyObject*, PyObject* args, PyObject* kwargs);
PyObject* py_linsolve_cg_sympos(PyObject*, PyObject* args, PyObject* kwargs);
PyObject* py_linsolve_cg_sympos_nocheck(PyObject*, PyObject* args, PyObject* kwargs);
