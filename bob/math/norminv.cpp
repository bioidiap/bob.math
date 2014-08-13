/**
 * @file math/python/norminv.cc
 * @date Wed Apr 13 09:20:40 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds the inverse normal cumulative distribution into python
 */

#include "pavx.h"
#include <bob.blitz/cppapi.h>
#include <bob.math/norminv.h>

PyObject* py_norminv (PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = { "p", "mu", "sigma", 0 /* Sentinel */ };
  static char** kwlist = const_cast<char**>(const_kwlist);

  double p = 0.;
  double mu = 0.;
  double sigma = 0.;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "ddd", kwlist, &p, &mu, &sigma))
    return 0;

  try {
    return PyBlitzArrayCxx_FromCScalar(bob::math::norminv(p, mu, sigma));
  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
  }
  catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "norminv failed: unknown exception caught");
  }

  return 0;

}

PyObject* py_normsinv (PyObject*, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = { "p", 0 /* Sentinel */ };
  static char** kwlist = const_cast<char**>(const_kwlist);

  double p = 0.;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "d", kwlist, &p)) return 0;

  try {
    return PyBlitzArrayCxx_FromCScalar(bob::math::normsinv(p));
  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
  }
  catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "normsinv failed: unknown exception caught");
  }

  return 0;

}
