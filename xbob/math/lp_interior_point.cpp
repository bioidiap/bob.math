/**
 * @file math/python/LPInteriorPoint.cc
 * @date Fri Jan 27 21:06:59 2012 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds the interior point methods which allow to solve a
 *        Linear Programming problem (LP).
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include "lp_interior_point.h"
#include <xbob.blitz/cppapi.h>
#include <bob/math/LPInteriorPoint.h>
#include <structmember.h>

PyDoc_STRVAR(s_lpinteriorpoint_str, XBOB_EXT_MODULE_PREFIX ".LPInteriorPoint");

PyDoc_STRVAR(s_lpinteriorpoint_doc,
"Base class to solve a linear program using interior point methods.\n\
For more details about the algorithms,please refer to the following\n\
book: *\"Primal-Dual Interior-Point Methods\", Stephen J. Wright,\n\
ISBN: 978-0898713824, Chapter 5, \"Path-Following Algorithms\"*.\n\
\n\
.. warning::\n\
\n\
   You cannot instantiate an object of this type directly, you must\n\
   use it through one of the inherited types.\n\
\n\
The primal linear program (LP) is defined as follows:\n\
\n\
   min transpose(c)*x, s.t. A*x=b, x>=0\n\
\n\
The dual formulation is:\n\
\n\
   min transpose(b)*lambda, s.t. transpose(A)*lambda+mu=c\n\
\n\
");

/* Type definition for PyBobMathLpInteriorPointObject */
typedef struct {
  PyObject_HEAD

  /* Type-specific fields go here. */
  std::shared_ptr<bob::math::LPInteriorPoint> base;

} PyBobMathLpInteriorPointObject;


static PyObject* PyBobMathLpInteriorPoint_new(PyTypeObject* type, PyObject*, PyObject*) {

  /* Allocates the python object itself */
  PyBobMathLpInteriorPointObject* self = (PyBobMathLpInteriorPointObject*)type->tp_alloc(type, 0);

  self->base.reset();

  return reinterpret_cast<PyObject*>(self);

}

static int PyBobMathLpInteriorPoint_init(PyBobMathLpInteriorPointObject* self, PyObject*, PyObject*) {

  PyErr_SetString(PyExc_NotImplementedError, "cannot initialize object of base type `LPInteriorPoint' - use one of the inherited classes");

  return -1;

}

static void PyBobMathLpInteriorPoint_delete (PyBobMathLpInteriorPointObject* self) {
  
  self->base.reset();
  self->ob_type->tp_free((PyObject*)self);

}

PyDoc_STRVAR(s_M_str, "M");
PyDoc_STRVAR(s_M_doc,
"The first dimension of the problem/A matrix"
);

static PyObject* PyBobMathLpInteriorPoint_M (PyBobMathLpInteriorPointObject* self) {
  return Py_BuildValue("n", self->base->getDimM());
}

PyDoc_STRVAR(s_N_str, "N");
PyDoc_STRVAR(s_N_doc,
"The second dimension of the problem/A matrix"
);

static PyObject* PyBobMathLpInteriorPoint_N (PyBobMathLpInteriorPointObject* self) {
  return Py_BuildValue("n", self->base->getDimN());
}

PyDoc_STRVAR(s_epsilon_str, "epsilon");
PyDoc_STRVAR(s_epsilon_doc,
"The precision to determine whether an equality constraint is fulfilled or not"
);

static PyObject* PyBobMathLpInteriorPoint_epsilon (PyBobMathLpInteriorPointObject* self) {
  return Py_BuildValue("d", self->base->getEpsilon());
}

PyDoc_STRVAR(s_lambda_str, "lambda");
PyDoc_STRVAR(s_lambda_doc,
"The value of the lambda dual variable (read-only)"
);

static PyObject* PyBobMathLpInteriorPoint_lambda (PyBobMathLpInteriorPointObject* self) {
  Py_ssize_t length = self->base->getDimM();
  PyObject* retval = PyBlitzArray_SimpleNew(NPY_FLOAT64, 1, &length);
  if (!retval) return 0;

  blitz::Array<double,1>* wrapper = PyBlitzArrayCxx_AsBlitz<double,1>
    (reinterpret_cast<PyBlitzArrayObject*>(retval));
  (*wrapper) = self->base->getLambda();

  return retval;
}

PyDoc_STRVAR(s_mu_str, "mu");
PyDoc_STRVAR(s_mu_doc,
"The value of the mu dual variable (read-only)"
);

static PyObject* PyBobMathLpInteriorPoint_mu (PyBobMathLpInteriorPointObject* self) {
  Py_ssize_t length = self->base->getDimN();
  PyObject* retval = PyBlitzArray_SimpleNew(NPY_FLOAT64, 1, &length);
  if (!retval) return 0;

  blitz::Array<double,1>* wrapper = PyBlitzArrayCxx_AsBlitz<double,1>
    (reinterpret_cast<PyBlitzArrayObject*>(retval));
  (*wrapper) = self->base->getMu();

  return retval;
}

static PyGetSetDef PyBobMathLpInteriorPoint_getseters[] = {
    {
      s_M_str, 
      (getter)PyBobMathLpInteriorPoint_M,
      0,
      s_M_doc,
      0
    },
    {
      s_N_str, 
      (getter)PyBobMathLpInteriorPoint_N,
      0,
      s_N_doc,
      0
    },
    {
      s_epsilon_str, 
      (getter)PyBobMathLpInteriorPoint_epsilon,
      0,
      s_epsilon_doc,
      0
    },
    {
      s_lambda_str, 
      (getter)PyBobMathLpInteriorPoint_lambda,
      0,
      s_lambda_doc,
      0
    },
    {
      s_mu_str, 
      (getter)PyBobMathLpInteriorPoint_mu,
      0,
      s_mu_doc,
      0
    },
    {0}  /* Sentinel */
};

PyDoc_STRVAR(s_reset_str, "reset");
PyDoc_STRVAR(s_reset_doc,
"o.reset(M, N) -> None\n\
\n\
Resets the size of the problem (M and N correspond to the dimensions of the\n\
A matrix");

static PyObject* PyBobMathLpInteriorPoint_reset 
(PyBobMathLpInteriorPointObject* self, PyObject *args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"M", "N", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  Py_ssize_t M = 0;
  Py_ssize_t N = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "nn", kwlist, &M, &N)) return 0;

  try {
    self->base->reset(M, N);
  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return 0;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot reset LPInteriorPoint: unknown exception caught");
    return 0;
  }

  Py_RETURN_NONE;

}

PyDoc_STRVAR(s_solve_str, "solve");
PyDoc_STRVAR(s_solve_doc, 
"o.solve(A, b, c, x0, [lambda, mu]) -> x\n\
\n\
Solves an LP problem\n\
");

static PyObject* PyBobMathLpInteriorPoint_solve 
(PyBobMathLpInteriorPointObject* self, PyObject *args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"A", "b", "c", "x0", "lambda", "mu", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* A = 0;
  PyBlitzArrayObject* b = 0;
  PyBlitzArrayObject* c = 0;
  PyBlitzArrayObject* x0 = 0;
  PyBlitzArrayObject* lambda = 0;
  PyBlitzArrayObject* mu = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&O&O&|O&O&", kwlist, 
        &PyBlitzArray_Converter, &A,
        &PyBlitzArray_Converter, &b,
        &PyBlitzArray_Converter, &c,
        &PyBlitzArray_Converter, &x0,
        &PyBlitzArray_Converter, &lambda,
        &PyBlitzArray_Converter, &mu
        )) return 0;

  if (A->type_num != NPY_FLOAT64 || A->ndim != 2) {
    PyErr_SetString(PyExc_TypeError, "Linear program solver only supports 64-bit floats 2D arrays for input vector `A'");
    Py_DECREF(A);
    Py_DECREF(b);
    Py_DECREF(c);
    Py_DECREF(x0);
    Py_XDECREF(lambda);
    Py_XDECREF(mu);
    return 0;
  }

  if (b->type_num != NPY_FLOAT64 || b->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "Linear program solver only supports 64-bit floats 1D arrays for input vector `b'");
    Py_DECREF(A);
    Py_DECREF(b);
    Py_DECREF(c);
    Py_DECREF(x0);
    Py_XDECREF(lambda);
    Py_XDECREF(mu);
    return 0;
  }

  if (c->type_num != NPY_FLOAT64 || c->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "Linear program solver only supports 64-bit floats 1D arrays for input vector `c'");
    Py_DECREF(A);
    Py_DECREF(b);
    Py_DECREF(c);
    Py_DECREF(x0);
    Py_XDECREF(lambda);
    Py_XDECREF(mu);
    return 0;
  }

  if (x0->type_num != NPY_FLOAT64 || x0->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "Linear program solver only supports 64-bit floats 1D arrays for input vector `x0'");
    Py_DECREF(A);
    Py_DECREF(b);
    Py_DECREF(c);
    Py_DECREF(x0);
    Py_XDECREF(lambda);
    Py_XDECREF(mu);
    return 0;
  }

  if (lambda->type_num != NPY_FLOAT64 || lambda->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "Linear program solver only supports 64-bit floats 1D arrays for input vector `lambda'");
    Py_DECREF(A);
    Py_DECREF(b);
    Py_DECREF(c);
    Py_DECREF(x0);
    Py_XDECREF(lambda);
    Py_XDECREF(mu);
    return 0;
  }

  if (mu->type_num != NPY_FLOAT64 || mu->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "Linear program solver only supports 64-bit floats 1D arrays for input vector `mu'");
    Py_DECREF(A);
    Py_DECREF(b);
    Py_DECREF(c);
    Py_DECREF(x0);
    Py_XDECREF(lambda);
    Py_XDECREF(mu);
    return 0;
  }

  if ( (lambda && !mu) || (mu && !lambda) ) {
    PyErr_SetString(PyExc_RuntimeError, "Linear program solver requires none or both `mu' and `lambda' - you cannot just specify one of them");
    Py_DECREF(A);
    Py_DECREF(b);
    Py_DECREF(c);
    Py_DECREF(x0);
    Py_XDECREF(lambda);
    Py_XDECREF(mu);
    return 0;
  }

  /* This is where the output is going to be stored */
  PyObject* retval = PyBlitzArray_SimpleNew(NPY_FLOAT64, x0->ndim, x0->shape);
  if (!retval) {
    Py_DECREF(A);
    Py_DECREF(b);
    Py_DECREF(c);
    Py_DECREF(x0);
    Py_XDECREF(lambda);
    Py_XDECREF(mu);
    return 0;
  }
  blitz::Array<double,1>* wrapper = PyBlitzArrayCxx_AsBlitz<double,1>(reinterpret_cast<PyBlitzArrayObject*>(retval));
  (*wrapper) = *PyBlitzArrayCxx_AsBlitz<double,1>(x0);
  Py_DECREF(x0);

  try {
    if (lambda && mu) {
      self->base->solve(
          *PyBlitzArrayCxx_AsBlitz<double,2>(A),
          *PyBlitzArrayCxx_AsBlitz<double,1>(b),
          *PyBlitzArrayCxx_AsBlitz<double,1>(c),
          *wrapper,
          *PyBlitzArrayCxx_AsBlitz<double,1>(lambda),
          *PyBlitzArrayCxx_AsBlitz<double,1>(mu)
          );
    }
    else {
      self->base->solve(
          *PyBlitzArrayCxx_AsBlitz<double,2>(A),
          *PyBlitzArrayCxx_AsBlitz<double,1>(b),
          *PyBlitzArrayCxx_AsBlitz<double,1>(c),
          *wrapper
          );
    }
  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return 0;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot solve LPInteriorPoint: unknown exception caught");
    return 0;
  }

  Py_DECREF(A);
  Py_DECREF(b);
  Py_DECREF(c);
  Py_XDECREF(lambda);
  Py_XDECREF(mu);

  if (PyErr_Occurred()) {
    Py_DECREF(retval);
    return 0;
  }

  /* We only "return" the first half of the `x' vector */
  (reinterpret_cast<PyBlitzArrayObject*>(retval))->shape[0] /= 2;
  return retval;

}

PyDoc_STRVAR(s_is_feasible_str, "is_feasible");
PyDoc_STRVAR(s_is_feasible_doc, 
"o.is_feasible(A, b, c, x, lambda, mu) -> bool\n\
\n\
Checks if a primal-dual point (x,lambda,mu) belongs to the set of\n\
feasible point (i.e. fulfill the constraints)\n\
\n\
");

static PyObject* PyBobMathLpInteriorPoint_is_feasible
(PyBobMathLpInteriorPointObject* self, PyObject *args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"A", "b", "c", "x", "lambda", "mu", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* A = 0;
  PyBlitzArrayObject* b = 0;
  PyBlitzArrayObject* c = 0;
  PyBlitzArrayObject* x = 0;
  PyBlitzArrayObject* lambda = 0;
  PyBlitzArrayObject* mu = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&O&O&O&O&", kwlist,
        &PyBlitzArray_Converter, &A,
        &PyBlitzArray_Converter, &b,
        &PyBlitzArray_Converter, &c,
        &PyBlitzArray_Converter, &x,
        &PyBlitzArray_Converter, &lambda,
        &PyBlitzArray_Converter, &mu
        )) return 0;

  if (A->type_num != NPY_FLOAT64 || A->ndim != 2) {
    PyErr_SetString(PyExc_TypeError, "Linear program is_feasible only supports 64-bit floats 2D arrays for input vector `A'");
    Py_DECREF(A);
    Py_DECREF(b);
    Py_DECREF(c);
    Py_DECREF(x);
    Py_DECREF(lambda);
    Py_DECREF(mu);
    return 0;
  }

  if (b->type_num != NPY_FLOAT64 || b->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "Linear program is_feasible only supports 64-bit floats 1D arrays for input vector `b'");
    Py_DECREF(A);
    Py_DECREF(b);
    Py_DECREF(c);
    Py_DECREF(x);
    Py_DECREF(lambda);
    Py_DECREF(mu);
    return 0;
  }

  if (c->type_num != NPY_FLOAT64 || c->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "Linear program is_feasible only supports 64-bit floats 1D arrays for input vector `c'");
    Py_DECREF(A);
    Py_DECREF(b);
    Py_DECREF(c);
    Py_DECREF(x);
    Py_DECREF(lambda);
    Py_DECREF(mu);
    return 0;
  }

  if (x->type_num != NPY_FLOAT64 || x->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "Linear program is_feasible only supports 64-bit floats 1D arrays for input vector `x0'");
    Py_DECREF(A);
    Py_DECREF(b);
    Py_DECREF(c);
    Py_DECREF(x);
    Py_DECREF(lambda);
    Py_DECREF(mu);
    return 0;
  }

  if (lambda->type_num != NPY_FLOAT64 || lambda->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "Linear program is_feasible only supports 64-bit floats 1D arrays for input vector `lambda'");
    Py_DECREF(A);
    Py_DECREF(b);
    Py_DECREF(c);
    Py_DECREF(x);
    Py_DECREF(lambda);
    Py_DECREF(mu);
    return 0;
  }

  if (mu->type_num != NPY_FLOAT64 || mu->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "Linear program is_feasible only supports 64-bit floats 1D arrays for input vector `mu'");
    Py_DECREF(A);
    Py_DECREF(b);
    Py_DECREF(c);
    Py_DECREF(x);
    Py_DECREF(lambda);
    Py_DECREF(mu);
    return 0;
  }

  bool feasible = false;

  try {
    feasible = self->base->isFeasible(
        *PyBlitzArrayCxx_AsBlitz<double,2>(A),
        *PyBlitzArrayCxx_AsBlitz<double,1>(b),
        *PyBlitzArrayCxx_AsBlitz<double,1>(c),
        *PyBlitzArrayCxx_AsBlitz<double,1>(x),
        *PyBlitzArrayCxx_AsBlitz<double,1>(lambda),
        *PyBlitzArrayCxx_AsBlitz<double,1>(mu)
        );
  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return 0;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot check feasibility of LPInteriorPoint: unknown exception caught");
    return 0;
  }

  Py_DECREF(A);
  Py_DECREF(b);
  Py_DECREF(c);
  Py_DECREF(x);
  Py_DECREF(lambda);
  Py_DECREF(mu);

  if (feasible) Py_RETURN_TRUE;
  Py_RETURN_FALSE;

}

PyDoc_STRVAR(s_is_in_v_str, "is_in_v");
PyDoc_STRVAR(s_is_in_v_doc, 
"o.is_in_v(x, mu, theta) -> bool\n\
\n\
Check if a primal-dual point (x,lambda,mu) belongs to the V2\n\
neighborhood of the central path.\n\
\n\
");

static PyObject* PyBobMathLpInteriorPoint_is_in_v
(PyBobMathLpInteriorPointObject* self, PyObject *args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"x", "mu", "theta", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* x = 0;
  PyBlitzArrayObject* mu = 0;
  double theta = 0.;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&d", kwlist,
        &PyBlitzArray_Converter, &x,
        &PyBlitzArray_Converter, &mu,
        &theta
        )) return 0;

  if (x->type_num != NPY_FLOAT64 || x->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "Linear program is_in_v only supports 64-bit floats 1D arrays for input vector `x0'");
    Py_DECREF(x);
    Py_DECREF(mu);
    return 0;
  }

  if (mu->type_num != NPY_FLOAT64 || mu->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "Linear program is_in_v only supports 64-bit floats 1D arrays for input vector `mu'");
    Py_DECREF(x);
    Py_DECREF(mu);
    return 0;
  }

  bool in_v = false;

  try {
    in_v = self->base->isInV(
        *PyBlitzArrayCxx_AsBlitz<double,1>(x),
        *PyBlitzArrayCxx_AsBlitz<double,1>(mu),
        theta
        );
  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return 0;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot check if point is in V at LPInteriorPoint: unknown exception caught");
    return 0;
  }

  Py_DECREF(x);
  Py_DECREF(mu);

  if (in_v) Py_RETURN_TRUE;
  Py_RETURN_FALSE;

}

PyDoc_STRVAR(s_is_in_v_s_str, "is_in_v_s");
PyDoc_STRVAR(s_is_in_v_s_doc, 
"o.is_in_v_s(A, b, c, x, lambda, mu) -> bool\n\
\n\
Checks if a primal-dual point (x,lambda,mu) belongs to the V\n\
neighborhood of the central path and the set of feasible points.\n\
\n\
");

static PyObject* PyBobMathLpInteriorPoint_is_in_v_s
(PyBobMathLpInteriorPointObject* self, PyObject *args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"A", "b", "c", "x", "lambda", "mu", "theta", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* A = 0;
  PyBlitzArrayObject* b = 0;
  PyBlitzArrayObject* c = 0;
  PyBlitzArrayObject* x = 0;
  PyBlitzArrayObject* lambda = 0;
  PyBlitzArrayObject* mu = 0;
  double theta = 0.;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&O&O&O&O&d", kwlist,
        &PyBlitzArray_Converter, &A,
        &PyBlitzArray_Converter, &b,
        &PyBlitzArray_Converter, &c,
        &PyBlitzArray_Converter, &x,
        &PyBlitzArray_Converter, &lambda,
        &PyBlitzArray_Converter, &mu,
        &theta
        )) return 0;

  if (A->type_num != NPY_FLOAT64 || A->ndim != 2) {
    PyErr_SetString(PyExc_TypeError, "Linear program is_in_v_s only supports 64-bit floats 2D arrays for input vector `A'");
    Py_DECREF(A);
    Py_DECREF(b);
    Py_DECREF(c);
    Py_DECREF(x);
    Py_DECREF(lambda);
    Py_DECREF(mu);
    return 0;
  }

  if (b->type_num != NPY_FLOAT64 || b->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "Linear program is_in_v_s only supports 64-bit floats 1D arrays for input vector `b'");
    Py_DECREF(A);
    Py_DECREF(b);
    Py_DECREF(c);
    Py_DECREF(x);
    Py_DECREF(lambda);
    Py_DECREF(mu);
    return 0;
  }

  if (c->type_num != NPY_FLOAT64 || c->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "Linear program is_in_v_s only supports 64-bit floats 1D arrays for input vector `c'");
    Py_DECREF(A);
    Py_DECREF(b);
    Py_DECREF(c);
    Py_DECREF(x);
    Py_DECREF(lambda);
    Py_DECREF(mu);
    return 0;
  }

  if (x->type_num != NPY_FLOAT64 || x->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "Linear program is_in_v_s only supports 64-bit floats 1D arrays for input vector `x0'");
    Py_DECREF(A);
    Py_DECREF(b);
    Py_DECREF(c);
    Py_DECREF(x);
    Py_DECREF(lambda);
    Py_DECREF(mu);
    return 0;
  }

  if (lambda->type_num != NPY_FLOAT64 || lambda->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "Linear program is_in_v_s only supports 64-bit floats 1D arrays for input vector `lambda'");
    Py_DECREF(A);
    Py_DECREF(b);
    Py_DECREF(c);
    Py_DECREF(x);
    Py_DECREF(lambda);
    Py_DECREF(mu);
    return 0;
  }

  if (mu->type_num != NPY_FLOAT64 || mu->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "Linear program is_in_v_s only supports 64-bit floats 1D arrays for input vector `mu'");
    Py_DECREF(A);
    Py_DECREF(b);
    Py_DECREF(c);
    Py_DECREF(x);
    Py_DECREF(lambda);
    Py_DECREF(mu);
    return 0;
  }

  bool in_v_s = false;

  try {
    in_v_s = self->base->isInVS(
        *PyBlitzArrayCxx_AsBlitz<double,2>(A),
        *PyBlitzArrayCxx_AsBlitz<double,1>(b),
        *PyBlitzArrayCxx_AsBlitz<double,1>(c),
        *PyBlitzArrayCxx_AsBlitz<double,1>(x),
        *PyBlitzArrayCxx_AsBlitz<double,1>(lambda),
        *PyBlitzArrayCxx_AsBlitz<double,1>(mu),
        theta
        );
  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return 0;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot check if point is in VS at LPInteriorPoint: unknown exception caught");
    return 0;
  }

  Py_DECREF(A);
  Py_DECREF(b);
  Py_DECREF(c);
  Py_DECREF(x);
  Py_DECREF(lambda);
  Py_DECREF(mu);

  if (in_v_s) Py_RETURN_TRUE;
  Py_RETURN_FALSE;

}

static PyMethodDef PyBobMathLpInteriorPoint_methods[] = {
    {
      s_reset_str,
      (PyCFunction)PyBobMathLpInteriorPoint_reset,
      METH_NOARGS,
      s_reset_doc
    },
    {
      s_solve_str,
      (PyCFunction)PyBobMathLpInteriorPoint_solve,
      METH_NOARGS,
      s_solve_doc
    },
    {
      s_is_feasible_str,
      (PyCFunction)PyBobMathLpInteriorPoint_is_feasible,
      METH_NOARGS,
      s_is_feasible_doc
    },
    {
      s_is_in_v_str,
      (PyCFunction)PyBobMathLpInteriorPoint_is_in_v,
      METH_NOARGS,
      s_is_in_v_doc
    },
    {
      s_is_in_v_s_str,
      (PyCFunction)PyBobMathLpInteriorPoint_is_in_v_s,
      METH_NOARGS,
      s_is_in_v_s_doc
    },
    {0}  /* Sentinel */
};

PyTypeObject PyBobMathLpInteriorPoint_Type = {
    PyObject_HEAD_INIT(0)
    0,                                          /*ob_size*/
    s_lpinteriorpoint_str,                      /*tp_name*/
    sizeof(PyBobMathLpInteriorPointObject),     /*tp_basicsize*/
    0,                                          /*tp_itemsize*/
    (destructor)PyBobMathLpInteriorPoint_delete,/*tp_dealloc*/
    0,                                          /*tp_print*/
    0,                                          /*tp_getattr*/
    0,                                          /*tp_setattr*/
    0,                                          /*tp_compare*/
    0,                                          /*tp_repr*/
    0,                                          /*tp_as_number*/
    0,                                          /*tp_as_sequence*/
    0,                                          /*tp_as_mapping*/
    0,                                          /*tp_hash */
    0,                                          /*tp_call*/
    0,                                          /*tp_str*/
    0,                                          /*tp_getattro*/
    0,                                          /*tp_setattro*/
    0,                                          /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /*tp_flags*/
    s_lpinteriorpoint_doc,                      /* tp_doc */
    0,		                                      /* tp_traverse */
    0,		                                      /* tp_clear */
    0,		                                      /* tp_richcompare */
    0,		                                      /* tp_weaklistoffset */
    0,		                                      /* tp_iter */
    0,		                                      /* tp_iternext */
    PyBobMathLpInteriorPoint_methods,           /* tp_methods */
    0,                                          /* tp_members */
    PyBobMathLpInteriorPoint_getseters,         /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc)PyBobMathLpInteriorPoint_init,    /* tp_init */
    0,                                          /* tp_alloc */
    PyBobMathLpInteriorPoint_new,               /* tp_new */
};

/**
static void initialize_dual_lambda_mu(bob::math::LPInteriorPoint& op, bob::python::const_ndarray A,
  bob::python::const_ndarray c)
{
  op.initializeDualLambdaMu(A.bz<double,2>(), c.bz<double,1>());
}

static bool is_in_vinf(bob::math::LPInteriorPointLongstep& op,
  bob::python::const_ndarray x, bob::python::const_ndarray mu,
  const double gamma)
{
  return op.isInV(x.bz<double,1>(), mu.bz<double,1>(), gamma);
}


void bind_math_lp_interiorpoint()
{
    .def(self == self)
    .def(self != self)
    .def("initialize_dual_lambda_mu", &initialize_dual_lambda_mu, (arg("self"), arg("A"), arg("c")), "Initialize the dual variables lambda and mu by minimizing the logarithmic barrier function")
  ;

  class_<bob::math::LPInteriorPointShortstep, boost::shared_ptr<bob::math::LPInteriorPointShortstep>, bases<bob::math::LPInteriorPoint> >("LPInteriorPointShortstep", "A Linear Program solver based on a short step interior point method", init<const size_t, const size_t, const double, const double>((arg("self"), arg("M"), arg("N"), arg("theta"), arg("epsilon")), "Constructs a new LPInteriorPointShortstep solver"))
    .def(init<const bob::math::LPInteriorPointShortstep&>((arg("self"), arg("solver")), "Copy constructs a solver"))
    .def(self == self)
    .def(self != self)
    .add_property("theta", &bob::math::LPInteriorPointShortstep::getTheta, &bob::math::LPInteriorPointShortstep::setTheta, "The value theta used to define a V2 neighborhood")
  ;

  class_<bob::math::LPInteriorPointPredictorCorrector, boost::shared_ptr<bob::math::LPInteriorPointPredictorCorrector>, bases<bob::math::LPInteriorPoint> >("LPInteriorPointPredictorCorrector", "A Linear Program solver based on a predictor-corrector interior point method", init<const size_t, const size_t, const double, const double, const double>((arg("self"), arg("M"), arg("N"), arg("theta_pred"), arg("theta_corr"), arg("epsilon")), "Constructs a new LPInteriorPointPredictorCorrector solver"))
    .def(init<const bob::math::LPInteriorPointPredictorCorrector&>((arg("self"), arg("solver")), "Copy constructs a solver"))
    .def(self == self)
    .def(self != self)
    .add_property("theta_pred", &bob::math::LPInteriorPointPredictorCorrector::getThetaPred, &bob::math::LPInteriorPointPredictorCorrector::setThetaPred, "The value theta_pred used to define a V2 neighborhood")
    .add_property("theta_corr", &bob::math::LPInteriorPointPredictorCorrector::getThetaCorr, &bob::math::LPInteriorPointPredictorCorrector::setThetaCorr, "The value theta_corr used to define a V2 neighborhood")
  ;

  class_<bob::math::LPInteriorPointLongstep, boost::shared_ptr<bob::math::LPInteriorPointLongstep>, bases<bob::math::LPInteriorPoint> >("LPInteriorPointLongstep", "A Linear Program solver based on a ong step interior point method", init<const size_t, const size_t, const double, const double, const double>((arg("self"), arg("M"), arg("N"), arg("gamma"), arg("sigma"), arg("epsilon")), "Constructs a new LPInteriorPointLongstep solver"))
    .def(init<const bob::math::LPInteriorPointLongstep&>((arg("self"), arg("solver")), "Copy constructs a solver"))
    .def(self == self)
    .def(self != self)
    .def("is_in_v", &is_in_vinf, (arg("self"), arg("x"), arg("mu"), arg("gamma")), "Check if a primal-dual point (x,lambda,mu) belongs to the V-inf neighborhood of the central path")
    .add_property("gamma", &bob::math::LPInteriorPointLongstep::getGamma, &bob::math::LPInteriorPointLongstep::setGamma, "The value gamma used to define a V-inf neighborhood")
    .add_property("sigma", &bob::math::LPInteriorPointLongstep::getSigma, &bob::math::LPInteriorPointLongstep::setSigma, "The value sigma used to define a V-inf neighborhood")
  ;
}
**/
