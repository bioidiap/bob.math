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
#include <bob.math/LPInteriorPoint.h>

#include <bob.blitz/cppapi.h>
#include <bob.blitz/cleanup.h>
#include <structmember.h>

#include <bob.extension/documentation.h>

/************************************************
 * Implementation of LPInteriorPoint base class *
 ************************************************/

static auto s_lpinteriorpoint = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".LPInteriorPoint",
  "Base class to solve a linear program using interior point methods.",
  "For more details about the algorithms,please refer to the following book: *'Primal-Dual Interior-Point Methods', Stephen J. Wright, ISBN: 978-0898713824, Chapter 5, 'Path-Following Algorithms'*.\n\n"
  ".. warning:: You cannot instantiate an object of this type directly, you must use it through one of the inherited types.\n\n"
  "The primal linear program (LP) is defined as follows:\n\n"
  ".. math:: \\min c^T*x \\text{, s.t. } A*x=b, x>=0\n\n"
  "The dual formulation is:\n\n"
  ".. math:: \\min b^T*\\lambda \\text{, s.t. } A^T*\\lambda+\\mu=c"
);

/* Type definition for PyBobMathLpInteriorPointObject */
typedef struct {
  PyObject_HEAD

  /* Type-specific fields go here. */
  bob::math::LPInteriorPoint* base;

} PyBobMathLpInteriorPointObject;


static int PyBobMathLpInteriorPoint_init(PyBobMathLpInteriorPointObject* self, PyObject*, PyObject*) {

  PyErr_Format(PyExc_NotImplementedError, "cannot initialize object of base type `%s' - use one of the inherited classes", Py_TYPE(self)->tp_name);
  return -1;

}

static auto s_M = bob::extension::VariableDoc(
  "m",
  "int",
  "The first dimension of the problem/A matrix"
);


static PyObject* PyBobMathLpInteriorPoint_getM (PyBobMathLpInteriorPointObject* self,
    void* /*closure*/) {
  return Py_BuildValue("n", self->base->getDimM());
}

static int PyBobMathLpInteriorPoint_setM (PyBobMathLpInteriorPointObject* self,
    PyObject* o, void* /*closure*/) {

  Py_ssize_t M = PyNumber_AsSsize_t(o, PyExc_OverflowError);
  if (PyErr_Occurred()) return -1;

  try {
    self->base->setDimM(M);
  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return -1;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot reset size M of `%s': unknown exception caught", Py_TYPE(self)->tp_name);
    return -1;
  }

  return 0;

}

static auto s_N = bob::extension::VariableDoc(
  "n",
  "int",
  "The second dimension of the problem/A matrix"
);

static PyObject* PyBobMathLpInteriorPoint_getN (PyBobMathLpInteriorPointObject* self,
    void* /*closure*/) {
  return Py_BuildValue("n", self->base->getDimN());
}

static int PyBobMathLpInteriorPoint_setN (PyBobMathLpInteriorPointObject* self,
    PyObject* o, void* /*closure*/) {

  Py_ssize_t N = PyNumber_AsSsize_t(o, PyExc_OverflowError);
  if (PyErr_Occurred()) return -1;

  try {
    self->base->setDimN(N);
  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return -1;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot reset size N of `%s': unknown exception caught", Py_TYPE(self)->tp_name);
    return -1;
  }

  return 0;

}

static auto s_epsilon = bob::extension::VariableDoc(
  "epsilon",
  "float",
  "The precision to determine whether an equality constraint is fulfilled or not"
);

static PyObject* PyBobMathLpInteriorPoint_getEpsilon (PyBobMathLpInteriorPointObject* self, void* /*closure*/) {
  return Py_BuildValue("d", self->base->getEpsilon());
}

static int PyBobMathLpInteriorPoint_setEpsilon (PyBobMathLpInteriorPointObject* self,
    PyObject* o, void* /*closure*/) {

  double e = PyFloat_AsDouble(o);
  if (PyErr_Occurred()) return -1;

  try {
    self->base->setEpsilon(e);
  }
  catch (std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
    return -1;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot reset `epsilon' of `%s': unknown exception caught", Py_TYPE(self)->tp_name);
    return -1;
  }

  return 0;

}

static auto s_lambda = bob::extension::VariableDoc(
  "lambda_",
  "float",
  "The value of the :math:`\\lambda` dual variable (read-only)"
);

static PyObject* PyBobMathLpInteriorPoint_lambda (PyBobMathLpInteriorPointObject* self) {
  Py_ssize_t length = self->base->getDimM();
  PyObject* retval = PyBlitzArray_SimpleNew(NPY_FLOAT64, 1, &length);
  if (!retval) return 0;

  blitz::Array<double,1>* wrapper = PyBlitzArrayCxx_AsBlitz<double,1>
    (reinterpret_cast<PyBlitzArrayObject*>(retval));
  (*wrapper) = self->base->getLambda();

  return PyBlitzArray_NUMPY_WRAP(retval);
}

static auto s_mu = bob::extension::VariableDoc(
  "mu",
  "float",
  "The value of the :math:`\\mu` dual variable (read-only)"
);

static PyObject* PyBobMathLpInteriorPoint_mu (PyBobMathLpInteriorPointObject* self) {
  Py_ssize_t length = self->base->getDimN();
  PyObject* retval = PyBlitzArray_SimpleNew(NPY_FLOAT64, 1, &length);
  if (!retval) return 0;

  blitz::Array<double,1>* wrapper = PyBlitzArrayCxx_AsBlitz<double,1>
    (reinterpret_cast<PyBlitzArrayObject*>(retval));
  (*wrapper) = self->base->getMu();

  return PyBlitzArray_NUMPY_WRAP(retval);
}

static PyGetSetDef PyBobMathLpInteriorPoint_getseters[] = {
    {
      s_M.name(),
      (getter)PyBobMathLpInteriorPoint_getM,
      (setter)PyBobMathLpInteriorPoint_setM,
      s_M.doc(),
      0
    },
    {
      s_N.name(),
      (getter)PyBobMathLpInteriorPoint_getN,
      (setter)PyBobMathLpInteriorPoint_setN,
      s_N.doc(),
      0
    },
    {
      s_epsilon.name(),
      (getter)PyBobMathLpInteriorPoint_getEpsilon,
      (setter)PyBobMathLpInteriorPoint_setEpsilon,
      s_epsilon.doc(),
      0
    },
    {
      s_lambda.name(),
      (getter)PyBobMathLpInteriorPoint_lambda,
      0,
      s_lambda.doc(),
      0
    },
    {
      s_mu.name(),
      (getter)PyBobMathLpInteriorPoint_mu,
      0,
      s_mu.doc(),
      0
    },
    {0}  /* Sentinel */
};

static auto s_reset = bob::extension::FunctionDoc(
    "reset",
    "Resets the size of the problem (M and N correspond to the dimensions of the A matrix)"
  )
  .add_prototype("M, N")
  .add_parameter("M", "int", "The new first dimension of the problem/A matrix")
  .add_parameter("N", "int", "The new second dimension of the problem/A matrix")
;

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
    PyErr_Format(PyExc_RuntimeError, "cannot reset `%s': unknown exception caught", Py_TYPE(self)->tp_name);
    return 0;
  }

  Py_RETURN_NONE;

}

static auto s_solve = bob::extension::FunctionDoc(
    "solve",
    "Solves an LP problem"
  )
  .add_prototype("A, b, c, x0, lambda, mu", "x")
  .add_parameter("lambda", "?, optional", ".. todo:: Document parameter labmda")
  .add_parameter("mu", "?, optional", ".. todo:: Document parameter mu")
;


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

  //protects acquired resources through this scope
  auto A_ = make_safe(A);
  auto b_ = make_safe(b);
  auto c_ = make_safe(c);
  auto x0_ = make_safe(x0);
  auto lambda_ = make_xsafe(lambda);
  auto mu_ = make_xsafe(mu);

  if (A->type_num != NPY_FLOAT64 || A->ndim != 2) {
    PyErr_SetString(PyExc_TypeError, "Linear program solver only supports 64-bit floats 2D arrays for input vector `A'");
    return 0;
  }

  if (b->type_num != NPY_FLOAT64 || b->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "Linear program solver only supports 64-bit floats 1D arrays for input vector `b'");
    return 0;
  }

  if (c->type_num != NPY_FLOAT64 || c->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "Linear program solver only supports 64-bit floats 1D arrays for input vector `c'");
    return 0;
  }

  if (x0->type_num != NPY_FLOAT64 || x0->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "Linear program solver only supports 64-bit floats 1D arrays for input vector `x0'");
    return 0;
  }

  if (lambda && (lambda->type_num != NPY_FLOAT64 || lambda->ndim != 1)) {
    PyErr_SetString(PyExc_TypeError, "Linear program solver only supports 64-bit floats 1D arrays for input vector `lambda'");
    return 0;
  }

  if (mu && (mu->type_num != NPY_FLOAT64 || mu->ndim != 1)) {
    PyErr_SetString(PyExc_TypeError, "Linear program solver only supports 64-bit floats 1D arrays for input vector `mu'");
    return 0;
  }

  if ( (lambda && !mu) || (mu && !lambda) ) {
    PyErr_SetString(PyExc_RuntimeError, "Linear program solver requires none or both `mu' and `lambda' - you cannot just specify one of them");
    return 0;
  }

  /* This is where the output is going to be stored */
  PyObject* retval = PyBlitzArray_SimpleNew(NPY_FLOAT64, x0->ndim, x0->shape);
  if (!retval) {
    return 0;
  }
  blitz::Array<double,1>* wrapper = PyBlitzArrayCxx_AsBlitz<double,1>(reinterpret_cast<PyBlitzArrayObject*>(retval));
  (*wrapper) = *PyBlitzArrayCxx_AsBlitz<double,1>(x0);

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
    Py_DECREF(retval);
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return 0;
  }
  catch (...) {
    Py_DECREF(retval);
    PyErr_Format(PyExc_RuntimeError, "cannot solve `%s': unknown exception caught", Py_TYPE(self)->tp_name);
    return 0;
  }

  /* We only "return" the first half of the `x' vector */
  (reinterpret_cast<PyBlitzArrayObject*>(retval))->shape[0] /= 2;
  return PyBlitzArray_NUMPY_WRAP(retval);

}

static auto s_is_feasible = bob::extension::FunctionDoc(
    "is_feasible",
    "Checks if a primal-dual point (x, lambda, mu) belongs to the set of feasible points (i.e. fulfills the constraints)."
  )
  .add_prototype("A, b, c, x, lambda, mu", "test")
  .add_return("test", "bool", "``True`` if (x, labmda, mu) belongs to the set of feasible points, otherwise ``False``")
;

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

  //protects acquired resources through this scope
  auto A_ = make_safe(A);
  auto b_ = make_safe(b);
  auto c_ = make_safe(c);
  auto x_ = make_safe(x);
  auto lambda_ = make_safe(lambda);
  auto mu_ = make_safe(mu);

  if (A->type_num != NPY_FLOAT64 || A->ndim != 2) {
    PyErr_SetString(PyExc_TypeError, "Linear program is_feasible only supports 64-bit floats 2D arrays for input vector `A'");
    return 0;
  }

  if (b->type_num != NPY_FLOAT64 || b->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "Linear program is_feasible only supports 64-bit floats 1D arrays for input vector `b'");
    return 0;
  }

  if (c->type_num != NPY_FLOAT64 || c->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "Linear program is_feasible only supports 64-bit floats 1D arrays for input vector `c'");
    return 0;
  }

  if (x->type_num != NPY_FLOAT64 || x->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "Linear program is_feasible only supports 64-bit floats 1D arrays for input vector `x0'");
    return 0;
  }

  if (lambda->type_num != NPY_FLOAT64 || lambda->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "Linear program is_feasible only supports 64-bit floats 1D arrays for input vector `lambda'");
    return 0;
  }

  if (mu->type_num != NPY_FLOAT64 || mu->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "Linear program is_feasible only supports 64-bit floats 1D arrays for input vector `mu'");
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
    PyErr_Format(PyExc_RuntimeError, "cannot check feasibility of `%s': unknown exception caught", Py_TYPE(self)->tp_name);
    return 0;
  }

  if (feasible) Py_RETURN_TRUE;
  Py_RETURN_FALSE;

}

static auto s_is_in_v = bob::extension::FunctionDoc(
    "is_in_v",
    "Checks if a primal-dual point (x, lambda, mu) belongs to the V2 neighborhood of the central path.",
    ".. todo:: This documentation seems wrong since lambda is not in the list of parameters."
  )
  .add_prototype("x, mu, theta", "test")
  .add_return("test", "bool", "``True`` if (x, labmda, mu) belongs to the V2 neighborhood of the central path, otherwise ``False``")
;


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

  //protects acquired resources through this scope
  auto x_ = make_safe(x);
  auto mu_ = make_safe(mu);

  if (x->type_num != NPY_FLOAT64 || x->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "Linear program is_in_v only supports 64-bit floats 1D arrays for input vector `x0'");
    return 0;
  }

  if (mu->type_num != NPY_FLOAT64 || mu->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "Linear program is_in_v only supports 64-bit floats 1D arrays for input vector `mu'");
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
    PyErr_Format(PyExc_RuntimeError, "cannot check if point is in V at `%s': unknown exception caught", Py_TYPE(self)->tp_name);
    return 0;
  }

  if (in_v) Py_RETURN_TRUE;
  Py_RETURN_FALSE;

}

static auto s_is_in_v_s = bob::extension::FunctionDoc(
    "is_in_v_s",
    "Checks if a primal-dual point (x,lambda,mu) belongs to the V neighborhood of the central path and the set of feasible points."
  )
  .add_prototype("A, b, c, x, lambda, mu", "test")
  .add_return("test", "bool", "``True`` if (x, labmda, mu) belongs to the V neighborhood of the central path and the set of feasible points, otherwise ``False``")
;

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

  //protects acquired resources through this scope
  auto A_ = make_safe(A);
  auto b_ = make_safe(b);
  auto c_ = make_safe(c);
  auto x_ = make_safe(x);
  auto lambda_ = make_safe(lambda);
  auto mu_ = make_safe(mu);

  if (A->type_num != NPY_FLOAT64 || A->ndim != 2) {
    PyErr_SetString(PyExc_TypeError, "Linear program is_in_v_s only supports 64-bit floats 2D arrays for input vector `A'");
    return 0;
  }

  if (b->type_num != NPY_FLOAT64 || b->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "Linear program is_in_v_s only supports 64-bit floats 1D arrays for input vector `b'");
    return 0;
  }

  if (c->type_num != NPY_FLOAT64 || c->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "Linear program is_in_v_s only supports 64-bit floats 1D arrays for input vector `c'");
    return 0;
  }

  if (x->type_num != NPY_FLOAT64 || x->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "Linear program is_in_v_s only supports 64-bit floats 1D arrays for input vector `x0'");
    return 0;
  }

  if (lambda->type_num != NPY_FLOAT64 || lambda->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "Linear program is_in_v_s only supports 64-bit floats 1D arrays for input vector `lambda'");
    return 0;
  }

  if (mu->type_num != NPY_FLOAT64 || mu->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "Linear program is_in_v_s only supports 64-bit floats 1D arrays for input vector `mu'");
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
    PyErr_Format(PyExc_RuntimeError, "cannot check if point is in VS at `%s': unknown exception caught", Py_TYPE(self)->tp_name);
    return 0;
  }

  if (in_v_s) Py_RETURN_TRUE;
  Py_RETURN_FALSE;

}

static auto s_initialize_dual_lambda_mu = bob::extension::FunctionDoc(
    "initialize_dual_lambda_mu",
    "Initializes the dual variables ``lambda`` and ``mu`` by minimizing the logarithmic barrier function."
  )
  .add_prototype("A, c")
;

static PyObject* PyBobMathLpInteriorPoint_initialize_dual_lambda_mu
(PyBobMathLpInteriorPointObject* self, PyObject *args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"A", "c", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* A = 0;
  PyBlitzArrayObject* c = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&", kwlist,
        &PyBlitzArray_Converter, &A,
        &PyBlitzArray_Converter, &c
        )) return 0;

  //protects acquired resources through this scope
  auto A_ = make_safe(A);
  auto c_ = make_safe(c);

  if (A->type_num != NPY_FLOAT64 || A->ndim != 2) {
    PyErr_SetString(PyExc_TypeError, "Linear program initialize_dual_lambda_mu only supports 64-bit floats 2D arrays for input vector `A'");
    return 0;
  }

  if (c->type_num != NPY_FLOAT64 || c->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "Linear program initialize_dual_lambda_mu only supports 64-bit floats 1D arrays for input vector `c'");
    return 0;
  }

  try {
    self->base->initializeDualLambdaMu(
        *PyBlitzArrayCxx_AsBlitz<double,2>(A),
        *PyBlitzArrayCxx_AsBlitz<double,1>(c)
        );
  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return 0;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot initialize dual lambda-mu at `%s': unknown exception caught", Py_TYPE(self)->tp_name);
    return 0;
  }

  Py_RETURN_NONE;

}

static PyMethodDef PyBobMathLpInteriorPoint_methods[] = {
    {
      s_reset.name(),
      (PyCFunction)PyBobMathLpInteriorPoint_reset,
      METH_VARARGS|METH_KEYWORDS,
      s_reset.doc()
    },
    {
      s_solve.name(),
      (PyCFunction)PyBobMathLpInteriorPoint_solve,
      METH_VARARGS|METH_KEYWORDS,
      s_solve.doc()
    },
    {
      s_is_feasible.name(),
      (PyCFunction)PyBobMathLpInteriorPoint_is_feasible,
      METH_VARARGS|METH_KEYWORDS,
      s_is_feasible.doc()
    },
    {
      s_is_in_v.name(),
      (PyCFunction)PyBobMathLpInteriorPoint_is_in_v,
      METH_VARARGS|METH_KEYWORDS,
      s_is_in_v.doc()
    },
    {
      s_is_in_v_s.name(),
      (PyCFunction)PyBobMathLpInteriorPoint_is_in_v_s,
      METH_VARARGS|METH_KEYWORDS,
      s_is_in_v_s.doc()
    },
    {
      s_initialize_dual_lambda_mu.name(),
      (PyCFunction)PyBobMathLpInteriorPoint_initialize_dual_lambda_mu,
      METH_VARARGS|METH_KEYWORDS,
      s_initialize_dual_lambda_mu.doc()
    },
    {0}  /* Sentinel */
};

static int PyBobMathLpInteriorPoint_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobMathLpInteriorPoint_Type));
}

static PyObject* PyBobMathLpInteriorPoint_RichCompare (PyBobMathLpInteriorPointObject* self, PyObject* other, int op) {

  if (!PyBobMathLpInteriorPoint_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'",
        Py_TYPE(self)->tp_name, other->ob_type->tp_name);
    return 0;
  }

  PyBobMathLpInteriorPointObject* other_ = reinterpret_cast<PyBobMathLpInteriorPointObject*>(other);

  switch (op) {
    case Py_EQ:
      if (*(self->base) == *(other_->base)) Py_RETURN_TRUE;
      Py_RETURN_FALSE;
      break;
    case Py_NE:
      if (*(self->base) != *(other_->base)) Py_RETURN_TRUE;
      Py_RETURN_FALSE;
      break;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }

}

PyTypeObject PyBobMathLpInteriorPoint_Type = {
    PyVarObject_HEAD_INIT(0, 0)
    s_lpinteriorpoint.name(),                          /*tp_name*/
    sizeof(PyBobMathLpInteriorPointObject),            /*tp_basicsize*/
    0,                                                 /*tp_itemsize*/
    0,                                                 /*tp_dealloc*/
    0,                                                 /*tp_print*/
    0,                                                 /*tp_getattr*/
    0,                                                 /*tp_setattr*/
    0,                                                 /*tp_compare*/
    0,                                                 /*tp_repr*/
    0,                                                 /*tp_as_number*/
    0,                                                 /*tp_as_sequence*/
    0,                                                 /*tp_as_mapping*/
    0,                                                 /*tp_hash */
    0,                                                 /*tp_call*/
    0,                                                 /*tp_str*/
    0,                                                 /*tp_getattro*/
    0,                                                 /*tp_setattro*/
    0,                                                 /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,          /*tp_flags*/
    s_lpinteriorpoint.doc(),                           /* tp_doc */
    0,		                                             /* tp_traverse */
    0,		                                             /* tp_clear */
    (richcmpfunc)PyBobMathLpInteriorPoint_RichCompare, /* tp_richcompare */
    0,		                                             /* tp_weaklistoffset */
    0,		                                             /* tp_iter */
    0,		                                             /* tp_iternext */
    PyBobMathLpInteriorPoint_methods,                  /* tp_methods */
    0,                                                 /* tp_members */
    PyBobMathLpInteriorPoint_getseters,                /* tp_getset */
    0,                                                 /* tp_base */
    0,                                                 /* tp_dict */
    0,                                                 /* tp_descr_get */
    0,                                                 /* tp_descr_set */
    0,                                                 /* tp_dictoffset */
    (initproc)PyBobMathLpInteriorPoint_init,           /* tp_init */
    0,                                                 /* tp_alloc */
    0,                                                 /* tp_new */
};

/****************************************************
 * Implementation of LPInteriorPointShortstep class *
 ****************************************************/

const auto s_lpinteriorpointshortstep = bob::extension::ClassDoc(
    BOB_EXT_MODULE_PREFIX ".LPInteriorPointShortstep",
    "A Linear Program solver based on a short step interior point method.\n"
    "See :py:class:`LPInteriorPoint` for more details on the base class."
  )
  .add_constructor(bob::extension::FunctionDoc(
      "LPInteriorPointShortstep",
      "Objects of this class can be initialized in two different ways: "
      "a detailed constructor with the parameters described below or "
      "a copy constructor that deep-copies the input object and creates a new object (**not** a new reference to the same object)."
    )
    .add_prototype("M, N, theta, epsilon", "")
    .add_prototype("solver", "")
    .add_parameter("M", "int", "first dimension of the A matrix")
    .add_parameter("N", "int", "second dimension of the A matrix")
    .add_parameter("theta", "float", "The value defining the size of the V2 neighborhood")
    .add_parameter("epsilon", "float", "The precision to determine whether an equality constraint is fulfilled or not.")
    .add_parameter("solver", "LPInteriorPointShortstep", "The solver to make a deep copy of")
  )
  .highlight(s_solve)
  .highlight(s_mu)
  .highlight(s_lambda)
;

/* Type definition for PyBobMathLpInteriorPointObject */
typedef struct {
  PyBobMathLpInteriorPointObject parent;

  /* Type-specific fields go here. */
  bob::math::LPInteriorPointShortstep* base;

} PyBobMathLpInteriorPointShortstepObject;

static int PyBobMathLpInteriorPointShortstep_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobMathLpInteriorPointShortstep_Type));
}

static int PyBobMathLpInteriorPointShortstep_init1(PyBobMathLpInteriorPointShortstepObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"solver", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* solver = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &solver)) return -1;

  if (!PyBobMathLpInteriorPointShortstep_Check(solver)) {
    PyErr_Format(PyExc_TypeError, "copy-constructor for %s requires an object of the same type, not %s", s_lpinteriorpointshortstep.name(), solver->ob_type->tp_name);
    return -1;
  }

  PyBobMathLpInteriorPointShortstepObject* other = reinterpret_cast<PyBobMathLpInteriorPointShortstepObject*>(solver);

  try {
    self->base = new bob::math::LPInteriorPointShortstep(*other->base);
  }
  catch (std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot deep-copy object of type %s: unknown exception caught", s_lpinteriorpointshortstep.name());
  }

  self->parent.base = self->base;

  if (PyErr_Occurred()) return -1;

  return 0;

}

static int PyBobMathLpInteriorPointShortstep_init4(PyBobMathLpInteriorPointShortstepObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"M", "N", "theta", "epsilon", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  Py_ssize_t M = 0;
  Py_ssize_t N = 0;
  double theta = 0.;
  double epsilon = 0.;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "nndd", kwlist,
        &M, &N, &theta, &epsilon)) return -1;

  try {
    self->base = new bob::math::LPInteriorPointShortstep(M, N, theta, epsilon);
  }
  catch (std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
    return -1;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot construct object of type %s: unknown exception caught", s_lpinteriorpointshortstep.name());
    return -1;
  }

  self->parent.base = self->base;

  return 0;

}

static int PyBobMathLpInteriorPointShortstep_init(PyBobMathLpInteriorPointShortstepObject* self, PyObject* args, PyObject* kwds) {

  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwds?PyDict_Size(kwds):0);

  switch (nargs) {

    case 1:
      return PyBobMathLpInteriorPointShortstep_init1(self, args, kwds);
      break;

    case 4:
      return PyBobMathLpInteriorPointShortstep_init4(self, args, kwds);
      break;

    default:

      PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - %s requires 1 or 4 arguments, but you provided %" PY_FORMAT_SIZE_T "d (see help)", s_lpinteriorpointshortstep.name(), nargs);

  }

  return -1;

}

static void PyBobMathLpInteriorPointShortstep_delete (PyBobMathLpInteriorPointShortstepObject* self) {

  delete self->base;
  self->parent.base = 0;
  self->base = 0;
  Py_TYPE(&self->parent)->tp_free((PyObject*)self);

}

static auto s_theta = bob::extension::VariableDoc(
  "theta",
  "float",
  "The value theta used to define a V2 neighborhood"
);

static PyObject* PyBobMathLpInteriorPointShortstep_getTheta (PyBobMathLpInteriorPointShortstepObject* self, void* /*closure*/) {
  return Py_BuildValue("d", self->base->getTheta());
}

static int PyBobMathLpInteriorPointShortstep_setTheta (PyBobMathLpInteriorPointShortstepObject* self,
    PyObject* o, void* /*closure*/) {

  double e = PyFloat_AsDouble(o);
  if (PyErr_Occurred()) return -1;

  try {
    self->base->setTheta(e);
  }
  catch (std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
    return -1;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot reset `theta' of %s: unknown exception caught", s_lpinteriorpointshortstep.name());
    return -1;
  }

  return 0;

}

static PyGetSetDef PyBobMathLpInteriorPointShortstep_getseters[] = {
    {
      s_theta.name(),
      (getter)PyBobMathLpInteriorPointShortstep_getTheta,
      (setter)PyBobMathLpInteriorPointShortstep_setTheta,
      s_theta.doc(),
      0
    },
    {0}  /* Sentinel */
};

static PyObject* PyBobMathLpInteriorPointShortstep_RichCompare
(PyBobMathLpInteriorPointShortstepObject* self, PyObject* other, int op) {

  if (!PyBobMathLpInteriorPointShortstep_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'",
        s_lpinteriorpointshortstep.name(), other->ob_type->tp_name);
    return 0;
  }

  PyBobMathLpInteriorPointShortstepObject* other_ = reinterpret_cast<PyBobMathLpInteriorPointShortstepObject*>(other);

  switch (op) {
    case Py_EQ:
      if (*(self->base) == *(other_->base)) Py_RETURN_TRUE;
      Py_RETURN_FALSE;
      break;
    case Py_NE:
      if (*(self->base) != *(other_->base)) Py_RETURN_TRUE;
      Py_RETURN_FALSE;
      break;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }

}


PyTypeObject PyBobMathLpInteriorPointShortstep_Type = {
    PyVarObject_HEAD_INIT(0, 0)
    s_lpinteriorpointshortstep.name(),                          /*tp_name*/
    sizeof(PyBobMathLpInteriorPointShortstepObject),            /*tp_basicsize*/
    0,                                                          /*tp_itemsize*/
    (destructor)PyBobMathLpInteriorPointShortstep_delete,       /*tp_dealloc*/
    0,                                                          /*tp_print*/
    0,                                                          /*tp_getattr*/
    0,                                                          /*tp_setattr*/
    0,                                                          /*tp_compare*/
    0,                                                          /*tp_repr*/
    0,                                                          /*tp_as_number*/
    0,                                                          /*tp_as_sequence*/
    0,                                                          /*tp_as_mapping*/
    0,                                                          /*tp_hash */
    0,                                                          /*tp_call*/
    0,                                                          /*tp_str*/
    0,                                                          /*tp_getattro*/
    0,                                                          /*tp_setattro*/
    0,                                                          /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,                   /*tp_flags*/
    s_lpinteriorpointshortstep.doc(),                           /* tp_doc */
    0,		                                                      /* tp_traverse */
    0,		                                                      /* tp_clear */
    (richcmpfunc)PyBobMathLpInteriorPointShortstep_RichCompare, /* tp_richcompare */
    0,		                                                      /* tp_weaklistoffset */
    0,		                                                      /* tp_iter */
    0,		                                                      /* tp_iternext */
    0,                                                          /* tp_methods */
    0,                                                          /* tp_members */
    PyBobMathLpInteriorPointShortstep_getseters,                /* tp_getset */
    0,                                                          /* tp_base */
    0,                                                          /* tp_dict */
    0,                                                          /* tp_descr_get */
    0,                                                          /* tp_descr_set */
    0,                                                          /* tp_dictoffset */
    (initproc)PyBobMathLpInteriorPointShortstep_init,           /* tp_init */
    0,                                                          /* tp_alloc */
    0,                                                          /* tp_new */
};

/*************************************************************
 * Implementation of LPInteriorPointPredictorCorrector class *
 *************************************************************/

static auto s_lpinteriorpointpredictorcorrector = bob::extension::ClassDoc(
    BOB_EXT_MODULE_PREFIX ".LPInteriorPointPredictorCorrector",
    "A Linear Program solver based on a predictor-corrector interior point method.",
    "See :py:class:`LPInteriorPoint` for more details on the base class."
  )
  .add_constructor(bob::extension::FunctionDoc(
      "LPInteriorPointPredictorCorrector",
      "Objects of this class can be initialized in two different ways: "
      "a detailed constructor with the parameters described below or "
      "a copy constructor, that deep-copies the input object and creates a new object (**not** a new reference to the same object)."
    )
    .add_prototype("M, N, theta_pred, theta_corr, epsilon", "")
    .add_prototype("solver", "")
    .add_parameter("M", "int", "first dimension of the A matrix")
    .add_parameter("N", "int", "second dimension of the A matrix")
    .add_parameter("theta_pred", "float", "the value theta_pred used to define a V2 neighborhood")
    .add_parameter("theta_corr", "float", "the value theta_corr used to define a V2 neighborhood")
    .add_parameter("epsilon", "float", "the precision to determine whether an equality constraint is fulfilled or not")
    .add_parameter("solver", "LPInteriorPointPredictorCorrector", "the solver to make a deep copy of")
  )
;

/* Type definition for PyBobMathLpInteriorPointObject */
typedef struct {
  PyBobMathLpInteriorPointObject parent;

  /* Type-specific fields go here. */
  bob::math::LPInteriorPointPredictorCorrector* base;

} PyBobMathLpInteriorPointPredictorCorrectorObject;

static int PyBobMathLpInteriorPointPredictorCorrector_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobMathLpInteriorPointPredictorCorrector_Type));
}

static int PyBobMathLpInteriorPointPredictorCorrector_init1(PyBobMathLpInteriorPointPredictorCorrectorObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"solver", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* solver = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &solver)) return -1;

  if (!PyBobMathLpInteriorPointPredictorCorrector_Check(solver)) {
    PyErr_Format(PyExc_TypeError, "copy-constructor for %s requires an object of the same type, not %s", s_lpinteriorpointpredictorcorrector.name(), solver->ob_type->tp_name);
    return -1;
  }

  PyBobMathLpInteriorPointPredictorCorrectorObject* other = reinterpret_cast<PyBobMathLpInteriorPointPredictorCorrectorObject*>(solver);

  try {
    self->base = new bob::math::LPInteriorPointPredictorCorrector(*other->base);
  }
  catch (std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot deep-copy object of type %s: unknown exception caught", s_lpinteriorpointpredictorcorrector.name());
  }

  self->parent.base = self->base;

  if (PyErr_Occurred()) return -1;

  return 0;

}

static int PyBobMathLpInteriorPointPredictorCorrector_init5(PyBobMathLpInteriorPointPredictorCorrectorObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"M", "N", "theta_pred", "theta_corr", "epsilon", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  Py_ssize_t M = 0;
  Py_ssize_t N = 0;
  double theta_pred = 0.;
  double theta_corr = 0.;
  double epsilon = 0.;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "nnddd", kwlist,
        &M, &N, &theta_pred, &theta_corr, &epsilon)) return -1;

  try {
    self->base = new bob::math::LPInteriorPointPredictorCorrector(M, N, theta_pred, theta_corr, epsilon);
  }
  catch (std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
    return -1;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot construct object of type %s: unknown exception caught", s_lpinteriorpointpredictorcorrector.name());
    return -1;
  }

  self->parent.base = self->base;

  return 0;

}

static int PyBobMathLpInteriorPointPredictorCorrector_init(PyBobMathLpInteriorPointPredictorCorrectorObject* self, PyObject* args, PyObject* kwds) {

  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwds?PyDict_Size(kwds):0);

  switch (nargs) {

    case 1:
      return PyBobMathLpInteriorPointPredictorCorrector_init1(self, args, kwds);
      break;

    case 5:
      return PyBobMathLpInteriorPointPredictorCorrector_init5(self, args, kwds);
      break;

    default:

      PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - %s requires 1 or 5 arguments, but you provided %" PY_FORMAT_SIZE_T "d (see help)", s_lpinteriorpointpredictorcorrector.name(), nargs);

  }

  return -1;

}

static void PyBobMathLpInteriorPointPredictorCorrector_delete (PyBobMathLpInteriorPointPredictorCorrectorObject* self) {

  delete self->base;
  self->parent.base = 0;
  self->base = 0;
  Py_TYPE(&self->parent)->tp_free((PyObject*)self);

}

static auto s_theta_pred = bob::extension::VariableDoc(
  "theta_pred",
  "float",
  "The value theta_pred used to define a V2 neighborhood"
);

static PyObject* PyBobMathLpInteriorPointPredictorCorrector_getThetaPred (PyBobMathLpInteriorPointPredictorCorrectorObject* self, void* /*closure*/) {
  return Py_BuildValue("d", self->base->getThetaPred());
}

static int PyBobMathLpInteriorPointPredictorCorrector_setThetaPred (PyBobMathLpInteriorPointPredictorCorrectorObject* self, PyObject* o, void* /*closure*/) {

  double e = PyFloat_AsDouble(o);
  if (PyErr_Occurred()) return -1;

  try {
    self->base->setThetaPred(e);
  }
  catch (std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
    return -1;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot reset `theta_pred' of %s: unknown exception caught", s_lpinteriorpointpredictorcorrector.name());
    return -1;
  }

  return 0;

}

static auto s_theta_corr = bob::extension::VariableDoc(
  "theta_corr",
  "float",
  "The value theta_corr used to define a V2 neighborhood"
);

static PyObject* PyBobMathLpInteriorPointPredictorCorrector_getThetaCorr (PyBobMathLpInteriorPointPredictorCorrectorObject* self, void* /*closure*/) {
  return Py_BuildValue("d", self->base->getThetaCorr());
}

static int PyBobMathLpInteriorPointPredictorCorrector_setThetaCorr (PyBobMathLpInteriorPointPredictorCorrectorObject* self, PyObject* o, void* /*closure*/) {

  double e = PyFloat_AsDouble(o);
  if (PyErr_Occurred()) return -1;

  try {
    self->base->setThetaCorr(e);
  }
  catch (std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
    return -1;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot reset `theta_corr' of %s: unknown exception caught", s_lpinteriorpointpredictorcorrector.name());
    return -1;
  }

  return 0;

}

static PyGetSetDef PyBobMathLpInteriorPointPredictorCorrector_getseters[] = {
    {
      s_theta_pred.name(),
      (getter)PyBobMathLpInteriorPointPredictorCorrector_getThetaPred,
      (setter)PyBobMathLpInteriorPointPredictorCorrector_setThetaPred,
      s_theta_pred.doc(),
      0
    },
    {
      s_theta_corr.name(),
      (getter)PyBobMathLpInteriorPointPredictorCorrector_getThetaCorr,
      (setter)PyBobMathLpInteriorPointPredictorCorrector_setThetaCorr,
      s_theta_corr.doc(),
      0
    },
    {0}  /* Sentinel */
};

static PyObject* PyBobMathLpInteriorPointPredictorCorrector_RichCompare
(PyBobMathLpInteriorPointPredictorCorrectorObject* self, PyObject* other, int op) {

  if (!PyBobMathLpInteriorPointPredictorCorrector_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'",
        s_lpinteriorpointpredictorcorrector.name(), other->ob_type->tp_name);
    return 0;
  }

  PyBobMathLpInteriorPointPredictorCorrectorObject* other_ = reinterpret_cast<PyBobMathLpInteriorPointPredictorCorrectorObject*>(other);

  switch (op) {
    case Py_EQ:
      if (*(self->base) == *(other_->base)) Py_RETURN_TRUE;
      Py_RETURN_FALSE;
      break;
    case Py_NE:
      if (*(self->base) != *(other_->base)) Py_RETURN_TRUE;
      Py_RETURN_FALSE;
      break;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }

}

PyTypeObject PyBobMathLpInteriorPointPredictorCorrector_Type = {
    PyVarObject_HEAD_INIT(0, 0)
    s_lpinteriorpointpredictorcorrector.name(),                 /*tp_name*/
    sizeof(PyBobMathLpInteriorPointPredictorCorrectorObject),   /*tp_basicsize*/
    0,                                                          /*tp_itemsize*/
    (destructor)PyBobMathLpInteriorPointPredictorCorrector_delete, /*tp_dealloc*/
    0,                                                          /*tp_print*/
    0,                                                          /*tp_getattr*/
    0,                                                          /*tp_setattr*/
    0,                                                          /*tp_compare*/
    0,                                                          /*tp_repr*/
    0,                                                          /*tp_as_number*/
    0,                                                          /*tp_as_sequence*/
    0,                                                          /*tp_as_mapping*/
    0,                                                          /*tp_hash */
    0,                                                          /*tp_call*/
    0,                                                          /*tp_str*/
    0,                                                          /*tp_getattro*/
    0,                                                          /*tp_setattro*/
    0,                                                          /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,                   /*tp_flags*/
    s_lpinteriorpointpredictorcorrector.doc(),                  /* tp_doc */
    0,		                                                      /* tp_traverse */
    0,		                                                      /* tp_clear */
    (richcmpfunc)PyBobMathLpInteriorPointPredictorCorrector_RichCompare, /* tp_richcompare */
    0,		                                                      /* tp_weaklistoffset */
    0,		                                                      /* tp_iter */
    0,		                                                      /* tp_iternext */
    0,                                                          /* tp_methods */
    0,                                                          /* tp_members */
    PyBobMathLpInteriorPointPredictorCorrector_getseters,       /* tp_getset */
    0,                                                          /* tp_base */
    0,                                                          /* tp_dict */
    0,                                                          /* tp_descr_get */
    0,                                                          /* tp_descr_set */
    0,                                                          /* tp_dictoffset */
    (initproc)PyBobMathLpInteriorPointPredictorCorrector_init,  /* tp_init */
    0,                                                          /* tp_alloc */
    0,                                                          /* tp_new */
};

/****************************************************
 * Implementation of LPInteriorPointLongstep class *
 ****************************************************/

static auto s_lpinteriorpointlongstep = bob::extension::ClassDoc(
    BOB_EXT_MODULE_PREFIX ".LPInteriorPointLongstep",
    "A Linear Program solver based on a long step interior point method.",
    "See :py:class:`LPInteriorPoint` for more details on the base class."
  )
  .add_constructor(bob::extension::FunctionDoc(
      "LPInteriorPointLongstep",
      "Objects of this class can be initialized in two different ways: "
      "a detailed constructor with the parameters described below or "
      "a copy constructor, that deep-copies the input object and creates a new object (**not** a new reference to the same object)"
    )
    .add_prototype("M, N, gamma, sigma, epsilon", "")
    .add_prototype("solver", "")
    .add_parameter("M", "int", "first dimension of the A matrix")
    .add_parameter("N", "int", "second dimension of the A matrix")
    .add_parameter("gamma", "float", "the value gamma used to define a V-inf neighborhood")
    .add_parameter("sigma", "float", "the value sigma used to define a V-inf neighborhood")
    .add_parameter("epsilon", "float", "the precision to determine whether an equality constraint is fulfilled or not")
    .add_parameter("solver", "LPInteriorPointLongstep", "the solver to make a deep copy of")
  )
;

typedef struct {
  PyBobMathLpInteriorPointObject parent;

  /* Type-specific fields go here. */
  bob::math::LPInteriorPointLongstep* base;

} PyBobMathLpInteriorPointLongstepObject;

static int PyBobMathLpInteriorPointLongstep_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobMathLpInteriorPointLongstep_Type));
}

static int PyBobMathLpInteriorPointLongstep_init1(PyBobMathLpInteriorPointLongstepObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"solver", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyObject* solver = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &solver)) return -1;

  if (!PyBobMathLpInteriorPointLongstep_Check(solver)) {
    PyErr_Format(PyExc_TypeError, "copy-constructor for %s requires an object of the same type, not %s", s_lpinteriorpointlongstep.name(), solver->ob_type->tp_name);
    return -1;
  }

  PyBobMathLpInteriorPointLongstepObject* other = reinterpret_cast<PyBobMathLpInteriorPointLongstepObject*>(solver);

  try {
    self->base = new bob::math::LPInteriorPointLongstep(*other->base);
  }
  catch (std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot deep-copy object of type %s: unknown exception caught", s_lpinteriorpointlongstep.name());
  }

  self->parent.base = self->base;

  if (PyErr_Occurred()) return -1;

  return 0;

}

static int PyBobMathLpInteriorPointLongstep_init5(PyBobMathLpInteriorPointLongstepObject* self, PyObject* args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"M", "N", "gamma", "sigma", "epsilon", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  Py_ssize_t M = 0;
  Py_ssize_t N = 0;
  double gamma = 0.;
  double sigma = 0.;
  double epsilon = 0.;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "nnddd", kwlist,
        &M, &N, &gamma, &sigma, &epsilon)) return -1;

  try {
    self->base = new bob::math::LPInteriorPointLongstep(M, N, gamma, sigma, epsilon);
  }
  catch (std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
    return -1;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot construct object of type %s: unknown exception caught", s_lpinteriorpointlongstep.name());
    return -1;
  }
  self->parent.base = self->base;

  return 0;

}

static int PyBobMathLpInteriorPointLongstep_init(PyBobMathLpInteriorPointLongstepObject* self, PyObject* args, PyObject* kwds) {

  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwds?PyDict_Size(kwds):0);

  switch (nargs) {

    case 1:
      return PyBobMathLpInteriorPointLongstep_init1(self, args, kwds);
      break;

    case 5:
      return PyBobMathLpInteriorPointLongstep_init5(self, args, kwds);
      break;

    default:

      PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - %s requires 1 or 5 arguments, but you provided %" PY_FORMAT_SIZE_T "d (see help)", s_lpinteriorpointlongstep.name(), nargs);

  }

  return -1;

}

static void PyBobMathLpInteriorPointLongstep_delete (PyBobMathLpInteriorPointLongstepObject* self) {

  delete self->base;
  self->parent.base = 0;
  self->base = 0;
  Py_TYPE(&self->parent)->tp_free((PyObject*)self);

}

static auto s_gamma = bob::extension::VariableDoc(
  "gamma",
  "float",
  "The value gamma used to define a V-Inf neighborhood"
);

static PyObject* PyBobMathLpInteriorPointLongstep_getGamma (PyBobMathLpInteriorPointLongstepObject* self, void* /*closure*/) {
  return Py_BuildValue("d", self->base->getGamma());
}

static int PyBobMathLpInteriorPointLongstep_setGamma (PyBobMathLpInteriorPointLongstepObject* self,
    PyObject* o, void* /*closure*/) {

  double e = PyFloat_AsDouble(o);
  if (PyErr_Occurred()) return -1;

  try {
    self->base->setGamma(e);
  }
  catch (std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
    return -1;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot reset `gamma' of %s: unknown exception caught", s_lpinteriorpointlongstep.name());
    return -1;
  }

  return 0;

}

static auto s_sigma = bob::extension::VariableDoc(
  "sigma",
  "float",
  "The value sigma used to define a V-Inf neighborhood"
);

static PyObject* PyBobMathLpInteriorPointLongstep_getSigma (PyBobMathLpInteriorPointLongstepObject* self, void* /*closure*/) {
  return Py_BuildValue("d", self->base->getSigma());
}

static int PyBobMathLpInteriorPointLongstep_setSigma (PyBobMathLpInteriorPointLongstepObject* self,
    PyObject* o, void* /*closure*/) {

  double e = PyFloat_AsDouble(o);
  if (PyErr_Occurred()) return -1;

  try {
    self->base->setSigma(e);
  }
  catch (std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
    return -1;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot reset `sigma' of %s: unknown exception caught", s_lpinteriorpointlongstep.name());
    return -1;
  }

  return 0;

}

static PyGetSetDef PyBobMathLpInteriorPointLongstep_getseters[] = {
    {
      s_gamma.name(),
      (getter)PyBobMathLpInteriorPointLongstep_getGamma,
      (setter)PyBobMathLpInteriorPointLongstep_setGamma,
      s_gamma.doc(),
      0
    },
    {
      s_sigma.name(),
      (getter)PyBobMathLpInteriorPointLongstep_getSigma,
      (setter)PyBobMathLpInteriorPointLongstep_setSigma,
      s_sigma.doc(),
      0
    },
    {0}  /* Sentinel */
};

static PyObject* PyBobMathLpInteriorPointLongstep_RichCompare
(PyBobMathLpInteriorPointLongstepObject* self, PyObject* other, int op) {

  if (!PyBobMathLpInteriorPointLongstep_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'",
        s_lpinteriorpointlongstep.name(), other->ob_type->tp_name);
    return 0;
  }

  PyBobMathLpInteriorPointLongstepObject* other_ = reinterpret_cast<PyBobMathLpInteriorPointLongstepObject*>(other);

  switch (op) {
    case Py_EQ:
      if (*(self->base) == *(other_->base)) Py_RETURN_TRUE;
      Py_RETURN_FALSE;
      break;
    case Py_NE:
      if (*(self->base) != *(other_->base)) Py_RETURN_TRUE;
      Py_RETURN_FALSE;
      break;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }

}

static auto s_is_in_vinf = bob::extension::FunctionDoc(
    "is_in_v",
    "Checks if a primal-dual point (x, lambda, mu) belongs to the V-Inf neighborhood of the central path.",
    ".. todo:: This documentation looks wrong since lambda is not part of the parameters"
  )
  .add_prototype("x, mu, gamma", "test")
  .add_return("test", "bool", "``True`` if (x, lambda, mu) belong to the  V-Inf neighborhood of the central path, otherwise ``False``")
;

static PyObject* PyBobMathLpInteriorPoint_is_in_vinf
(PyBobMathLpInteriorPointObject* self, PyObject *args, PyObject* kwds) {

  /* Parses input arguments in a single shot */
  static const char* const_kwlist[] = {"x", "mu", "gamma", 0};
  static char** kwlist = const_cast<char**>(const_kwlist);

  PyBlitzArrayObject* x = 0;
  PyBlitzArrayObject* mu = 0;
  double gamma = 0.;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&O&d", kwlist,
        &PyBlitzArray_Converter, &x,
        &PyBlitzArray_Converter, &mu,
        &gamma
        )) return 0;

  //protects acquired resources through this scope
  auto x_ = make_safe(x);
  auto mu_ = make_safe(mu);

  if (x->type_num != NPY_FLOAT64 || x->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "Linear program is_in_vinf only supports 64-bit floats 1D arrays for input vector `x0'");
    return 0;
  }

  if (mu->type_num != NPY_FLOAT64 || mu->ndim != 1) {
    PyErr_SetString(PyExc_TypeError, "Linear program is_in_vinf only supports 64-bit floats 1D arrays for input vector `mu'");
    return 0;
  }

  bool in_vinf = false;

  try {
    in_vinf = self->base->isInV(
        *PyBlitzArrayCxx_AsBlitz<double,1>(x),
        *PyBlitzArrayCxx_AsBlitz<double,1>(mu),
        gamma
        );
  }
  catch (std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return 0;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot check if point is in V-Inf at `%s': unknown exception caught", Py_TYPE(self)->tp_name);
    return 0;
  }

  if (in_vinf) Py_RETURN_TRUE;
  Py_RETURN_FALSE;

}

static PyMethodDef PyBobMathLpInteriorPointLongstep_methods[] = {
    {
      s_is_in_vinf.name(),
      (PyCFunction)PyBobMathLpInteriorPoint_is_in_vinf,
      METH_VARARGS|METH_KEYWORDS,
      s_is_in_vinf.doc()
    },
    {0}  /* Sentinel */
};

PyTypeObject PyBobMathLpInteriorPointLongstep_Type = {
    PyVarObject_HEAD_INIT(0, 0)
    s_lpinteriorpointlongstep.name(),                           /*tp_name*/
    sizeof(PyBobMathLpInteriorPointLongstepObject),             /*tp_basicsize*/
    0,                                                          /*tp_itemsize*/
    (destructor)PyBobMathLpInteriorPointLongstep_delete,        /*tp_dealloc*/
    0,                                                          /*tp_print*/
    0,                                                          /*tp_getattr*/
    0,                                                          /*tp_setattr*/
    0,                                                          /*tp_compare*/
    0,                                                          /*tp_repr*/
    0,                                                          /*tp_as_number*/
    0,                                                          /*tp_as_sequence*/
    0,                                                          /*tp_as_mapping*/
    0,                                                          /*tp_hash */
    0,                                                          /*tp_call*/
    0,                                                          /*tp_str*/
    0,                                                          /*tp_getattro*/
    0,                                                          /*tp_setattro*/
    0,                                                          /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,                   /*tp_flags*/
    s_lpinteriorpointlongstep.doc(),                            /* tp_doc */
    0,		                                                      /* tp_traverse */
    0,		                                                      /* tp_clear */
    (richcmpfunc)PyBobMathLpInteriorPointLongstep_RichCompare,  /* tp_richcompare */
    0,		                                                      /* tp_weaklistoffset */
    0,		                                                      /* tp_iter */
    0,		                                                      /* tp_iternext */
    PyBobMathLpInteriorPointLongstep_methods,                   /* tp_methods */
    0,                                                          /* tp_members */
    PyBobMathLpInteriorPointLongstep_getseters,                 /* tp_getset */
    0,                                                          /* tp_base */
    0,                                                          /* tp_dict */
    0,                                                          /* tp_descr_get */
    0,                                                          /* tp_descr_set */
    0,                                                          /* tp_dictoffset */
    (initproc)PyBobMathLpInteriorPointLongstep_init,            /* tp_init */
    0,                                                          /* tp_alloc */
    0,                                                          /* tp_new */
};
