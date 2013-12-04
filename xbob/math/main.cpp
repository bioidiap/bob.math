/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue  3 Dec 14:13:22 2013 CET
 *
 * @brief Bindings to bob::math
 */

#define XBOB_MATH_MODULE
#include <xbob.math/api.h>

#ifdef NO_IMPORT_ARRAY
#undef NO_IMPORT_ARRAY
#endif
#include <xbob.blitz/capi.h>

#include "histogram.h"
#include "linsolve.h"

PyDoc_STRVAR(s_histogram_intersection_str, "histogram_intersection");
PyDoc_STRVAR(s_histogram_intersection_doc,
"histogram_intersection(h1, h2) -> scalar\n\
histogram_intersection(index_1, value_1, index_2, value_2) -> scalar\n\
\n\
Computes the histogram intersection between the given histograms, which\n\
might be of singular dimension only. The histogram intersection defines\n\
a similarity measure, so higher values are better.\n\
\n\
You can use this method in two different formats. The first interface\n\
accepts non-sparse histograms. The second interface accepts sparse\n\
histograms represented by index and values.\n\
"
);

PyDoc_STRVAR(s_chi_square_str, "chi_square");
PyDoc_STRVAR(s_chi_square_doc,
"chi_square(h1, h2) -> scalar\n\
chi_square(index_1, value_1, index_2, value_2) -> scalar\n\
\n\
Computes the chi square distance between the given histograms, which\n\
might be of singular dimension only. The chi square function is a \n\
distance measure, so lower values are better.\n\
\n\
You can use this method in two different formats. The first interface\n\
accepts non-sparse histograms. The second interface accepts sparse\n\
histograms represented by index and values.\n\
"
);

PyDoc_STRVAR(s_kullback_leibler_str, "kullback_leibler");
PyDoc_STRVAR(s_kullback_leibler_doc,
"kullback_leibler(h1, h2) -> scalar\n\
kullback_leibler(index_1, value_1, index_2, value_2) -> scalar\n\
\n\
Computes the Kullback-Leibler histogram divergence between the given\n\
histograms, which might be of singular dimension only. The\n\
Kullback-Leibler divergence is a distance measure, so lower values\n\
are better.\n\
\n\
You can use this method in two different formats. The first interface\n\
accepts non-sparse histograms. The second interface accepts sparse\n\
histograms represented by index and values.\n\
"
);

PyDoc_STRVAR(s_linsolve_str, "linsolve");
PyDoc_STRVAR(s_linsolve_doc,
"linsolve(A, b) -> array\n\
linsolve(A, x, b) -> None\n\
\n\
Solves the linear system :py:math:`Ax=b` and returns the result in ``x``.\n\
This method uses LAPACK's ``dgesv`` generic solver.\n\
\n\
You can use this method in two different formats. The first interface\n\
accepts the matrices ``A`` and ``b`` returning ``x``. The second one\n\
accepts a pre-allocated ``x`` matrix and sets it with the linear system\n\
solution.\n\
"
);

PyDoc_STRVAR(s_linsolve_nocheck_str, "linsolve_");
PyDoc_STRVAR(s_linsolve_nocheck_doc,
"linsolve_(A, b) -> array\n\
linsolve_(A, x, b) -> None\n\
\n\
Solves the linear system :py:math:`Ax=b` and returns the result in ``x``.\n\
This method uses LAPACK's ``dgesv`` generic solver.\n\
\n\
THIS VARIANT DOES NOT PERFORM ANY CHECKS ON THE INPUT MATRICES AND IS,\n\
FASTER THEN THE VARIANT NOT ENDING IN ``_``. Use it when you are sure\n\
your input matrices are well-behaved (contiguous, c-style, memory aligned).\n\
\n\
You can use this method in two different formats. The first interface\n\
accepts the matrices ``A`` and ``b`` returning ``x``. The second one\n\
accepts a pre-allocated ``x`` matrix and sets it with the linear system\n\
solution.\n\
"
);

PyDoc_STRVAR(s_linsolve_sympos_str, "linsolve_sympos");
PyDoc_STRVAR(s_linsolve_sympos_doc,
"linsolve_sympos(A, b) -> array\n\
linsolve_sympos(A, x, b) -> None\n\
\n\
Solves the linear system :py:math:`Ax=b` and returns the result in ``x``.\n\
This method uses LAPACK's ``dposv`` solver, assuming ``A`` is a symmetric.\n\
positive definite matrix.\n\
\n\
You can use this method in two different formats. The first interface\n\
accepts the matrices ``A`` and ``b`` returning ``x``. The second one\n\
accepts a pre-allocated ``x`` matrix and sets it with the linear system\n\
solution.\n\
"
);

PyDoc_STRVAR(s_linsolve_sympos_nocheck_str, "linsolve_sympos_");
PyDoc_STRVAR(s_linsolve_sympos_nocheck_doc,
"linsolve_sympos_(A, b) -> array\n\
linsolve_sympos_(A, x, b) -> None\n\
\n\
Solves the linear system :py:math:`Ax=b` and returns the result in ``x``.\n\
This method uses LAPACK's ``dposv`` solver, assuming ``A`` is a symmetric.\n\
positive definite matrix.\n\
\n\
THIS VARIANT DOES NOT PERFORM ANY CHECKS ON THE INPUT MATRICES AND IS,\n\
FASTER THEN THE VARIANT NOT ENDING IN ``_``. Use it when you are sure\n\
your input matrices are well-behaved (contiguous, c-style, memory aligned).\n\
\n\
You can use this method in two different formats. The first interface\n\
accepts the matrices ``A`` and ``b`` returning ``x``. The second one\n\
accepts a pre-allocated ``x`` matrix and sets it with the linear system\n\
solution.\n\
"
);

PyDoc_STRVAR(s_linsolve_cg_sympos_str, "linsolve_cg_sympos");
PyDoc_STRVAR(s_linsolve_cg_sympos_doc,
"linsolve_cg_sympos(A, b) -> array\n\
linsolve_cg_sympos(A, x, b) -> None\n\
\n\
Solves the linear system :py:math:`Ax=b` and returns the result in ``x``.\n\
This method solves the linear system via conjugate gradients and assumes\n\
``A`` is a symmetric positive definite matrix.\n\
\n\
You can use this method in two different formats. The first interface\n\
accepts the matrices ``A`` and ``b`` returning ``x``. The second one\n\
accepts a pre-allocated ``x`` matrix and sets it with the linear system\n\
solution.\n\
"
);

PyDoc_STRVAR(s_linsolve_cg_sympos_nocheck_str, "linsolve_cg_sympos_");
PyDoc_STRVAR(s_linsolve_cg_sympos_nocheck_doc,
"linsolve_cg_sympos_(A, b) -> array\n\
linsolve_cg_sympos_(A, x, b) -> None\n\
\n\
Solves the linear system :py:math:`Ax=b` and returns the result in ``x``.\n\
This method solves the linear system via conjugate gradients and assumes\n\
``A`` is a symmetric positive definite matrix.\n\
\n\
THIS VARIANT DOES NOT PERFORM ANY CHECKS ON THE INPUT MATRICES AND IS,\n\
FASTER THEN THE VARIANT NOT ENDING IN ``_``. Use it when you are sure\n\
your input matrices are well-behaved (contiguous, c-style, memory aligned).\n\
\n\
You can use this method in two different formats. The first interface\n\
accepts the matrices ``A`` and ``b`` returning ``x``. The second one\n\
accepts a pre-allocated ``x`` matrix and sets it with the linear system\n\
solution.\n\
"
);

static PyMethodDef module_methods[] = {
    {
      s_histogram_intersection_str,
      (PyCFunction)py_histogram_intersection,
      METH_VARARGS|METH_KEYWORDS,
      s_histogram_intersection_doc
    },
    {
      s_chi_square_str,
      (PyCFunction)py_chi_square,
      METH_VARARGS|METH_KEYWORDS,
      s_chi_square_doc
    },
    {
      s_kullback_leibler_str,
      (PyCFunction)py_kullback_leibler,
      METH_VARARGS|METH_KEYWORDS,
      s_kullback_leibler_doc
    },
    {
      s_linsolve_str,
      (PyCFunction)py_linsolve,
      METH_VARARGS|METH_KEYWORDS,
      s_linsolve_doc
    },
    {
      s_linsolve_nocheck_str,
      (PyCFunction)py_linsolve_nocheck,
      METH_VARARGS|METH_KEYWORDS,
      s_linsolve_nocheck_doc
    },
    {
      s_linsolve_sympos_str,
      (PyCFunction)py_linsolve_sympos,
      METH_VARARGS|METH_KEYWORDS,
      s_linsolve_sympos_doc
    },
    {
      s_linsolve_sympos_nocheck_str,
      (PyCFunction)py_linsolve_sympos_nocheck,
      METH_VARARGS|METH_KEYWORDS,
      s_linsolve_sympos_nocheck_doc
    },
    {
      s_linsolve_cg_sympos_str,
      (PyCFunction)py_linsolve_cg_sympos,
      METH_VARARGS|METH_KEYWORDS,
      s_linsolve_cg_sympos_doc
    },
    {
      s_linsolve_cg_sympos_nocheck_str,
      (PyCFunction)py_linsolve_cg_sympos_nocheck,
      METH_VARARGS|METH_KEYWORDS,
      s_linsolve_cg_sympos_nocheck_doc
    },
    {0}  /* Sentinel */
};

PyDoc_STRVAR(module_docstr, "bob::math classes and methods");

int PyXbobMath_APIVersion = XBOB_MATH_API_VERSION;

PyMODINIT_FUNC XBOB_EXT_ENTRY_NAME (void) {

  PyObject* m = Py_InitModule3(XBOB_EXT_MODULE_NAME,
      module_methods, module_docstr);

  /* register some constants */
  PyModule_AddIntConstant(m, "__api_version__", XBOB_MATH_API_VERSION);
  PyModule_AddStringConstant(m, "__version__", XBOB_EXT_MODULE_VERSION);

  /* exhaustive list of C APIs */
  static void* PyXbobMath_API[PyXbobMath_API_pointers];

  /**************
   * Versioning *
   **************/

  PyXbobMath_API[PyXbobMath_APIVersion_NUM] = (void *)&PyXbobMath_APIVersion;

#if PY_VERSION_HEX >= 0x02070000

  /* defines the PyCapsule */

  PyObject* c_api_object = PyCapsule_New((void *)PyXbobMath_API,
      XBOB_EXT_MODULE_PREFIX "." XBOB_EXT_MODULE_NAME "._C_API", 0);

#else

  PyObject* c_api_object = PyCObject_FromVoidPtr((void *)PyXbobMath_API, 0);

#endif

  if (c_api_object) PyModule_AddObject(m, "_C_API", c_api_object);

  /* imports the NumPy C-API */
  import_array();

  /* imports xbob.blitz C-API */
  import_xbob_blitz();

}
