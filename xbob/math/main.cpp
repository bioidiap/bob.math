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
#include "pavx.h"
#include "norminv.h"
#include "scatter.h"

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
.. warning::\n\
\n\
   THIS VARIANT DOES NOT PERFORM ANY CHECKS ON THE INPUT MATRICES AND IS,\n\
   FASTER THEN THE VARIANT NOT ENDING IN ``_``. Use it when you are sure\n\
   your input matrices sizes match.\n\
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
.. warning::\n\
\n\
   THIS VARIANT DOES NOT PERFORM ANY CHECKS ON THE INPUT MATRICES AND IS,\n\
   FASTER THEN THE VARIANT NOT ENDING IN ``_``. Use it when you are sure\n\
   your input matrices sizes match.\n\
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
.. warning::\n\
\n\
   THIS VARIANT DOES NOT PERFORM ANY CHECKS ON THE INPUT MATRICES AND IS,\n\
   FASTER THEN THE VARIANT NOT ENDING IN ``_``. Use it when you are sure\n\
   your input matrices sizes match.\n\
\n\
You can use this method in two different formats. The first interface\n\
accepts the matrices ``A`` and ``b`` returning ``x``. The second one\n\
accepts a pre-allocated ``x`` matrix and sets it with the linear system\n\
solution.\n\
"
);

PyDoc_STRVAR(s_pavx_str, "pavx");
PyDoc_STRVAR(s_pavx_doc,
"pavx(input, output) -> None\n\
pavx(input) -> array\n\
\n\
Applies the Pool-Adjacent-Violators Algorithm to ``input``. The ``input``\n\
and ``output`` arrays should have the same size. This is a simplified\n\
C++ port of the isotonic regression code made available at the `University\n\
of Bern website <http://www.imsv.unibe.ch/content/staff/personalhomepages/duembgen/software/isotonicregression/index_eng.html>`_.\n\
\n\
You can use this method in two different formats. The first interface\n\
accepts the 1D float arrays ``input`` and ``output``. The second one\n\
accepts the input array ``input`` and allocates a new ``output`` array\n\
which is returned. In such a case, the ``output`` is a 1D float array\n\
with the same length as ``input``.\n\
");

PyDoc_STRVAR(s_pavx_nocheck_str, "pavx_");
PyDoc_STRVAR(s_pavx_nocheck_doc,
"pavx(input, output) -> None\n\
\n\
Applies the Pool-Adjacent-Violators Algorithm to ``input`` and places the\n\
result on ``output``. The ``input`` and ``output`` arrays should be 1D\n\
float arrays with the same length.\n\
\n\
This is a simplified C++ port of the isotonic regression code\n\
made available at the `University of Bern website <http://www.imsv.unibe.ch/content/staff/personalhomepages/duembgen/software/isotonicregression/index_eng.html>`_.\n\
\n\
.. warning::\n\
\n\
   THIS VARIANT DOES NOT PERFORM ANY CHECKS ON THE INPUT MATRICES AND IS,\n\
   FASTER THEN THE VARIANT NOT ENDING IN ``_``. Use it when you are sure\n\
   your input and output vector sizes match.\n\
\n\
");

PyDoc_STRVAR(s_pavx_width_str, "pavxWidth");
PyDoc_STRVAR(s_pavx_width_doc,
"pavxWidth(input, output) -> array\n\
\n\
Applies the Pool-Adjacent-Violators Algorithm to ``input`` and places the\n\
result on ``output``. The ``input`` and ``output`` arrays should be 1D\n\
float arrays with the same length.\n\
\n\
The width array (64-bit unsigned integer 1D) is returned and has the\n\
same size as ``input`` and ``output``.\n\
");

PyDoc_STRVAR(s_pavx_width_height_str, "pavxWidthHeight");
PyDoc_STRVAR(s_pavx_width_height_doc,
"pavxWidthHeight(input, output) -> (array, array)\n\
\n\
Applies the Pool-Adjacent-Violators Algorithm to ``input`` and sets the\n\
result on ``output``. The ``input`` and ``output`` arrays should be 1D\n\
float arrays of the same length.\n\
\n\
This is a simplified C++ port of the isotonic regression code\n\
made available at the `University of Bern website <http://www.imsv.unibe.ch/content/staff/personalhomepages/duembgen/software/isotonicregression/index_eng.html>`_.\n\
\n\
The width and height arrays are returned. The width array is a 64-bit\n\
**unsigned integer** 1D array, while the height array (second component\n\
of the returned tuple) is a 64-bit **float** 1D array of the same size.\n\
");

PyDoc_STRVAR(s_norminv_str, "norminv");
PyDoc_STRVAR(s_norminv_doc,
"norminv(p, mu, sigma) -> scalar\n\
\n\
Computes the inverse normal cumulative distribution for a probability\n\
``p``, given a distribution with mean ``mu`` and standard deviation\n\
``sigma``. The value ``p`` must lie in the range [0,1].\n\
\n\
Reference: `<http://home.online.no/~pjacklam/notes/invnorm/>`_\n\
");

PyDoc_STRVAR(s_normsinv_str, "normsinv");
PyDoc_STRVAR(s_normsinv_doc,
"normsinv(p) -> scalar\n\
\n\
Computes the inverse normal cumulative distribution for a probability\n\
``p``, given a distribution with mean 0.0 and standard deviation 1.0.\n\
It is equivalent as calling :py:func:`norminv(p, 0, 1)`. The value\n\
``p`` must lie in the range [0,1].\n\
\n\
Reference: `<http://home.online.no/~pjacklam/notes/invnorm/>`_\n\
");

PyDoc_STRVAR(s_scatter_str, "scatter");
PyDoc_STRVAR(s_scatter_doc,
"scatter(a) -> (array, array)\n\
scatter(a, s) -> array\n\
scatter(a, s, m) -> None\n\
\n\
Computes the scatter matrix of a 2D array *considering data is organized\n\
row-wise* (each sample is a row, each feature is a column). The\n\
resulting array ``s`` is squared with extents equal to the\n\
number of columns in ``a``. The resulting array ``m`` is a 1D array\n\
with the row means of ``a``. This method supports only 32 or 64-bit\n\
float arrays as input.\n\
\n\
This function supports many calling modes, but you should provide, at\n\
least, the input data matrix ``a``. All non-provided arguments will be\n\
allocated internally and returned.\n\
");

PyDoc_STRVAR(s_scatter_nocheck_str, "scatter_");
PyDoc_STRVAR(s_scatter_nocheck_doc,
"scatter_(a, s, m) -> None\n\
\n\
Computes the scatter matrix of a 2D array *considering data is organized\n\
row-wise* (each sample is a row, each feature is a column). The\n\
resulting array ``s`` is squared with extents equal to the\n\
number of columns in ``a``. The resulting array ``m`` is a 1D array\n\
with the row means of ``a``. This method supports only 32 or 64-bit\n\
float arrays as input.\n\
\n\
.. warning::\n\
\n\
   THIS VARIANT DOES NOT PERFORM ANY CHECKS ON THE INPUT MATRICES AND IS,\n\
   FASTER THEN THE VARIANT NOT ENDING IN ``_``. Use it when you are sure\n\
   your input matrices sizes match.\n\
");

PyDoc_STRVAR(s_scatters_str, "scatters");
PyDoc_STRVAR(s_scatters_doc,
"scatters(data) -> (array, array, array)\n\
scatters(data, sw, sb) -> array\n\
scatters(data, sw, sb, m) -> None\n\
\n\
Computes the within-class (``sw``) and between-class (``sb``) scatter\n\
matrices of a set of 2D arrays considering data is organized row-wise\n\
(each sample is a row, each feature is a column).\n\
\n\
This function supports many calling modes, but you should provide, at\n\
least, the input data matrices ``data``, which should be sequence of\n\
2D 32-bit or 64-bit float values in which every row of each matrix\n\
represents one observation for a given class. **Every matrix in\n\
``data`` should have exactly the same number of columns.**\n\
\n\
The returned values ``sw`` and ``sb`` are square matrices with the\n\
same number of rows and columns as the number of columns in ``data``.\n\
The returned value ``m`` (last call variant) is a 1D array with the\n\
same length as number of columns in each ``data`` matrix and represents\n\
the ensemble mean with no prior (i.e., biased towards classes with more\n\
samples.\n\
\n\
Strategy implemented:\n\
\n\
1. Evaluate the overall mean (``m``), class means (:math:`m_k`) and the\n\
   total class counts (:math:`N`).\n\
2. Evaluate ``sw`` and ``sb`` using normal loops.\n\
\n\
Note that ``sw`` and ``sb``, in this implementation, will be normalized\n\
by N-1 (number of samples) and K (number of classes). This procedure\n\
makes the eigen values scaled by (N-1)/K, effectively increasing their\n\
values. The main motivation for this normalization are numerical\n\
precision concerns with the increasing number of samples causing a\n\
rather large Sw matrix. A normalization strategy mitigates this\n\
problem. The eigen vectors will see no effect on this normalization as\n\
they are normalized in the euclidean sense (:math:`||a|| = 1`) so that\n\
does not change those.\n\
");

PyDoc_STRVAR(s_scatters_nocheck_str, "scatters_");
PyDoc_STRVAR(s_scatters_nocheck_doc,
"scatters_(data, sw, sb) -> None\n\
scatters_(data, sw, sb, m) -> None\n\
\n\
Computes the within-class (``sw``) and between-class (``sb``) scatter\n\
matrices of a set of 2D arrays considering data is organized row-wise\n\
(each sample is a row, each feature is a column).\n\
\n\
.. warning::\n\
\n\
   THIS VARIANT DOES NOT PERFORM ANY CHECKS ON THE INPUT MATRICES AND IS,\n\
   FASTER THEN THE VARIANT NOT ENDING IN ``_``. Use it when you are sure\n\
   your input matrices sizes match.\n\
\n\
This function supports many calling modes, but you should provide, at\n\
least, the input data matrices ``data``, which should be sequence of\n\
2D 32-bit or 64-bit float values in which every row of each matrix\n\
represents one observation for a given class. **Every matrix in\n\
``data`` should have exactly the same number of columns.**\n\
\n\
The returned values ``sw`` and ``sb`` are square matrices with the\n\
same number of rows and columns as the number of columns in ``data``.\n\
The returned value ``m`` (last call variant) is a 1D array with the\n\
same length as number of columns in each ``data`` matrix and represents\n\
the ensemble mean with no prior (i.e., biased towards classes with more\n\
samples. **In this variant, you should pre-allocate all output matrices\n\
so that scatters (and the overall mean) are stored on your provided\n\
arrays**.\n\
\n\
Strategy implemented:\n\
\n\
1. Evaluate the overall mean (``m``), class means (:math:`m_k`) and the\n\
   total class counts (:math:`N`).\n\
2. Evaluate ``sw`` and ``sb`` using normal loops.\n\
\n\
Note that ``sw`` and ``sb``, in this implementation, will be normalized\n\
by N-1 (number of samples) and K (number of classes). This procedure\n\
makes the eigen values scaled by (N-1)/K, effectively increasing their\n\
values. The main motivation for this normalization are numerical\n\
precision concerns with the increasing number of samples causing a\n\
rather large Sw matrix. A normalization strategy mitigates this\n\
problem. The eigen vectors will see no effect on this normalization as\n\
they are normalized in the euclidean sense (:math:`||a|| = 1`) so that\n\
does not change those.\n\
");

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
    {
      s_pavx_str,
      (PyCFunction)py_pavx,
      METH_VARARGS|METH_KEYWORDS,
      s_pavx_doc
    },
    {
      s_pavx_nocheck_str,
      (PyCFunction)py_pavx_nocheck,
      METH_VARARGS|METH_KEYWORDS,
      s_pavx_nocheck_doc
    },
    {
      s_pavx_width_str,
      (PyCFunction)py_pavx_width,
      METH_VARARGS|METH_KEYWORDS,
      s_pavx_width_doc
    },
    {
      s_pavx_width_height_str,
      (PyCFunction)py_pavx_width_height,
      METH_VARARGS|METH_KEYWORDS,
      s_pavx_width_height_doc
    },
    {
      s_norminv_str,
      (PyCFunction)py_norminv,
      METH_VARARGS|METH_KEYWORDS,
      s_norminv_doc
    },
    {
      s_normsinv_str,
      (PyCFunction)py_normsinv,
      METH_VARARGS|METH_KEYWORDS,
      s_normsinv_doc
    },
    {
      s_scatter_str,
      (PyCFunction)py_scatter,
      METH_VARARGS|METH_KEYWORDS,
      s_scatter_doc
    },
    {
      s_scatter_nocheck_str,
      (PyCFunction)py_scatter_nocheck,
      METH_VARARGS|METH_KEYWORDS,
      s_scatter_nocheck_doc
    },
    {
      s_scatters_str,
      (PyCFunction)py_scatters,
      METH_VARARGS|METH_KEYWORDS,
      s_scatters_doc
    },
    {
      s_scatters_nocheck_str,
      (PyCFunction)py_scatters_nocheck,
      METH_VARARGS|METH_KEYWORDS,
      s_scatters_nocheck_doc
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
