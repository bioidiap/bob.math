/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue  3 Dec 14:13:22 2013 CET
 *
 * @brief Bindings to bob::math
 */

#ifdef NO_IMPORT_ARRAY
#undef NO_IMPORT_ARRAY
#endif
#include <bob.blitz/capi.h>
#include <bob.blitz/cleanup.h>
#include <bob.core/api.h>

#include <bob.extension/documentation.h>

#include "histogram.h"
#include "linsolve.h"
#include "pavx.h"
#include "norminv.h"
#include "scatter.h"
#include "lp_interior_point.h"

static bob::extension::FunctionDoc s_histogram_intersection = bob::extension::FunctionDoc(
    "histogram_intersection",
    "Computes the histogram intersection between the given histograms, which might be of singular dimension only.",
    "The histogram intersection is computed as follows:\n\n"
    ".. math:: sim(h_1,h_2) = \\sum_i \\min \\{h_{1i}, h_{2i}\\}\n\n"
    "The histogram intersection defines a similarity measure, so higher values are better. "
    "You can use this method in two different formats. "
    "The first interface accepts non-sparse histograms. "
    "The second interface accepts sparse histograms represented by indexes and values.\n\n"
    ".. note:: Histograms are given as two matrices, one with the indexes and one with the data. All data points that for which no index exists are considered to be zero.\n\n"
    ".. note:: In general, histogram intersection with sparse histograms needs more time to be computed."
  )
  .add_prototype("h1, h2", "sim")
  .add_prototype("index_1, value_1, index_2, value_2", "sim")
  .add_parameter("h1, h2", "array_like (1D)", "Histograms to compute the histogram intersection for")
  .add_parameter("index_1, index_2", "array_like (int, 1D)", "Indices of the sparse histograms value_1 and value_2")
  .add_parameter("value_1, value_2", "array_like (1D)", "Sparse histograms to compute the histogram intersection for")
  .add_return("sim", "float", "The histogram intersection value for the given histograms.")
;

static bob::extension::FunctionDoc s_chi_square = bob::extension::FunctionDoc(
    "chi_square",
    "Computes the chi square distance between the given histograms, which might be of singular dimension only.",
    "The chi square distance is computed as follows:\n\n"
    ".. math:: dist(h_1,h_2) = \\sum_i \\frac{(h_{1i} - h_{2i})^2}{h_{1i} + h_{2i}}\n\n"
    "Chi square defines a distance metric, so lower values are better. "
    "You can use this method in two different formats. "
    "The first interface accepts non-sparse histograms. "
    "The second interface accepts sparse histograms represented by indexes and values.\n\n"
    ".. note:: Histograms are given as two matrices, one with the indexes and one with the data. All data points that for which no index exists are considered to be zero.\n\n"
    ".. note:: In general, histogram intersection with sparse histograms needs more time to be computed."
  )
  .add_prototype("h1, h2", "dist")
  .add_prototype("index_1, value_1, index_2, value_2", "dist")
  .add_parameter("h1, h2", "array_like (1D)", "Histograms to compute the chi square distance for")
  .add_parameter("index_1, index_2", "array_like (int, 1D)", "Indices of the sparse histograms value_1 and value_2")
  .add_parameter("value_1, value_2", "array_like (1D)", "Sparse histograms to compute the chi square distance for")
  .add_return("dist", "float", "The chi square distance value for the given histograms.")
;

static bob::extension::FunctionDoc s_kullback_leibler = bob::extension::FunctionDoc(
    "kullback_leibler",
    "Computes the Kullback-Leibler histogram divergence between the given histograms, which might be of singular dimension only.",
    "The chi square distance is inspired by `link <http://www.informatik.uni-freiburg.de/~tipaldi/FLIRTLib/HistogramDistances_8hpp_source.html>`_ and computed as follows:\n\n"
    ".. math:: dist(h_1,h_2) = \\sum_i (h_{1i} - h_{2i}) * \\log (h_{1i} / h_{2i})\n\n"
    "The Kullback-Leibler divergence defines a distance metric, so lower values are better. "
    "You can use this method in two different formats. "
    "The first interface accepts non-sparse histograms. "
    "The second interface accepts sparse histograms represented by indexes and values.\n\n"
    ".. note:: Histograms are given as two matrices, one with the indexes and one with the data. All data points that for which no index exists are considered to be zero.\n\n"
    ".. note:: In general, histogram intersection with sparse histograms needs more time to be computed."
  )
  .add_prototype("h1, h2", "dist")
  .add_prototype("index_1, value_1, index_2, value_2", "dist")
  .add_parameter("h1, h2", "array_like (1D)", "Histograms to compute the Kullback-Leibler divergence for")
  .add_parameter("index_1, index_2", "array_like (int, 1D)", "Indices of the sparse histograms value_1 and value_2")
  .add_parameter("value_1, value_2", "array_like (1D)", "Sparse histograms to compute the Kullback-Leibler divergence for")
  .add_return("dist", "float", "The Kullback-Leibler divergence value for the given histograms.")
;

static bob::extension::FunctionDoc s_linsolve = bob::extension::FunctionDoc(
  "linsolve",
  "Solves the linear system :math:`Ax=b` and returns the result in :math:`x`.",
  "This method uses LAPACK's ``dgesv`` generic solver. "
  "You can use this method in two different formats. "
  "The first interface accepts the matrices :math:`A` and :math:`b` returning :math:`x`. "
  "The second one accepts a pre-allocated :math:`x` vector and sets it with the linear system solution."
  )
  .add_prototype("A, b", "x")
  .add_prototype("A, b, x")
  .add_parameter("A", "array_like (2D)", "The matrix :math:`A` of the linear system")
  .add_parameter("b", "array_like (1D)", "The vector :math:`b` of the linear system")
  .add_parameter("x", "array_like (1D)", "The result vector :math:`x`, as parameter")
  .add_return("x", "array_like (1D)", "The result vector :math:`x`, as return value")
;

static bob::extension::FunctionDoc s_linsolve_nocheck = bob::extension::FunctionDoc(
  "linsolve_",
  "Solves the linear system :math:`Ax=b` and returns the result in :math:`x`.",
  ".. warning:: This variant does not perform any checks on the input matrices and is faster then :py:func:`linsolve`. "
  "Use it when you are sure your input matrices sizes match.\n\n"
  "This method uses LAPACK's ``dgesv`` generic solver. "
  "You can use this method in two different formats. "
  "The first interface accepts the matrices :math:`A` and :math:`b` returning :math:`x`. "
  "The second one accepts a pre-allocated :math:`x` vector and sets it with the linear system solution."
  )
  .add_prototype("A, b", "x")
  .add_prototype("A, b, x")
  .add_parameter("A", "array_like (2D)", "The matrix :math:`A` of the linear system")
  .add_parameter("b", "array_like (1D)", "The vector :math:`b` of the linear system")
  .add_parameter("x", "array_like (1D)", "The result vector :math:`x`, as parameter")
  .add_return("x", "array_like (1D)", "The result vector :math:`x`, as return value")
;

static bob::extension::FunctionDoc s_linsolve_sympos = bob::extension::FunctionDoc(
  "linsolve_sympos",
  "Solves the linear system :math:`Ax=b` and returns the result in :math:`x` for symmetric :math:`A` matrix.",
  "This method uses LAPACK's ``dposv`` solver, assuming :math:`A` is a symmetric positive definite matrix. "
  "You can use this method in two different formats. "
  "The first interface accepts the matrices :math:`A` and :math:`b` returning :math:`x`. "
  "The second one accepts a pre-allocated :math:`x` vector and sets it with the linear system solution."
  )
  .add_prototype("A, b", "x")
  .add_prototype("A, b, x")
  .add_parameter("A", "array_like (2D)", "The matrix :math:`A` of the linear system")
  .add_parameter("b", "array_like (1D)", "The vector :math:`b` of the linear system")
  .add_parameter("x", "array_like (1D)", "The result vector :math:`x`, as parameter")
  .add_return("x", "array_like (1D)", "The result vector :math:`x`, as return value")
;

static bob::extension::FunctionDoc s_linsolve_sympos_nocheck = bob::extension::FunctionDoc(
  "linsolve_sympos_",
  "Solves the linear system :math:`Ax=b` and returns the result in :math:`x` for symmetric :math:`A` matrix.",
  ".. warning:: This variant does not perform any checks on the input matrices and is faster then :py:func:`linsolve_sympos`. "
  "Use it when you are sure your input matrices sizes match.\n\n"
  "This method uses LAPACK's ``dposv`` solver, assuming :math:`A` is a symmetric positive definite matrix. "
  "You can use this method in two different formats. "
  "The first interface accepts the matrices :math:`A` and :math:`b` returning :math:`x`. "
  "The second one accepts a pre-allocated :math:`x` vector and sets it with the linear system solution."
  )
  .add_prototype("A, b", "x")
  .add_prototype("A, b, x")
  .add_parameter("A", "array_like (2D)", "The matrix :math:`A` of the linear system")
  .add_parameter("b", "array_like (1D)", "The vector :math:`b` of the linear system")
  .add_parameter("x", "array_like (1D)", "The result vector :math:`x`, as parameter")
  .add_return("x", "array_like (1D)", "The result vector :math:`x`, as return value")
;

static bob::extension::FunctionDoc s_linsolve_cg_sympos = bob::extension::FunctionDoc(
  "linsolve_cg_sympos",
  "Solves the linear system :math:`Ax=b` using conjugate gradients and returns the result in :math:`x` for symmetric :math:`A` matrix.",
  "This method uses the conjugate gradient solver, assuming :math:`A` is a symmetric positive definite matrix. "
  "You can use this method in two different formats. "
  "The first interface accepts the matrices :math:`A` and :math:`b` returning :math:`x`. "
  "The second one accepts a pre-allocated :math:`x` vector and sets it with the linear system solution."
  )
  .add_prototype("A, b", "x")
  .add_prototype("A, b, x")
  .add_parameter("A", "array_like (2D)", "The matrix :math:`A` of the linear system")
  .add_parameter("b", "array_like (1D)", "The vector :math:`b` of the linear system")
  .add_parameter("x", "array_like (1D)", "The result vector :math:`x`, as parameter")
  .add_return("x", "array_like (1D)", "The result vector :math:`x`, as return value")
;

static bob::extension::FunctionDoc s_linsolve_cg_sympos_nocheck = bob::extension::FunctionDoc(
  "linsolve_cg_sympos_",
  "Solves the linear system :math:`Ax=b` using conjugate gradients and returns the result in :math:`x` for symmetric :math:`A` matrix.",
  ".. warning:: This variant does not perform any checks on the input matrices and is faster then :py:func:`linsolve_cg_sympos`. "
  "Use it when you are sure your input matrices sizes match.\n\n"
  "This method uses the conjugate gradient solver, assuming :math:`A` is a symmetric positive definite matrix. "
  "You can use this method in two different formats. "
  "The first interface accepts the matrices :math:`A` and :math:`b` returning :math:`x`. "
  "The second one accepts a pre-allocated :math:`x` vector and sets it with the linear system solution."
  )
  .add_prototype("A, b", "x")
  .add_prototype("A, b, x")
  .add_parameter("A", "array_like (2D)", "The matrix :math:`A` of the linear system")
  .add_parameter("b", "array_like (1D)", "The vector :math:`b` of the linear system")
  .add_parameter("x", "array_like (1D)", "The result vector :math:`x`, as parameter")
  .add_return("x", "array_like (1D)", "The result vector :math:`x`, as return value")
;

static bob::extension::FunctionDoc s_pavx = bob::extension::FunctionDoc(
  "pavx",
  "Applies the Pool-Adjacent-Violators Algorithm",
  "Applies the Pool-Adjacent-Violators Algorithm to ``input``. "
  "This is a simplified C++ port of the isotonic regression code made available at the `University of Bern website <http://www.imsv.unibe.ch/content/staff/personalhomepages/duembgen/software/isotonicregression/index_eng.html>`_.\n\n"
  "You can use this method in two different formats. "
  "The first interface accepts the ``input`` and ``output``. "
  "The second one accepts the input array ``input`` and allocates a new ``output`` array, which is returned. "
  )
  .add_prototype("input, output")
  .add_prototype("input", "output")
  .add_parameter("input", "array_like (float, 1D)", "The input matrix for the PAV algorithm.")
  .add_parameter("output", "array_like (float, 1D)", "The output matrix, must be of the same size as ``input``")
  .add_return("output", "array_like (float, 1D)", "The output matrix; will be created in the same size as ``input``")
;

static bob::extension::FunctionDoc s_pavx_nocheck = bob::extension::FunctionDoc(
  "pavx_",
  "Applies the Pool-Adjacent-Violators Algorithm",
  ".. warning:: This variant does not perform any checks on the input matrices and is faster then :py:func:`pavx`. "
  "Use it when you are sure your input matrices sizes match.\n\n"
  "Applies the Pool-Adjacent-Violators Algorithm to ``input``. "
  "This is a simplified C++ port of the isotonic regression code made available at the `University of Bern website <http://www.imsv.unibe.ch/content/staff/personalhomepages/duembgen/software/isotonicregression/index_eng.html>`_.\n\n"
  "You can use this method in two different formats. "
  "The first interface accepts the ``input`` and ``output``. "
  "The second one accepts the input array ``input`` and allocates a new ``output`` array, which is returned. "
  )
  .add_prototype("input, output")
  .add_prototype("input", "output")
  .add_parameter("input", "array_like (float, 1D)", "The input matrix for the PAV algorithm.")
  .add_parameter("output", "array_like (float, 1D)", "The output matrix, must be of the same size as ``input``")
  .add_return("output", "array_like (float, 1D)", "The output matrix; will be created in the same size as ``input``")
;

static bob::extension::FunctionDoc s_pavx_width = bob::extension::FunctionDoc(
  "pavxWidth",
  "Applies the Pool-Adjacent-Violators Algorithm and returns the width.",
  "Applies the Pool-Adjacent-Violators Algorithm to ``input``. "
  "This is a simplified C++ port of the isotonic regression code made available at the `University of Bern website <http://www.imsv.unibe.ch/content/staff/personalhomepages/duembgen/software/isotonicregression/index_eng.html>`_."
  )
  .add_prototype("input, output", "width")
  .add_parameter("input", "array_like (float, 1D)", "The input matrix for the PAV algorithm.")
  .add_parameter("output", "array_like (float, 1D)", "The output matrix, must be of the same size as ``input``")
  .add_return("width", "array_like (uint64, 1D)", "The width matrix will be created in the same size as ``input``\n\n.. todo:: Explain, what width means in this case")
;

static bob::extension::FunctionDoc s_pavx_width_height = bob::extension::FunctionDoc(
  "pavxWidthHeight",
  "Applies the Pool-Adjacent-Violators Algorithm and returns the width and the height.",
  "Applies the Pool-Adjacent-Violators Algorithm to ``input``. "
  "This is a simplified C++ port of the isotonic regression code made available at the `University of Bern website <http://www.imsv.unibe.ch/content/staff/personalhomepages/duembgen/software/isotonicregression/index_eng.html>`_."
  )
  .add_prototype("input, output", "width, height")
  .add_parameter("input", "array_like (float, 1D)", "The input matrix for the PAV algorithm.")
  .add_parameter("output", "array_like (float, 1D)", "The output matrix, must be of the same size as ``input``")
  .add_return("width", "array_like (uint64, 1D)", "The width matrix will be created in the same size as ``input``\n\n.. todo:: Explain, what width means in this case")
  .add_return("height", "array_like (float, 1D)", "The height matrix will be created in the same size as ``input``\n\n.. todo:: Explain, what height means in this case")
;

static bob::extension::FunctionDoc s_norminv = bob::extension::FunctionDoc(
  "norminv",
  "Computes the inverse normal cumulative distribution",
  "Computes the inverse normal cumulative distribution for a probability :math:`p`, given a distribution with mean :math:`\\mu` and standard deviation :math:`\\sigma`. "
  "Reference: http://home.online.no/~pjacklam/notes/invnorm/"
  )
  .add_prototype("p, mu, sigma", "inv")
  .add_parameter("p", "float", "The value to get the inverse distribution of, must lie in the range :math:`[0,1]`")
  .add_parameter("mu", "float", "The mean :math:`\\mu` of the normal distribution")
  .add_parameter("sigma", "float", "The standard deviation :math:`\\sigma` of the normal distribution")
  .add_return("inv", "float", "The inverse of the normal distribution")
;

static bob::extension::FunctionDoc s_normsinv = bob::extension::FunctionDoc(
  "normsinv",
  "Computes the inverse normal cumulative distribution",
  "Computes the inverse normal cumulative distribution for a probability :math:`p`, given a distribution with mean :math:`\\mu=0` and standard deviation :math:`\\sigma=1`. "
  "It is equivalent as calling ``norminv(p, 0, 1)`` (see :py:func:`norminv`). "
  "Reference: http://home.online.no/~pjacklam/notes/invnorm/"
  )
  .add_prototype("p", "inv")
  .add_parameter("p", "float", "The value to get the inverse distribution of, must lie in the range :math:`[0,1]`")
  .add_return("inv", "float", "The inverse of the normal distribution")
;

static bob::extension::FunctionDoc s_scatter = bob::extension::FunctionDoc(
  "scatter",
  "Computes scatter matrix of a 2D array.",
  "Computes the scatter matrix of a 2D array *considering data is organized row-wise* (each sample is a row, each feature is a column). "
  "The resulting array ``s`` is squared with extents equal to the number of columns in ``a``. "
  "The resulting array ``m`` is a 1D array with the row means of ``a``. "
  "This function supports many calling modes, but you should provide, at least, the input data matrix ``a``. "
  "All non-provided arguments will be allocated internally and returned."
  )
  .add_prototype("a", "s, m")
  .add_prototype("a, s", "m")
  .add_prototype("a, m", "s")
  .add_prototype("a, s, m")
  .add_parameter("a", "array_like (float, 2D)", "The sample matrix, *considering data is organized row-wise* (each sample is a row, each feature is a column)")
  .add_parameter("s", "array_like (float, 2D)", "The scatter matrix, squared with extents equal to the number of columns in ``a``")
  .add_parameter("m", "array_like (float,1D)", "The mean matrix, with with the row means of ``a``")
  .add_return("s", "array_like (float, 2D)", "The scatter matrix, squared with extents equal to the number of columns in ``a``")
  .add_return("m", "array_like (float, 1D)", "The mean matrix, with with the row means of ``a``")
;

static bob::extension::FunctionDoc s_scatter_nocheck = bob::extension::FunctionDoc(
  "scatter_",
  "Computes scatter matrix of a 2D array.",
  ".. warning:: This variant does not perform any checks on the input matrices and is faster then :py:func:`scatter`."
  "Use it when you are sure your input matrices sizes match.\n\n"
  "Computes the scatter matrix of a 2D array *considering data is organized row-wise* (each sample is a row, each feature is a column). "
  "The resulting array ``s`` is squared with extents equal to the number of columns in ``a``. "
  "The resulting array ``m`` is a 1D array with the row means of ``a``. "
  "This function supports many calling modes, but you should provide, at least, the input data matrix ``a``. "
  "All non-provided arguments will be allocated internally and returned."
  )
  .add_prototype("a, s, m")
  .add_parameter("a", "array_like (float, 2D)", "The sample matrix, *considering data is organized row-wise* (each sample is a row, each feature is a column)")
  .add_parameter("s", "array_like (float, 2D)", "The scatter matrix, squared with extents equal to the number of columns in ``a``")
  .add_parameter("m", "array_like (float,1D)", "The mean matrix, with with the row means of ``a``")
;


static bob::extension::FunctionDoc s_scatters = bob::extension::FunctionDoc(
  "scatters",
  "Computes :math:`S_w` and :math:`S_b` scatter matrices of a set of 2D arrays.",
  "Computes the within-class :math:`S_w` and between-class :math:`S_b` scatter matrices of a set of 2D arrays considering data is organized row-wise (each sample is a row, each feature is a column), and each matrix contains data of one class. "
  "Computes the scatter matrix of a 2D array *considering data is organized row-wise* (each sample is a row, each feature is a column). "
  "The implemented strategy is:\n\n"
  "1. Evaluate the overall mean (``m``), class means (:math:`m_k`) and the  total class counts (:math:`N`).\n"
  "2. Evaluate ``sw`` and ``sb`` using normal loops.\n\n"
  "Note that in this implementation, ``sw`` and ``sb`` will be normalized by N-1 (number of samples) and K (number of classes). "
  "This procedure makes the eigen values scaled by (N-1)/K, effectively increasing their values. "
  "The main motivation for this normalization are numerical precision concerns with the increasing number of samples causing a rather large :math:`S_w` matrix. "
  "A normalization strategy mitigates this problem. "
  "The eigen vectors will see no effect on this normalization as they are normalized in the euclidean sense (:math:`||a|| = 1`) so that does not change those.\n\n"
  "This function supports many calling modes, but you should provide, at least, the input ``data``. "
  "All non-provided arguments will be allocated internally and returned."
  )
  .add_prototype("data", "sw, sb, m")
  .add_prototype("data, sw, sb", "m")
  .add_prototype("data, sw, sb, m")
  .add_parameter("data", "[array_like (float, 2D)]", "The list of sample matrices. "
      "In each sample matrix the data is organized row-wise (each sample is a row, each feature is a column). "
      "Each matrix stores the data of a particular class. "
      "**Every matrix in ``data`` must have exactly the same number of columns.**")
  .add_parameter("sw", "array_like (float, 2D)", "The within-class scatter matrix :math:`S_w`, squared with extents equal to the number of columns in ``data``")
  .add_parameter("sb", "array_like (float, 2D)", "The between-class scatter matrix :math:`S_b`, squared with extents equal to the number of columns in ``data``")
  .add_parameter("m", "array_like (float,1D)", "The mean matrix, representing the ensemble mean with no prior (i.e., biased towards classes with more samples)")
  .add_return("sw", "array_like (float, 2D)", "The within-class scatter matrix :math:`S_w`")
  .add_return("sb", "array_like (float, 2D)", "The between-class scatter matrix :math:`S_b`")
  .add_return("m", "array_like (float, 1D)", "The mean matrix, representing the ensemble mean with no prior (i.e., biased towards classes with more samples)")
;

static bob::extension::FunctionDoc s_scatters_nocheck = bob::extension::FunctionDoc(
  "scatters_",
  "Computes :math:`S_w` and :math:`S_b` scatter matrices of a set of 2D arrays.",
  ".. warning:: This variant does not perform any checks on the input matrices and is faster then :py:func:`scatters`. "
  "Use it when you are sure your input matrices sizes match.\n\n"
  "For a detailed description of the function, please see :func:`scatters`."
  )
  .add_prototype("data, sw, sb, m")
  .add_prototype("data, sw, sb")
  .add_parameter("data", "[array_like (float, 2D)]", "The list of sample matrices. "
      "In each sample matrix the data is organized row-wise (each sample is a row, each feature is a column). "
      "Each matrix stores the data of a particular class. "
      "**Every matrix in ``data`` must have exactly the same number of columns.**")
  .add_parameter("sw", "array_like (float, 2D)", "The within-class scatter matrix :math:`S_w`, squared with extents equal to the number of columns in ``data``")
  .add_parameter("sb", "array_like (float, 2D)", "The between-class scatter matrix :math:`S_b`, squared with extents equal to the number of columns in ``data``")
  .add_parameter("m", "array_like (float,1D)", "The mean matrix, representing the ensemble mean with no prior (i.e., biased towards classes with more samples)")
;


static PyMethodDef module_methods[] = {
    {
      s_histogram_intersection.name(),
      (PyCFunction)py_histogram_intersection,
      METH_VARARGS|METH_KEYWORDS,
      s_histogram_intersection.doc()
    },
    {
      s_chi_square.name(),
      (PyCFunction)py_chi_square,
      METH_VARARGS|METH_KEYWORDS,
      s_chi_square.doc()
    },
    {
      s_kullback_leibler.name(),
      (PyCFunction)py_kullback_leibler,
      METH_VARARGS|METH_KEYWORDS,
      s_kullback_leibler.doc()
    },
    {
      s_linsolve.name(),
      (PyCFunction)py_linsolve,
      METH_VARARGS|METH_KEYWORDS,
      s_linsolve.doc()
    },
    {
      s_linsolve_nocheck.name(),
      (PyCFunction)py_linsolve_nocheck,
      METH_VARARGS|METH_KEYWORDS,
      s_linsolve_nocheck.doc()
    },
    {
      s_linsolve_sympos.name(),
      (PyCFunction)py_linsolve_sympos,
      METH_VARARGS|METH_KEYWORDS,
      s_linsolve_sympos.doc()
    },
    {
      s_linsolve_sympos_nocheck.name(),
      (PyCFunction)py_linsolve_sympos_nocheck,
      METH_VARARGS|METH_KEYWORDS,
      s_linsolve_sympos_nocheck.doc()
    },
    {
      s_linsolve_cg_sympos.name(),
      (PyCFunction)py_linsolve_cg_sympos,
      METH_VARARGS|METH_KEYWORDS,
      s_linsolve_cg_sympos.doc()
    },
    {
      s_linsolve_cg_sympos_nocheck.name(),
      (PyCFunction)py_linsolve_cg_sympos_nocheck,
      METH_VARARGS|METH_KEYWORDS,
      s_linsolve_cg_sympos_nocheck.doc()
    },
    {
      s_pavx.name(),
      (PyCFunction)py_pavx,
      METH_VARARGS|METH_KEYWORDS,
      s_pavx.doc()
    },
    {
      s_pavx_nocheck.name(),
      (PyCFunction)py_pavx_nocheck,
      METH_VARARGS|METH_KEYWORDS,
      s_pavx_nocheck.doc()
    },
    {
      s_pavx_width.name(),
      (PyCFunction)py_pavx_width,
      METH_VARARGS|METH_KEYWORDS,
      s_pavx_width.doc()
    },
    {
      s_pavx_width_height.name(),
      (PyCFunction)py_pavx_width_height,
      METH_VARARGS|METH_KEYWORDS,
      s_pavx_width_height.doc()
    },
    {
      s_norminv.name(),
      (PyCFunction)py_norminv,
      METH_VARARGS|METH_KEYWORDS,
      s_norminv.doc()
    },
    {
      s_normsinv.name(),
      (PyCFunction)py_normsinv,
      METH_VARARGS|METH_KEYWORDS,
      s_normsinv.doc()
    },
    {
      s_scatter.name(),
      (PyCFunction)py_scatter,
      METH_VARARGS|METH_KEYWORDS,
      s_scatter.doc()
    },
    {
      s_scatter_nocheck.name(),
      (PyCFunction)py_scatter_nocheck,
      METH_VARARGS|METH_KEYWORDS,
      s_scatter_nocheck.doc()
    },
    {
      s_scatters.name(),
      (PyCFunction)py_scatters,
      METH_VARARGS|METH_KEYWORDS,
      s_scatters.doc()
    },
    {
      s_scatters_nocheck.name(),
      (PyCFunction)py_scatters_nocheck,
      METH_VARARGS|METH_KEYWORDS,
      s_scatters_nocheck.doc()
    },
    {0}  /* Sentinel */
};

PyDoc_STRVAR(module_docstr, "bob::math classes and methods");

#if PY_VERSION_HEX >= 0x03000000
static PyModuleDef module_definition = {
  PyModuleDef_HEAD_INIT,
  BOB_EXT_MODULE_NAME,
  module_docstr,
  -1,
  module_methods,
  0, 0, 0, 0
};
#endif

static PyObject* create_module (void) {

  PyBobMathLpInteriorPoint_Type.tp_new = PyType_GenericNew;
  if (PyType_Ready(&PyBobMathLpInteriorPoint_Type) < 0) return 0;

  PyBobMathLpInteriorPointShortstep_Type.tp_base = &PyBobMathLpInteriorPoint_Type;
  if (PyType_Ready(&PyBobMathLpInteriorPointShortstep_Type) < 0) return 0;

  PyBobMathLpInteriorPointPredictorCorrector_Type.tp_base = &PyBobMathLpInteriorPoint_Type;
  if (PyType_Ready(&PyBobMathLpInteriorPointPredictorCorrector_Type) < 0) return 0;

  PyBobMathLpInteriorPointLongstep_Type.tp_base = &PyBobMathLpInteriorPoint_Type;
  if (PyType_Ready(&PyBobMathLpInteriorPointLongstep_Type) < 0) return 0;

# if PY_VERSION_HEX >= 0x03000000
  PyObject* m = PyModule_Create(&module_definition);
# else
  PyObject* m = Py_InitModule3(BOB_EXT_MODULE_NAME, module_methods, module_docstr);
# endif
  if (!m) return 0;
  auto m_ = make_safe(m); ///< protects against early returns

  /* register the types to python */
  Py_INCREF(&PyBobMathLpInteriorPoint_Type);
  if (PyModule_AddObject(m, "LPInteriorPoint",
        (PyObject *)&PyBobMathLpInteriorPoint_Type) < 0) return 0;

  Py_INCREF(&PyBobMathLpInteriorPointShortstep_Type);
  if (PyModule_AddObject(m, "LPInteriorPointShortstep",
        (PyObject *)&PyBobMathLpInteriorPointShortstep_Type) < 0) return 0;

  Py_INCREF(&PyBobMathLpInteriorPointPredictorCorrector_Type);
  if (PyModule_AddObject(m, "LPInteriorPointPredictorCorrector",
        (PyObject *)&PyBobMathLpInteriorPointPredictorCorrector_Type) < 0) return 0;

  Py_INCREF(&PyBobMathLpInteriorPointLongstep_Type);
  if (PyModule_AddObject(m, "LPInteriorPointLongstep",
        (PyObject *)&PyBobMathLpInteriorPointLongstep_Type) < 0) return 0;

  /* imports dependencies */
  if (import_bob_blitz() < 0) return 0;
  if (import_bob_core_logging() < 0) return 0;

  return Py_BuildValue("O", m);
}

PyMODINIT_FUNC BOB_EXT_ENTRY_NAME (void) {
# if PY_VERSION_HEX >= 0x03000000
  return
# endif
    create_module();
}
