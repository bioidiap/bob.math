/**
 * @author Manuel Guenther <manuel.guenther@idiap.ch>
 * @date Mon Aug 18 16:57:06 CEST 2014
 *
 * @brief General directives for all modules in bob.math
 */

#ifndef BOB_MATH_CONFIG_H
#define BOB_MATH_CONFIG_H

/* Macros that define versions and important names */
#define BOB_MATH_API_VERSION 0x0200

#ifdef BOB_IMPORT_VERSION

  /***************************************
  * Here we define some functions that should be used to build version dictionaries in the version.cpp file
  * There will be a compiler warning, when these functions are not used, so use them!
  ***************************************/

  #include <Python.h>
  #include <boost/preprocessor/stringize.hpp>

  /**
   * bob.math c/c++ api version
   */
  static PyObject* bob_math_version() {
    return Py_BuildValue("{ss}", "api", BOOST_PP_STRINGIZE(BOB_MATH_API_VERSION));
  }

#endif // BOB_IMPORT_VERSION

#endif /* BOB_MATH_CONFIG_H */
