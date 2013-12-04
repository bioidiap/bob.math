/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue  3 Dec 14:27:26 2013 CET 
 *
 * @brief C/C++ API for bob::math
 */

#ifndef XBOB_MATH_H
#define XBOB_MATH_H

/* Define Module Name and Prefix for other Modules
   Note: We cannot use XBOB_EXT_* macros here, unfortunately */
#define XBOB_MATH_PREFIX    "xbob.math"
#define XBOB_MATH_FULL_NAME "xbob.math._library"

#include <xbob.math/config.h>
#include <Python.h>

/*******************
 * C API functions *
 *******************/

/**************
 * Versioning *
 **************/

#define PyXbobMath_APIVersion_NUM 0
#define PyXbobMath_APIVersion_TYPE int

/* Total number of C API pointers */
#define PyXbobMath_API_pointers 1

#ifdef XBOB_MATH_MODULE

  /* This section is used when compiling `xbob.math' itself */

  /**************
   * Versioning *
   **************/

  extern int PyXbobMath_APIVersion;

#else

  /* This section is used in modules that use `xbob.math's' C-API */

/************************************************************************
 * Macros to avoid symbol collision and allow for separate compilation. *
 * We pig-back on symbols already defined for NumPy and apply the same  *
 * set of rules here, creating our own API symbol names.                *
 ************************************************************************/

#  if defined(PY_ARRAY_UNIQUE_SYMBOL)
#    define XBOB_MATH_MAKE_API_NAME_INNER(a) XBOB_MATH_ ## a
#    define XBOB_MATH_MAKE_API_NAME(a) XBOB_MATH_MAKE_API_NAME_INNER(a)
#    define PyXbobIo_API XBOB_MATH_MAKE_API_NAME(PY_ARRAY_UNIQUE_SYMBOL)
#  endif

#  if defined(NO_IMPORT_ARRAY)
  extern void **PyXbobMath_API;
#  else
#    if defined(PY_ARRAY_UNIQUE_SYMBOL)
  void **PyXbobMath_API;
#    else
  static void **PyXbobMath_API=NULL;
#    endif
#  endif

  static void **PyXbobMath_API;

  /**************
   * Versioning *
   **************/

# define PyXbobMath_APIVersion (*(PyXbobMath_APIVersion_TYPE *)PyXbobMath_API[PyXbobMath_APIVersion_NUM])

# if !defined(NO_IMPORT_ARRAY)

  /**
   * Returns -1 on error, 0 on success.
   */
  static int import_xbob_math(void) {

    PyObject *c_api_object;
    PyObject *module;

    module = PyImport_ImportModule(XBOB_MATH_FULL_NAME);

    if (module == NULL) return -1;

    c_api_object = PyObject_GetAttrString(module, "_C_API");

    if (c_api_object == NULL) {
      Py_DECREF(module);
      return -1;
    }

#   if PY_VERSION_HEX >= 0x02070000
    if (PyCapsule_CheckExact(c_api_object)) {
      PyXbobMath_API = (void **)PyCapsule_GetPointer(c_api_object, 
          PyCapsule_GetName(c_api_object));
    }
#   else
    if (PyCObject_Check(c_api_object)) {
      PyXbobMath_API = (void **)PyCObject_AsVoidPtr(c_api_object);
    }
#   endif

    Py_DECREF(c_api_object);
    Py_DECREF(module);

    if (!PyXbobMath_API) {
      PyErr_SetString(PyExc_ImportError, "cannot find C/C++ API "
#   if PY_VERSION_HEX >= 0x02070000
          "capsule"
#   else
          "cobject"
#   endif
          " at `" XBOB_MATH_FULL_NAME "._C_API'");
      return -1;
    }

    /* Checks that the imported version matches the compiled version */
    int imported_version = *(int*)PyXbobMath_API[PyXbobMath_APIVersion_NUM];

    if (XBOB_MATH_API_VERSION != imported_version) {
      PyErr_Format(PyExc_ImportError, XBOB_MATH_FULL_NAME " import error: you compiled against API version 0x%04x, but are now importing an API with version 0x%04x which is not compatible - check your Python runtime environment for errors", XBOB_MATH_API_VERSION, imported_version);
      return -1;
    }

    /* If you get to this point, all is good */
    return 0;

  }

# endif //!defined(NO_IMPORT_ARRAY)

#endif /* XBOB_MATH_MODULE */

#endif /* XBOB_MATH_H */
