# import Libraries of other lib packages
import bob.core

# import our own Library
import bob.extension
bob.extension.load_bob_library('bob.math', __file__)

from ._library import __version__
from ._library import *

# To get the full API documentation automatically
__all__ = dir()

def get_config():
  """Returns a string containing the configuration information.
  """

  import pkg_resources
  from .version import externals

  packages = pkg_resources.require(__name__)
  this = packages[0]
  deps = packages[1:]

  retval =  "%s: %s (%s)\n" % (this.key, this.version, this.location)
  retval += "  - c/c++ dependencies:\n"
  for k in sorted(externals): retval += "    - %s: %s\n" % (k, externals[k])
  retval += "  - python dependencies:\n"
  for d in deps: retval += "    - %s: %s (%s)\n" % (d.key, d.version, d.location)

  return retval.strip()

# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]
