#!/usr/bin/env python
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
# Sun Jan  15 19:12:43 CET 2017
#

"""
Tests GSVD

 Basically these tests test the GSVD relation.
 Given 2 matrices A and B  GSVD(A,B) = [U,V,X,C,S] where,

  A= (X * C.T * U^T)^T and 
  B= (X * S.T * V^T)^T and
  
  C**2 + S**2 = 1

"""

import bob.math
import numpy
import nose.tools
numpy.random.seed(10)


def gsvd_relations(A,B):
  [U,V,X,C,S] = bob.math.gsvd(A, B)

  # Cheking the relation  C**2 + S**2 = 1
  I = numpy.eye(A.shape[1])
  I_check = numpy.dot(C.T, C) + numpy.dot(S.T, S)
  nose.tools.eq_( (abs(I-I_check) < 1e-10).all(), True )

  # Cheking the relation A= (X * C.T * U^T)^T
  A_check = numpy.dot(numpy.dot(X,C.T),U.T).T
  nose.tools.eq_( (abs(A-A_check) < 1e-10).all(), True )

  # Cheking the relation B= (X * S.T * V^T)^T 
  B_check = numpy.dot(numpy.dot(X,S.T),V.T).T
  nose.tools.eq_( (abs(B-B_check) < 1e-10).all(), True )



def test_first_case():
  """
  Testing the first scenario of gsvd:
  M-K-L >= 0 (check http://www.netlib.org/lapack/explore-html/d1/d7e/group__double_g_esing_ga4a187519e5c71da3b3f67c85e9baf0f2.html#ga4a187519e5c71da3b3f67c85e9baf0f2)  
  """

  A = numpy.random.rand(10,10)
  B = numpy.random.rand(790,10)

  gsvd_relations(A, B)


def test_second_case():
  """
  Testing the second scenario of gsvd:
  M-K-L < 0 (check http://www.netlib.org/lapack/explore-html/d1/d7e/group__double_g_esing_ga4a187519e5c71da3b3f67c85e9baf0f2.html#ga4a187519e5c71da3b3f67c85e9baf0f2)  
  """

  A = numpy.random.rand(4,5)
  B = numpy.random.rand(11,5)

  gsvd_relations(A, B)


def test_corner_case():
  """
  Testing when P <= N.
    
  """

  A = numpy.random.rand(25, 25)
  B = numpy.random.rand(25, 25)
  gsvd_relations(A, B)



