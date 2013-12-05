#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon Jun 20 16:15:36 2011 +0200
#
# Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland

"""Tests for statistical methods
"""

import os, sys
from xbob.math import scatter, scatters
import numpy
import nose.tools

def means(data):
  return numpy.mean(data, axis=0)

def py_scatters(data):
  # Step 1: compute the class means mu_c, starting from the sum_c
  mu_c = numpy.array([numpy.mean(data[k][:], axis=0) for k in range(len(data))])

  # Step 2: computes the number of elements in each class
  n_c = numpy.array([data[k].shape[0] for k in range(len(data))])

  # Step 3: computes the global mean mu
  mu = numpy.sum(mu_c.T * n_c, axis=1) / sum(data[k].shape[0] for k in range(len(data)))

  # Step 4: compute the between-class scatter Sb
  mu_c_mu = (mu_c - mu)
  Sb = numpy.dot(n_c * mu_c_mu.T, mu_c_mu)

  # Step 5: compute the within-class scatter Sw
  Sw = numpy.zeros((data[0].shape[1], data[0].shape[1]), dtype=float)
  for k in range(len(data)):
    X_c_mu_c = (data[k][:] - mu_c[k,:])
    Sw += numpy.dot(X_c_mu_c.T, X_c_mu_c)

  return (Sw, Sb, mu)

def test_scatter():

  data = numpy.random.rand(50,4)

  # This test demonstrates how to use the scatter matrix function of bob.
  S, M = scatter(data.T)
  S = S.as_ndarray()
  M = M.as_ndarray()
  S /= (data.shape[1]-1)

  # Do the same with numpy and compare. Note that with numpy we are computing
  # the covariance matrix which is the scatter matrix divided by (N-1).
  K = numpy.array(numpy.cov(data))
  M_ = means(data.T)
  assert  (abs(S-K) < 1e-10).all()
  assert  (abs(M-M_) < 1e-10).all()

def test_scatters():

  data = [
      numpy.random.rand(50,4),
      numpy.random.rand(50,4),
      numpy.random.rand(50,4),
      ]

  # Compares bob's implementation against pythonic one
  # 1. python
  Sw_, Sb_, m_ = py_scatters(data)

  # 2.a. bob
  Sw, Sb, m = scatters(data)
  # 3.a. comparison
  assert numpy.allclose(Sw, Sw_)
  assert numpy.allclose(Sb, Sb_)
  assert numpy.allclose(m, m_)

  N = data[0].shape[1]
  # 2.b. bob
  Sw = numpy.ndarray((N,N), numpy.float64)
  Sb = numpy.ndarray((N,N), numpy.float64)
  m = numpy.ndarray((N,), numpy.float64)
  scatters(data, Sw, Sb, m)
  # 3.b comparison
  assert numpy.allclose(Sw, Sw_)
  assert numpy.allclose(Sb, Sb_)
  assert numpy.allclose(m, m_)

  # 2.c. bob
  Sw = numpy.ndarray((N,N), numpy.float64)
  Sb = numpy.ndarray((N,N), numpy.float64)
  scatters(data, Sw, Sb)
  # 3.c comparison
  assert numpy.allclose(Sw, Sw_)
  assert numpy.allclose(Sb, Sb_)
