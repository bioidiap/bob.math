/**
 * @date Sat Dec 8 19:35:25 2012 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#include <algorithm>

#include <bob.math/pavx.h>

#include <bob.core/assert.h>

void bob::math::pavx(const blitz::Array<double,1>& y, blitz::Array<double,1>& ghat)
{
  bob::core::array::assertSameShape(y, ghat);
  assert(y.extent(0) > 0);
  math::pavx_(y, ghat);
}

static size_t pavx_1(const blitz::Array<double,1>& y, blitz::Array<double,1>& ghat,
  blitz::Array<size_t,1>& index, blitz::Array<size_t,1>& len)
{
  // Sets output and working arrays to 0
  index = 0;
  len = 0;
  ghat = 0.;

  // ci is the index of the interval currently considered
  // ghat(ci) is the mean of y-values within this interval
  size_t ci = 0;
  index(ci) = 0;
  len(ci) = 1;
  ghat(ci) = y(ci);
  for (int j=1; j<y.extent(0); ++j)
  {
    // a new index interval "j" is created:
    ++ci;
    index(ci) = j;
    len(ci) = 1;
    ghat(ci) = y(j);
    while (ci >= 1 && ghat(std::max((int)ci-1,0)) >= ghat(ci))
    {
      // "pool adjacent violators"
      double nw = len(ci-1) + len(ci);
      ghat(ci-1) += (len(ci) / nw) * (ghat(ci) - ghat(ci-1));
      len(ci-1) = (size_t)nw;
      --ci;
    }
  }
  return ci;
}

static void pavx_2(blitz::Array<double,1>& ghat, blitz::Array<size_t,1>& index, size_t ci)
{
  // define ghat for all indices
  int n = index.extent(0);
  while (n >= 1)
  {
    blitz::Range r((int)index(ci),n-1);
    ghat(r) = ghat(ci);
    n = index(ci);
    --ci;
  }
}

void bob::math::pavx_(const blitz::Array<double,1>& y, blitz::Array<double,1>& ghat)
{
  // Define working arrays: An interval of indices is represented by its left
  // endpoint "index" and its length "len"
  int N = y.extent(0);
  blitz::Array<size_t,1> index(N);
  blitz::Array<size_t,1> len(N);

  // First step
  size_t ci = pavx_1(y, ghat, index, len);

  // Second step
  pavx_2(ghat, index, ci);
}

blitz::Array<size_t,1> bob::math::pavxWidth(const blitz::Array<double,1>& y, blitz::Array<double,1>& ghat)
{
  // Check arguments
  bob::core::array::assertSameShape(y, ghat);
  assert(y.extent(0) > 0);

  // Define working arrays: An interval of indices is represented by its left
  // endpoint "index" and its length "len"
  int N = y.extent(0);
  blitz::Array<size_t,1> index(N);
  blitz::Array<size_t,1> len(N);

  // First step
  size_t ci = pavx_1(y, ghat, index, len);

  // Get the width vector
  blitz::Array<size_t,1> width(ci+1);
  width = len(blitz::Range(0,ci));

  // Second step
  pavx_2(ghat, index, ci);

  return width;
}

std::pair<blitz::Array<size_t,1>,blitz::Array<double,1> > bob::math::pavxWidthHeight(const blitz::Array<double,1>& y, blitz::Array<double,1>& ghat)
{
  // Check arguments
  bob::core::array::assertSameShape(y, ghat);
  assert(y.extent(0) > 0);

  // Define working arrays: An interval of indices is represented by its left
  // endpoint "index" and its length "len"
  int N = y.extent(0);
  blitz::Array<size_t,1> index(N);
  blitz::Array<size_t,1> len(N);

  // First step
  size_t ci = pavx_1(y, ghat, index, len);

  // Get the width vector
  blitz::Array<size_t,1> width(ci+1);
  width = len(blitz::Range(0,ci));
  blitz::Array<double,1> height(ci+1);
  height = ghat(blitz::Range(0,ci));

  // Second step
  pavx_2(ghat, index, ci);

  return std::make_pair(width,height);
}
