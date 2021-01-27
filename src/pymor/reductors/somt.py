# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
from pymor.algorithms.gram_schmidt import gram_schmidt, gram_schmidt_biorth
from pymor.algorithms.somddpa import somddpa
from pymor.core.base import BasicObject
from pymor.models.iosys import SecondOrderModel
from pymor.parameters.base import Mu
from pymor.reductors.basic import SOLTIPGReductor


class SOMTReductor(BasicObject):
    """Modal Truncation reductor for second order model.

    Parameters
    ----------
    fom
        The full-order |SecondOrderModel| to reduce.
    mu
        |Parameter values|.
    """

    def __init__(self, fom, mu=None):
        assert isinstance(fom, SecondOrderModel)
        if not isinstance(mu, Mu):
            mu = fom.parameters.parse(mu)
        assert fom.parameters.assert_compatible(mu)
        self.fom = fom
        self.mu = mu
        self.V = None
        self._pg_reductor = None

    def reduce(self, r=None, projection='orth', which='NR', method_options=None):
        """Modal Truncation for Second Order Systems.

        Parameters
        ----------
        r
            Order of the reduced model.
        projection
            Projection method used:

            - `'orth'`: projection matrices are orthogonalized with
              respect to the Euclidean inner product
            - `'biorth'`: projection matrices are biorthogolized with
              respect to the M product
        method_options
            Optional dict with more options for the samdp algorithm.

        Returns
        -------
        rom
            Reduced-order model.
        """
        assert 0 < r < self.fom.order
        assert projection in ('orth', 'biorth')
        assert method_options is None or isinstance(method_options, dict)
        if not method_options:
            method_options = {}

        # find the reduced model
        if self.fom.parametric:
            fom = self.fom.with_(**{op: getattr(self.fom, op).assemble(mu=self.mu)
                                    for op in ['M', 'E', 'K', 'B', 'Cp', 'Cv']})
        else:
            fom = self.fom

        pos, neg, rev, res = somddpa(fom.M, fom.E, fom.K,
                                     fom.B.as_range_array(),
                                     fom.Cp.as_source_array(),
                                     r, **method_options)

        self.V = rev

        if projection == 'orth':
            gram_schmidt(self.V, atol=0, rtol=0, copy=False)

        elif projection == 'biorth':
            gram_schmidt_biorth(self.V, self.V, product=fom.M, copy=False)

        self._pg_reductor = SOLTIPGReductor(fom, self.V, self.V, projection == 'biorth')
        rom = self._pg_reductor.reduce()
        return rom

    def reconstruct(self, u):
        """Reconstruct high-dimensional vector from reduced vector `u`."""
        return self._pg_reductor.reconstruct(u)
