# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import numpy.lib.scimath as cnp
import scipy.linalg as spla

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.algorithms.projection import project
from pymor.algorithms.to_matrix import to_matrix
from pymor.core.logger import getLogger
from pymor.operators.block import BlockDiagonalOperator, BlockOperator
from pymor.operators.constructions import IdentityOperator, ZeroOperator
from pymor.operators.interface import Operator
from pymor.vectorarrays.block import BlockVectorArray, BlockVectorSpace


def somddpa(M, E, K, B, C, nwanted, init_shift=-1, which='LR',
            kmin=10, krestart=30, maxiter=100, lin=3, tdefl=False, reortho=True, tol=1e-8, imagtol=1e-8):
    """Compute the dominant pole triplets and residues of the transfer function of a Second Order System based on [SSW19]_.

    This function implements a version of the subspace accelerated
    quadratic MIMO dominant pole algorithm for modally-damped second-order
    systems with the assumptions M, E, K symmetric positive definite and
    :math:`E*M^{-1}*K = K*M^{-1}*E`. It computes poles of the transfer function

    .. math::
       H(s) = C * (s^2 * M + s * E + K)^{-1} * B

    that are dominant with respect to a certain measure. The poles are
    computed in pairs lambda_p, lambda_m such that

    .. math::
       lambda_p = a + b, lambda_m = a - b

    are eigenvalues of :math:`(s^2 * M + s * E + K)X = 0`, with X containing the
    corresponding real eigenvectors.

    Parameters
    ----------
    M
        The |Operator| M.
    E
        The |Operator| E.
    K
        The |Operator| K.
    B
        The operator B as a |VectorArray| from `M.source`.
    C
        The operator C as a |VectorArray| from `M.source`.
    nwanted
        The number of dominant poles that should be computed.
    init_shift
        Initial shift.
    which
        A string specifying the strategy by which the dominant poles and residues are selected.
        Possible values are:

        - `'LR'`: select poles with largest norm(residual) / abs(Re(pole))
        - `'LS'`: select poles with largest norm(residual) / abs(pole)
        - `'LP'`: select poles with largest norm(residual) / product pair
        - `'LRP'`: select poles with largest norm(residual) / product of real parts
        - `'LRPI'`: as LRP restricted to compex conjugate poles
    kmin
        Minimal dimension of search space after performing a restart.
    krestart
        Maximum dimension of search space before performing a restart.
    maxiter
        The maximum number of iterations.
    lin
        A number specifying which first-order formulation to use for turbo deflation.
        Possible values and the corresponding models are:

         -1: first companion form with J = -K
         -2: first companion form with J = I
         -3: second companion form
    tdefl
        If `True`, deflation method uses the first-order realization.
    reortho
        If `True`, search space is reorthonormalized after poles were found or restart.
    tol
        Tolerance for the residual of the poles.
    imagtol
        Relative tolerance for imaginary parts of pairs of complex conjugate eigenvalues.

    Returns
    -------
    pos
        A 1D |NumPy array| containing the positive part of the computed dominant poles.
    neg
        A !D |NumPy array| containing the negative part of the computed dominant poles.
    X
        A |VectorArray| containing the right eigenvectors of the computed poles.
    residues
        A |NumPy array| containing the computed residuals.

    """

    logger = getLogger('pymor.algorithms.somddpa.somddpa')

    assert isinstance(M, Operator) and M.linear
    assert not M.parametric
    assert M.source == M.range
    assert isinstance(E, Operator) and E.linear
    assert not E.parametric
    assert E.source == E.range
    assert isinstance(K, Operator) and M.linear
    assert not K.parametric
    assert K.source == K.range
    assert lin in (1, 2, 3)
    assert which in ('LR', 'LS', 'LP', 'LRP', 'LRPI')

    assert B in M.source
    assert C in M.source

    lambda_p = np.zeros(nwanted, dtype=complex)
    lambda_n = np.zeros(nwanted, dtype=complex)
    Xfound = M.source.empty()
    exp_space = BlockVectorSpace([M.source, M.source])
    Xfound_aug = exp_space.empty()
    EX_found_aug = exp_space.empty()
    if lin == 1:
        Eaug = BlockDiagonalOperator([-K, M])
    elif lin == 2:
        Eaug = BlockDiagonalOperator([IdentityOperator(K.source), M])
    elif lin == 3:
        Eaug = BlockOperator([[E, M], [M, ZeroOperator(E.range, E.source)]])

    V = M.source.empty()
    iteration = 0
    converged = 0
    st = init_shift

    s2MsEK = st * st * M + st * E + K
    s2MsEKB = s2MsEK.apply_inverse(B)
    Hs = C.inner(s2MsEKB)

    y_all, d_all, u_all = spla.svd(Hs)

    u = u_all.conj()[0]
    y = y_all[:, 0]
    d = d_all[0]

    if tdefl:
        C_defl = BlockVectorArray([C.conj(), C.zeros(len(C))], exp_space)
        if lin == 3:
            B_defl = BlockVectorArray([B, B.zeros(len(B))], exp_space)
        else:
            B_defl = BlockVectorArray([B.zeros(len(B)), B], exp_space)
        Bk_d = K.apply_inverse(B_defl.block(0))
        Ck_d = K.apply_inverse_adjoint(C_defl.block(0))

    else:
        B_defl = B.copy()
        C_defl = C.conj().copy()

    while iteration < maxiter:
        iteration += 1

        s2MsEK = st * st * M + st * E + K

        if tdefl:
            v = s2MsEK.apply_inverse(
                (st*B_defl.block(1)+B_defl.block(0)).lincomb(u.T))
            w = s2MsEK.apply_inverse_adjoint(
                (st.conjugate()*C_defl.block(1)+C_defl.block(0)).lincomb(y.T))
            if lin != 3:
                v.axpy(1, -Bk_d.lincomb(u.T))
                v.scal(1 / st)
                w.axpy(1, -Ck_d.lincomb(y.T))
                w.scal(1 / st)
        else:
            v = s2MsEK.apply_inverse(B_defl.lincomb(u.T))
            w = s2MsEK.apply_inverse_adjoint(C_defl.lincomb(y.T))

            newton_upd = complex(st - d
                                 / (w.conj().inner(2 * st * M.apply(v) + E.apply(v))))

            tmp = BlockVectorArray([v, newton_upd * v], exp_space)
            tmp = _rmgs_two(tmp, Xfound_aug, EX_found_aug)
            v = tmp.block(0)

            tmp = BlockVectorArray([w, newton_upd * w], exp_space)
            tmp = _rmgs_two(tmp, Xfound_aug, EX_found_aug)
            w = tmp.block(0)

        k = len(V)
        V.append(v.real)
        V.append(v.imag)
        V.append(w.real)
        V.append(w.imag)
        V, _ = spla.qr(V.to_numpy().T, mode='economic')
        V = M.source.from_numpy(V.T)
        # gram_schmidt(V, copy=False)
        dk = len(V) - k  # necessary?
        if dk <= 0:
            logger.error('Stagnation')

        Mr = project(M, V, V)
        Kr = project(K, V, V)
        Er = project(E, V, V)

        k = k + dk

        pos, neg, X = _solve_mdeig_problem(Mr, Er, Kr, V, B_defl, C_defl, tdefl=tdefl, which=which)

        found = True

        while found:
            v = V.lincomb(X[:, 0])
            v.scal(1/v.norm())
            theta = pos[0]

            st = complex(pos[0])
            nr = (st * st * M.apply(v) + st
                  * E.apply(v) + K.apply(v)).norm()

            logger.info(
                f'Step: {iteration}, Theta: {theta:.5e}, Residual: {nr[0]:.5e}')
            found = (nr < tol)
            if found:
                if abs(pos[0].imag) / abs(pos[0]) < imagtol:
                    logger.info('Real pole found.')
                    logger.info(f'Real Pole: {pos[0]:.5e}, {neg[0]:.5e}')
                    lambda_p[converged] = pos[0].real
                    lambda_n[converged] = neg[0].real
                    converged += 1

                    v_pos = BlockVectorArray([v, float(pos[0].real)*v], exp_space)
                    v_neg = BlockVectorArray([v, float(neg[0].real)*v], exp_space)
                else:
                    logger.info('Complex conjugated pair found.')
                    logger.info(f'Conjugate Pole: {pos[0]:.5e}, {neg[0]:.5e}')
                    lambda_p[converged] = pos[0]
                    lambda_n[converged] = neg[0]
                    converged += 1

                    v_pos = BlockVectorArray([v, float(pos[0].real)*v], exp_space)
                    v_neg = BlockVectorArray([v, float(pos[0].imag)*v], exp_space)

                Xfound_aug.append(v_pos)
                Xfound_aug.append(v_neg)
                Xfound.append(v)

                ex_found = Eaug.apply(Xfound_aug[-2:])
                nc = Xfound_aug[-2:].inner(ex_found)
                ex_found_scal = exp_space.from_numpy(
                    spla.solve(nc.conj(), ex_found.to_numpy().conj()))
                EX_found_aug.append(ex_found_scal)

                pos = pos[1:]
                neg = neg[1:]
                V = V.lincomb(X[:, 1:].T)
                k = len(V)

                if tdefl:
                    B_defl = B_defl - \
                        EX_found_aug.lincomb(Xfound_aug.conj().inner(B_defl).T)
                    C_defl = C_defl - \
                        EX_found_aug.lincomb(Xfound_aug.conj().inner(C_defl).T)

                    if lin != 3:
                        Bk_d = K.apply_inverse(B_defl.block(0))
                        Ck_d = K.apply_inverse_adjoint(C_defl.block(0))

                if reortho:
                    gram_schmidt(V, atol=0, rtol=0, copy=False)

                Mr = project(M, V, V)
                Kr = project(K, V, V)
                Er = project(E, V, V)

                if converged >= nwanted:
                    break

                found = (k > 0)

                if found:
                    pos, neg, X = _solve_mdeig_problem(Mr, Er, Kr, V, B_defl, C_defl, tdefl=tdefl, which=which)

                if converged < nwanted and not(found):
                    s2MsEK = st * st * M + st * E + K
                    s2MsEKB = s2MsEK.apply_inverse(B)
                    Hs = C.inner(s2MsEKB)

                    y_all, d_all, u_all = spla.svd(Hs)

                    u = u_all.conj()[0]
                    y = y_all[:, 0]
                    d = d_all[0]
            elif k >= krestart:

                logger.info('Perform restart..')

                idx = list(range(kmin))
                Ld = np.zeros((2*len(pos), 2*len(pos)), dtype=complex)
                Xt = np.zeros((len(pos), 2*len(pos)), dtype=complex)

                for h in range(len(pos)):
                    Ld[2 * h: 2 * h+2, 2 * h: 2 * h+2] = np.array(
                        [[pos.real[h], pos.imag[h]], [neg.real[h], neg.imag[h]]])
                    Xt[:, 2*h] = X[:, h]
                    Xt[:, 2*h+1] = X[:, h]

                RB = M.apply(V.lincomb((Xt @ (Ld @ Ld)).T)) + \
                    E.apply(V.lincomb((Xt @ Ld).T)) + \
                    K.apply(V.lincomb(Xt.T))

                nres = RB.norm()
                minidx = np.argmin(nres)

                minidx = (minidx - 1) / 2 if np.mod(minidx, 2) else minidx / 2
                if minidx >= kmin:
                    idx = [int(minidx)] + list(range(kmin-1))

                pos = pos[idx]
                neg = neg[idx]
                V = V.lincomb(X[:, idx].T)

                if reortho:
                    gram_schmidt(V, atol=0, rtol=0, copy=False)

                s2MsEK = st * st * M + st * E + K
                s2MsEKB = s2MsEK.apply_inverse(B)
                Hs = C.inner(s2MsEKB)

                y_all, d_all, u_all = spla.svd(Hs)

                u = u_all.conj()[0]
                y = y_all[:, 0]
                d = d_all[0]

        if (converged >= nwanted) or (iteration >= maxiter):
            if iteration >= maxiter:
                logger.warning('No convergence in maxiter iterations.')

            lambda_p = lambda_p[:converged]
            lambda_n = lambda_n[:converged]

            absres = np.empty(len(lambda_p))
            residues = []
            for i in range(len(lambda_p)):
                residues.append(np.sqrt(
                    abs(lambda_p[i]*lambda_n[i]))*C.inner(Xfound[i]) @ Xfound[i].conj().inner(B))
                absres[i] = spla.norm(residues[-1], ord=2)
            residues = np.array(residues)

            if which == 'LR':
                idx = np.argsort(-absres / np.abs(np.real(lambda_p)))
            elif which == 'LS':
                idx = np.argsort(-absres / np.abs(lambda_p))
            elif which == 'LP':
                idx = np.argsort(-absres / np.abs(lambda_p * lambda_n))
            elif which == 'LRP':
                idx = np.argsort(
                    -absres / abs(np.real(lambda_p) * np.real(lambda_n)))
            elif which == 'LRPI':
                idx = np.argsort(-absres * abs(np.imag(lambda_p))
                                 / abs(np.real(lambda_p) * np.real(lambda_n)))
            else:
                raise ValueError('Unknown SAMDP selection strategy.')

            residues = residues[idx]
            lambda_p = lambda_p[idx]
            lambda_n = lambda_n[idx]
            X = Xfound[idx]
            break

    return lambda_p, lambda_n, X, residues


def _solve_mdeig_problem(Mr, Er, Kr, V, B, C, tdefl=False, which='LRP'):
    """Solve reduced eigenvalue problem and return poles sorted according to specified dominance measure.

    Parameters
    ----------
    Mr
        The reduced |Operator| Mr.
    Er
        The reduced |Operator| Er.
    Kr
        The reduced |Operator| Kr.
    V
        A |VectorArray| the projection matrix.
    B
        The |VectorArray| B from the corresponding LTI system modified by deflation.
    C
        The |VectorArray| C from the corresponding LTI system modified by deflation.
    which
        A string that indicates which poles to select. See :func:`somddpa`.

    Returns
    -------
    pos
        A 1D |Numpy array| containing the positive part of the eigenvalue pairs sorted according to the chosen strategy.
    neg
        A 1D |Numpy array| containing the negative part of the eigenvalue pairs sorted according to the chosen strategy.
    X
        A |Numpy array| containing the eigenvectors of the computed poles.

    """

    n = Mr.source.dim
    pos = np.zeros(n, dtype=complex)
    neg = np.zeros(n, dtype=complex)
    Xi = np.zeros(n)
    Omega2, X = spla.eig(to_matrix(Kr, format='dense'), to_matrix(Mr, format='dense'))
    Omega = cnp.sqrt(Omega2)

    X = X @ np.diag(1/cnp.sqrt(np.diag(X.T @ (to_matrix(Mr, format='dense') @ X))))
    X = X @ np.diag(1/cnp.sqrt(Omega))
    X = X.real

    Xi = np.diag(0.5 * (X.T @ (to_matrix(Er, format='dense') @ X)))

    for k in range(n):
        if abs(Xi[k] - 1) < 1.0e-09:
            pos[k] = -Omega[k]
            neg[k] = -Omega[k]
        else:
            pos[k] = -Xi[k] * Omega[k] + Omega[k] * \
                cnp.sqrt((Xi[k] + 1) * (Xi[k] - 1))
            neg[k] = -Xi[k] * Omega[k] - Omega[k] * \
                cnp.sqrt((Xi[k] + 1) * (Xi[k] - 1))

    Xs = V.lincomb(X.T)
    residues = []

    for i in range(len(pos)):
        omega = cnp.sqrt((pos[i] * neg[i]).real)
        if tdefl:
            Xaug = BlockVectorArray([Xs[i], complex(pos[i])*Xs[i]], C.space)
        else:
            Xaug = Xs[i].copy()

        residues.append(spla.norm(omega * C.inner(Xaug) @ Xaug.conj().inner(B)))

    residues = np.array(residues)

    if which == 'LR':
        idx = np.argsort(-residues / np.abs(np.real(pos)))
    elif which == 'LS':
        idx = np.argsort(-residues / np.abs(pos))
    elif which == 'LP':
        idx = np.argsort(-residues / np.abs(pos * neg))
    elif which == 'LRP':
        idx = np.argsort(
            -residues / abs(np.real(pos) * np.real(neg)))
    elif which == 'LRPI':
        idx = np.argsort(-residues * abs(np.imag(pos))
                         / abs(np.real(pos) * np.real(neg)))
    else:
        raise ValueError('Unknown SAMDP selection strategy.')

    residues = residues[idx]
    pos = pos[idx]
    neg = neg[idx]
    X = X[:, idx]

    return pos, neg, X


def _rmgs_two(v, Q, Qt):

    if len(Q) == 0:
        return v
    else:
        v = v - Q.lincomb(Qt.inner(v).T)
        v = v - Q.lincomb(Qt.inner(v).T)
        return v
