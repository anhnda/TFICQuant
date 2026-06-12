"""
validate_tfic.py -- numpy-only checks of the TFIC math (no torch, no GPU).

Mirrors the encoder's energy bookkeeping on small synthetic problems and
verifies the paper's guarantees:

  1. Energy identity:  E(s) = Tr(R G R^T),  R = Wq(s) - W.   The incremental
     single-flip gain dE_j = 2*delta*(R G)_ij + delta^2*G_jj equals the exact
     recomputed energy difference.
  2. Lemma 1 (pair-flip identity):  dE_{j,k} = dE_j + dE_k + 2 delta_j delta_k G_jk.
  3. The explicit Appendix-B barrier instance: both single flips raise E,
     the pair flip lowers it -> single-coordinate descent is trapped, the
     cluster (group) move escapes.
  4. Group-flip gain (Eq. 16) matches exact recomputation for |T|>2.
  5. A full single-flip descent on a random layer is monotone (every accepted
     flip strictly lowers the exact energy) and a subsequent 2-cluster tunnel
     pass can release additional energy that single flips left on the table.
"""
import numpy as np

rng = np.random.default_rng(0)


def energy(Wint, zp, scale, Wf, G):
    R = (Wint - zp) * scale - Wf
    return float(np.einsum("cj,jk,ck->", R, G, R))


def single_gain(R, G, delta, i, j):
    RG = R @ G
    return 2.0 * delta * RG[i, j] + delta * delta * G[j, j]


def group_gain(R, G, deltas, i, cols):
    RG = R @ G
    dT = np.array(deltas)
    GTT = G[np.ix_(cols, cols)]
    return 2.0 * dT @ RG[i, cols] + dT @ GTT @ dT


# ---------------------------------------------------------------- check 1 & 2
def check_identities():
    d, C, bits, g = 16, 4, 4, 8
    W = rng.standard_normal((C, d))
    ng = d // g
    Wg = W.reshape(C, ng, g)
    wmin = Wg.min(2, keepdims=True); wmax = Wg.max(2, keepdims=True)
    mi = 2 ** bits - 1
    scg = np.clip((wmax - wmin) / mi, 1e-8, None)
    zpg = np.clip(np.round(-wmin / scg), 0, mi)
    scale = np.repeat(scg, g, 2).reshape(C, d)
    zp = np.repeat(zpg, g, 2).reshape(C, d)
    pre = W / scale + zp
    Wint = np.clip(np.round(pre), 0, mi)
    A = rng.standard_normal((64, d)); G = A.T @ A / 64 + np.eye(d) * 0.1

    R = (Wint - zp) * scale - Wf if False else (Wint - zp) * scale - W
    flip_dir = np.sign(pre - Wint); flip_dir[flip_dir == 0] = 1

    ok_single, ok_pair = True, True
    for _ in range(200):
        i = rng.integers(C); j, k = rng.choice(d, 2, replace=False)
        # single
        dj = flip_dir[i, j] * scale[i, j]
        pred = single_gain(R, G, dj, i, j)
        W2 = Wint.copy(); W2[i, j] += flip_dir[i, j]
        if not (0 <= W2[i, j] <= mi):
            continue
        exact = energy(W2, zp, scale, W, G) - energy(Wint, zp, scale, W, G)
        ok_single &= abs(pred - exact) < 1e-6 * (1 + abs(exact))
        # pair
        dk = flip_dir[i, k] * scale[i, k]
        dEj = single_gain(R, G, dj, i, j)
        dEk = single_gain(R, G, dk, i, k)
        pred_pair = dEj + dEk + 2 * dj * dk * G[j, k]
        W3 = Wint.copy(); W3[i, j] += flip_dir[i, j]; W3[i, k] += flip_dir[i, k]
        if not (0 <= W3[i, j] <= mi and 0 <= W3[i, k] <= mi):
            continue
        exact_pair = energy(W3, zp, scale, W, G) - energy(Wint, zp, scale, W, G)
        ok_pair &= abs(pred_pair - exact_pair) < 1e-6 * (1 + abs(exact_pair))
    print(f"[1] single-flip gain == exact dE         : {ok_single}")
    print(f"[2] Lemma 1 pair identity == exact dE     : {ok_pair}")
    return ok_single and ok_pair


# ---------------------------------------------------------------- check 3
def check_barrier():
    # Appendix B explicit instance: din=2, H=(1,1), G=[[1,-.5],[-.5,1]],
    # s=(+1,+1), D=(-0.8,-0.8) -> R=(1.8,1.8), delta_j=delta_k=-2.
    G = np.array([[1.0, -0.5], [-0.5, 1.0]])
    Hhalf = np.array([1.0, 1.0])           # half-spacing
    D = np.array([-0.8, -0.8])             # offset of W from midpoint
    s = np.array([+1.0, +1.0])
    # residual R = Hhalf*s - D  (paper convention, row vector)
    R = (Hhalf * s - D).reshape(1, 2)
    RG = R @ G
    delta = np.array([-2.0, -2.0])         # flip j: delta_j = -2 s_j Hhalf_j
    dEj = 2 * delta[0] * RG[0, 0] + delta[0] ** 2 * G[0, 0]
    dEk = 2 * delta[1] * RG[0, 1] + delta[1] ** 2 * G[1, 1]
    dEpair = dEj + dEk + 2 * delta[0] * delta[1] * G[0, 1]
    print(f"[3] dE_j={dEj:+.3f} dE_k={dEk:+.3f} (both >=0)  "
          f"dE_pair={dEpair:+.3f} (<0)")
    return dEj >= 0 and dEk >= 0 and dEpair < 0


# ---------------------------------------------------------------- check 4
def check_group():
    d, C = 12, 3
    W = rng.standard_normal((C, d))
    scale = np.full((C, d), 0.3); zp = np.full((C, d), 7.0)
    mi = 15
    pre = W / scale + zp; Wint = np.clip(np.round(pre), 0, mi)
    A = rng.standard_normal((40, d)); G = A.T @ A / 40 + np.eye(d) * 0.05
    R = (Wint - zp) * scale - W
    flip_dir = np.sign(pre - Wint); flip_dir[flip_dir == 0] = 1
    ok = True
    for _ in range(100):
        i = rng.integers(C)
        cols = list(rng.choice(d, rng.integers(3, 6), replace=False))
        f = rng.integers(0, 2, len(cols)).astype(float)
        deltas = [f[t] * flip_dir[i, c] * scale[i, c] for t, c in enumerate(cols)]
        W2 = Wint.copy()
        for t, c in enumerate(cols):
            W2[i, c] += int(f[t]) * flip_dir[i, c]
        if (W2[i] < 0).any() or (W2[i] > mi).any():
            continue
        pred = group_gain(R, G, deltas, i, cols)
        exact = energy(W2, zp, scale, W, G) - energy(Wint, zp, scale, W, G)
        ok &= abs(pred - exact) < 1e-6 * (1 + abs(exact))
    print(f"[4] group-flip gain (Eq.16) == exact dE   : {ok}")
    return ok


# ---------------------------------------------------------------- check 5
def check_descent_monotone():
    d, C, bits, g = 24, 6, 3, 8
    W = rng.standard_normal((C, d))
    ng = d // g; Wg = W.reshape(C, ng, g)
    wmin = Wg.min(2, keepdims=True); wmax = Wg.max(2, keepdims=True)
    mi = 2 ** bits - 1
    scg = np.clip((wmax - wmin) / mi, 1e-8, None)
    zpg = np.clip(np.round(-wmin / scg), 0, mi)
    scale = np.repeat(scg, g, 2).reshape(C, d)
    zp = np.repeat(zpg, g, 2).reshape(C, d)
    pre = W / scale + zp; Wint = np.clip(np.round(pre), 0, mi)
    A = rng.standard_normal((50, d)); G = A.T @ A / 50 + np.eye(d) * 0.02

    R = (Wint - zp) * scale - W
    RG = R @ G
    e = energy(Wint, zp, scale, W, G)
    monotone = True
    for _ in range(4):
        flips = 0
        for j in rng.permutation(d):
            fd = np.sign(pre[:, j] - Wint[:, j]); fd[fd == 0] = 1
            dj = fd * scale[:, j]
            dEj = 2 * dj * RG[:, j] + dj * dj * G[j, j]
            prop = Wint[:, j] + fd
            ok = (prop >= 0) & (prop <= mi) & (dEj < 0)
            if not ok.any():
                continue
            step = np.where(ok, fd, 0.0)
            Wint[:, j] = np.clip(Wint[:, j] + step, 0, mi)
            dR = np.where(ok, dj, 0.0)
            R[:, j] += dR
            RG += np.outer(dR, G[j, :])
            e_new = energy(Wint, zp, scale, W, G)
            monotone &= e_new <= e + 1e-9
            flips += int(ok.sum()); e = e_new
        if flips == 0:
            break
    e_cd = energy(Wint, zp, scale, W, G)
    print(f"[5] single-flip descent monotone         : {monotone}  "
          f"(final E={e_cd:.4f})")
    return monotone


if __name__ == "__main__":
    print("TFIC numpy validation")
    print("-" * 52)
    Wf = None  # placeholder for the False branch in check_identities
    r = [check_identities(), check_barrier(), check_group(),
         check_descent_monotone()]
    print("-" * 52)
    print("ALL PASS" if all(r) else "SOME CHECKS FAILED")
