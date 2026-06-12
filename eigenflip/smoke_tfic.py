"""
smoke_tfic.py -- end-to-end torch smoke test for the TFIC encoder on a small
synthetic linear layer. Compares RTN baseline vs. CLC vs. TFIC on the EXACT
layer reconstruction energy Tr(R G R^T), with no model download.

Run (needs torch; CPU is fine):
    PYTHONPATH=. python eigenflip/smoke_tfic.py
"""
import torch

from eigenflip.quantization.state import IntegerQuantizedTensorState
from eigenflip.statistics.trust_region import LayerStats
from eigenflip.encoders.base_encoder import IdentityEncoder
from eigenflip.encoders.flip import make_clc
from eigenflip.encoders.tfic import TFICEncoder
from eigenflip.statistics.james_stein import james_stein_mean


@torch.no_grad()
def recon_energy(W_corrected, Wf, G):
    R = W_corrected - Wf
    return float((R * (R @ G)).sum())


@torch.no_grad()
def main():
    torch.manual_seed(0)
    C, d = 64, 256          # out_features x in_features
    bits, gs = 3, 128
    W = torch.randn(C, d)

    # synthetic correlated activations -> Gram G (= second moment H)
    n = 512
    A = torch.randn(n, d)
    A = A + 0.7 * A.roll(1, dims=1)          # induce cross-channel correlation
    G = (A.t() @ A) / n
    mu = A.mean(0)
    Sigma = G - torch.outer(mu, mu)

    stats = LayerStats(
        d=d, mu_hat=james_stein_mean(mu), diag_H=torch.diagonal(G).clone(),
        diag_Sigma=torch.diagonal(Sigma).clone(), U_k=None, Lam_k=None,
        eps=1e-6, Sigma=Sigma, backend="gram").build()

    def energy_of(encoder):
        st = IntegerQuantizedTensorState.from_rtn(W, bits, gs)
        out, info = encoder.apply(st, stats)
        # pad/strip consistent with G
        Wf = st.float_weights[:, :d]
        return recon_energy(out[:, :d], Wf, G), info

    e_none, _ = energy_of(IdentityEncoder())
    e_clc, i_clc = energy_of(make_clc(max_flip_frac=0.05))
    tfic = TFICEncoder(alpha=1.0, beta=1.0, eta=1.0, gamma_th=0.4,
                       kappa=2.0, gmax=6, n_stages=2, sweeps=3)
    e_tfic, i_tfic = energy_of(tfic)

    print("Layer reconstruction energy Tr(R G R^T)  (lower is better)")
    print(f"  RTN  (none) : {e_none:.6f}")
    print(f"  CLC         : {e_clc:.6f}   ({100*(e_none-e_clc)/e_none:+.2f}% vs RTN)")
    print(f"  TFIC        : {e_tfic:.6f}   ({100*(e_none-e_tfic)/e_none:+.2f}% vs RTN)")
    print("  TFIC info   :", {k: i_tfic[k] for k in
          ("total_flips", "cluster_moves", "energy_drop",
           "cluster_energy_released")})
    assert e_tfic <= e_none + 1e-6, "TFIC must not regress vs RTN baseline"
    print("OK: TFIC monotone vs RTN baseline.")


if __name__ == "__main__":
    main()
