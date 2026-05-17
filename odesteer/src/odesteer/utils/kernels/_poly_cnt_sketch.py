import torch
from torch import nn, Tensor

class PolyCntSketch(nn.Module):
    def __init__(
        self,
        degree: int = 2,
        n_components: int = 100,
        gamma: float = 1.0,
        coef0: float = 0.0,
    ):
        super().__init__()
        assert degree >= 1, "degree must be >= 1"
        assert n_components >= 1, "n_components must be >= 1"
        self.degree = int(degree)
        self.n_components = int(n_components)
        self.gamma = float(gamma)
        self.coef0 = float(coef0)

    @staticmethod
    def _ensure_fitted(buf):
        if buf is None:
            raise RuntimeError("Call .fit(X) first to initialize sketch hashes.")

    def _ext_feature_count(self) -> int:
        nf = int(self.n_features_)
        return nf + (1 if self.coef0 != 0 else 0)

    def extra_repr(self) -> str:
        return (f"degree={self.degree}, n_components={self.n_components}, "
                f"gamma={self.gamma}, coef0={self.coef0}, "
                f"n_features_={self.n_features_})")

    # ----------------------- API -------------------------
    def fit(self, X: Tensor):        
        self.n_features_ = X.shape[0] if X.dim() == 1 else X.shape[1]
        n_features_ext = self._ext_feature_count()

        # Indices: long; Signs: int8 in {-1, +1}
        indexHash = torch.randint(
            low = 0, high = self.n_components,
            size = (self.degree, n_features_ext),
            dtype = torch.long, device = X.device,
        )
        bitHash = (torch.randint(
            low = 0, high = 2,
            size = (self.degree, n_features_ext),
            dtype = torch.int8, device = X.device,
        ) * 2 - 1)  # -> {-1, +1} as int8
        
        self.register_buffer("indexHash_", indexHash)
        self.register_buffer("bitHash_", bitHash)
        return self

    def forward(self, X: Tensor) -> Tensor:
        """
        X: [F] or [B, F]
        returns: [n_components] or [B, n_components]
        """
        self._ensure_fitted(self.indexHash_)

        if X.dim() == 1:
            y = self._forward_batch(X.unsqueeze(0)).squeeze(0)
        elif X.dim() == 2:
            y = self._forward_batch(X)
        else:
            raise ValueError("X must be 1D or 2D tensor.")
        return y

    def transform(self, X: Tensor) -> Tensor:
        return self(X)

    def fit_transform(self, X: Tensor) -> Tensor:
        self.fit(X)
        return self.transform(X)

    def _forward_batch(self, X: Tensor) -> Tensor:
        """
        Vectorized CountSketch + TensorSketch via rfft/irfft.
        X: [B, F]
        returns: [B, n_components]
        """
        B, F = X.shape
        dtype = X.dtype
        D, n = self.degree, self.n_components

        # scale by sqrt(gamma)
        X_gamma = X * (self.gamma ** 0.5)

        # optionally append bias as a constant feature
        if self.coef0 != 0:
            bias = X_gamma.new_full((B, 1), (self.coef0 ** 0.5))
            X_ext = torch.cat([X_gamma, bias], dim=1)  # [B, F']
        else:
            X_ext = X_gamma  # [B, F']
        Fext = X_ext.shape[1]

        idx = self.indexHash_[:, :Fext]                         # [D, F']
        sgn = self.bitHash_[:, :Fext].to(dtype = dtype)           # [D, F'] (cast int8 -> float/half)

        # Build D sketches in a vectorized way
        # per degree: Z_d[b, j] = sgn[d, j] * X_ext[b, j]
        # sketch_d[b, k] = sum_j Z_d[b, j] for which idx[d, j] == k
        sketches = []
        for d in range(D):
            vals_d = X_ext * sgn[d].unsqueeze(0)                # [B, F']
            sketch_d = X_ext.new_zeros((B, n))                  # [B, n]
            sketch_d.scatter_add_(1, idx[d].unsqueeze(0).expand(B, Fext), vals_d)
            sketches.append(sketch_d)
        sketches = torch.stack(sketches, dim=0)                 # [D, B, n]

        # TensorSketch via FFT product
        # Use rfft/irfft for speed and memory (real inputs).
        F_r = torch.fft.rfft(sketches, dim = -1)                  # [D, B, n_r]
        prod = torch.prod(F_r, dim = 0)                           # [B, n_r]
        y = torch.fft.irfft(prod, n = n, dim = -1)                  # [B, n]
        return y

    # ----------------- Analytical Jacobian ----------------
    @torch.no_grad()
    def grad(self, X: Tensor) -> Tensor:
        """
        Analytical Jacobian of the sketch wrt X.
        If X is [n_features], returns [n_components, n_features].
        If X is [B, n_features], returns [B, n_components, n_features].
        """
        if X.dim() == 1:
            return self._grad_single(X)
        else:
            # Prefer vmap if available; otherwise fall back to a loop
            if hasattr(torch, "vmap"):
                return torch.vmap(self._grad_single)(X)
            else:
                outs = [self._grad_single(x) for x in X]
                return torch.stack(outs, dim=0)

    def _grad_single(self, x: Tensor) -> Tensor:
        self._ensure_fitted(self.indexHash_)
        device, dtype = x.device, x.dtype
        n, D = self.n_components, self.degree

        # x with sqrt(gamma); append bias if needed (no grad wrt bias itself)
        x_gamma = x * (self.gamma ** 0.5)
        if self.coef0 != 0:
            x_ext = torch.cat([x_gamma, x.new_tensor([(self.coef0 ** 0.5)])], dim=0)
        else:
            x_ext = x_gamma
        Fext = x_ext.shape[0]

        # Vectorized CountSketch per degree
        sketches = []
        for d in range(D):
            idxs = self.indexHash_[d, :Fext]                    # [Fext]
            bits = self.bitHash_[d, :Fext].to(dtype=dtype)      # [Fext]
            vals = bits * x_ext                                 # [Fext]
            sd = x.new_zeros(n)
            sd.index_add_(0, idxs, vals)                        # [n]
            sketches.append(sd)
        S = torch.stack(sketches, dim=0)                        # [D, n]

        # FFT product excluding each degree using exclusive prefix/suffix via cumprod
        Fr = torch.fft.rfft(S, dim=1)                           # [D, n_r]
        n_r = Fr.shape[1]

        # exclusive prefix: P[d] = ∏_{k<d} F[k]
        P = torch.empty_like(Fr)
        P[0] = torch.ones(n_r, dtype=Fr.dtype, device=device)
        if D > 1:
            P[1:] = torch.cumprod(Fr[:-1], dim=0)

        # exclusive suffix: U[d] = ∏_{k>d} F[k]
        U = torch.empty_like(Fr)
        U[-1] = torch.ones(n_r, dtype=Fr.dtype, device=device)
        if D > 1:
            # cumprod over reversed F[1:], then flip back and align
            tmp = torch.cumprod(torch.flip(Fr[1:], dims=[0]), dim=0)
            U[:-1] = torch.flip(tmp, dims=[0])

        other_prod = P * U                                       # [D, n_r]
        q = torch.fft.irfft(other_prod, n=n, dim=1)              # [D, n], real

        # Build Jacobian columns (still O(D*F*n), exact)
        Forig = self.n_features_
        J_cols = []
        sqrt_gamma = self.gamma ** 0.5
        base = torch.arange(n, device=device)

        for j in range(Forig):
            # sum_d bit[d,j] * roll(q[d], idx[d,j])
            idxs_d = self.indexHash_[:D, j]                     # [D]
            bits_d = self.bitHash_[:D, j].to(dtype=dtype)       # [D]
            col = x.new_zeros(n)
            # small D loop is cheap
            for d in range(D):
                shift = int(idxs_d[d])
                # roll via modular index: q[d][(base - shift) % n]
                col.add_(bits_d[d] * q[d].take((base - shift) % n))
            J_cols.append(sqrt_gamma * col)

        J = torch.stack(J_cols, dim=1)                           # [n, F]
        return J

    # ----------------- Efficient VJP ----------------------
    @torch.no_grad()
    def vjp(self, X: Tensor, v: Tensor) -> Tensor:
        """
        Vector–Jacobian product: returns J(X)^T @ v.

        X: [F] or [B, F]
        v: [n_components]  (ALWAYS 1-D)
        Output: [F] or [B, F]
        """
        self._ensure_fitted(self.indexHash_)
        if v.dim() != 1 or v.numel() != self.n_components:
            raise ValueError(
                f"v must be 1-D of length n_components={self.n_components}, "
                f"got shape {tuple(v.shape)}"
            )
        if X.dim() == 1:
            return self._vjp_single(X, v)
        elif X.dim() == 2:
            return self._vjp_batch(X, v)
        else:
            raise ValueError("X must be 1D or 2D tensor.")


    def _vjp_single(self, x: Tensor, v: Tensor) -> Tensor:
        """J(x)^T @ v for a single input x: [F]."""
        device, dtype = x.device, x.dtype
        n, D = self.n_components, self.degree

        # 1) extended x (bias has no grad wrt x)
        x_gamma = x * (self.gamma ** 0.5)
        if self.coef0 != 0:
            x_ext = torch.cat([x_gamma, x.new_tensor([(self.coef0 ** 0.5)])], dim=0)
        else:
            x_ext = x_gamma
        Fext = x_ext.shape[0]

        # 2) CountSketch per degree (vectorized)
        S_rows = []
        for d in range(D):
            idxs = self.indexHash_[d, :Fext]                   # [Fext]
            bits = self.bitHash_[d, :Fext].to(dtype=dtype)     # [Fext]
            vals = bits * x_ext                                # [Fext]
            sd = x.new_zeros(n)
            sd.index_add_(0, idxs, vals)                       # [n]
            S_rows.append(sd)
        S = torch.stack(S_rows, dim=0)                         # [D, n]

        # 3) rFFT and exclusive product to get q_d
        Fr = torch.fft.rfft(S, dim=1)                          # [D, n_r]
        n_r = Fr.shape[1]
        P = torch.empty_like(Fr)
        U = torch.empty_like(Fr)
        P[0] = torch.ones(n_r, dtype=Fr.dtype, device=device)
        U[-1] = torch.ones(n_r, dtype=Fr.dtype, device=device)
        if D > 1:
            P[1:] = torch.cumprod(Fr[:-1], dim=0)
            tmp = torch.cumprod(torch.flip(Fr[1:], dims=[0]), dim=0)
            U[:-1] = torch.flip(tmp, dims=[0])
        other_prod = P * U
        q = torch.fft.irfft(other_prod, n=n, dim=1).to(dtype)  # [D, n]

        # 4) Correct circular cross-correlation:
        #    c_d = ifft( conj(FFT(q_d)) * FFT(v) ), so c_d[shift] = sum_m q_d[m] * v[m+shift]
        Vr = torch.fft.rfft(v.to(dtype), dim=0)                # [n_r]
        Q = torch.fft.rfft(q, dim=1)                           # [D, n_r]
        C = torch.fft.irfft(torch.conj(Q) * Vr.unsqueeze(0), n=n, dim=1).real  # [D, n]

        # 5) Assemble J^T v (ignore bias column)
        Forig = self.n_features_
        idx = self.indexHash_[:D, :Forig]                      # [D, Forig]
        bits = self.bitHash_[:D, :Forig].to(dtype=dtype)       # [D, Forig]
        # gather C[d, idx[d,j]] for all d,j at once
        gathered = C.gather(1, idx)                            # [D, Forig]
        out = (gathered * bits).sum(dim=0)                     # [Forig]
        return (self.gamma ** 0.5) * out


    def _vjp_batch(self, X: Tensor, v: Tensor) -> Tensor:
        """J(X)^T @ v for a batch X: [B, F], v: [n]."""
        B, F = X.shape
        device, dtype = X.device, X.dtype
        D, n = self.degree, self.n_components

        # 1) extend and scale X
        X_gamma = X * (self.gamma ** 0.5)
        if self.coef0 != 0:
            bias = X_gamma.new_full((B, 1), (self.coef0 ** 0.5))
            X_ext = torch.cat([X_gamma, bias], dim=1)          # [B, Fext]
        else:
            X_ext = X_gamma                                    # [B, Fext]
        Fext = X_ext.shape[1]

        idx = self.indexHash_[:, :Fext]                        # [D, Fext]
        bits = self.bitHash_[:, :Fext].to(dtype=dtype)         # [D, Fext]

        # 2) D sketches, batched (scatter-add)
        sketches = []
        for d in range(D):
            vals_d = X_ext * bits[d].unsqueeze(0)              # [B, Fext]
            sk = X_ext.new_zeros((B, n))                       # [B, n]
            sk.scatter_add_(1, idx[d].unsqueeze(0).expand(B, Fext), vals_d)
            sketches.append(sk)
        S = torch.stack(sketches, dim=1)                       # [B, D, n]

        # 3) rFFT per batch, exclusive products over degree
        Fr = torch.fft.rfft(S, dim=2)                          # [B, D, n_r]
        n_r = Fr.shape[2]
        P = torch.empty_like(Fr)
        U = torch.empty_like(Fr)
        P[:, 0] = torch.ones(n_r, dtype=Fr.dtype, device=device)
        U[:, -1] = torch.ones(n_r, dtype=Fr.dtype, device=device)
        if D > 1:
            P[:, 1:] = torch.cumprod(Fr[:, :-1], dim=1)
            tmp = torch.cumprod(torch.flip(Fr[:, 1:], dims=[1]), dim=1)
            U[:, :-1] = torch.flip(tmp, dims=[1])
        other = P * U                                          # [B, D, n_r]
        q = torch.fft.irfft(other, n=n, dim=2).to(dtype)       # [B, D, n]

        # 4) Correct circular cross-correlation against v (shared across batch)
        Vr = torch.fft.rfft(v.to(dtype), dim=0)                # [n_r]
        Q = torch.fft.rfft(q, dim=2)                           # [B, D, n_r]
        C = torch.fft.irfft(torch.conj(Q) * Vr.view(1, 1, -1), n=n, dim=2).real  # [B, D, n]

        # 5) Gather at hashed indices and sum over degrees; drop bias column
        Forig = self.n_features_
        idxF = idx[:, :Forig]                                  # [D, Forig]
        bitsF = bits[:, :Forig]                                # [D, Forig]
        # expand idx over batch and gather
        gathered = C.gather(2, idxF.unsqueeze(0).expand(B, -1, -1))  # [B, D, Forig]
        out = (gathered * bitsF.unsqueeze(0)).sum(dim=1)            # [B, Forig]
        return out



class NormedPolyCntSketch(PolyCntSketch):
    def __init__(
        self,
        degree: int = 2,
        n_components: int = 100,
        gamma: float = 1.0,
        coef0: float = 0.0,
        eps: float = 1e-12,   # numerical stability for ||x||
    ):
        super().__init__(degree=degree, n_components=n_components, gamma=gamma, coef0=coef0)
        self.eps = float(eps)

    # --------- helpers ---------
    def _normalize(self, X: Tensor):
        """
        Returns (x_hat, r), where x_hat = X / (||X|| + eps).
        r is scalar for 1D input, and shape [B, 1] for batched input.
        """
        r = X.norm(p = 2, dim = -1, keepdim=True) + self.eps  # [B,1]
        return X / r, r

    # --------- forward ---------
    def forward(self, X: Tensor) -> Tensor:
        """
        X: [F] or [B, F]
        returns: [n_components] or [B, n_components]
        """
        x_hat, _ = self._normalize(X)
        return super().forward(x_hat)

    def transform(self, X: Tensor) -> Tensor:
        return self(X)

    # --------- analytical Jacobian ---------
    @torch.no_grad()
    def grad(self, X: Tensor) -> Tensor:
        """
        If X is [F], returns [n_components, F].
        If X is [B, F], returns [B, n_components, F].
        """
        x_hat, r = self._normalize(X)

        # J_f := d(PolyCntSketch)/d(x_hat) evaluated at x_hat
        J_f = super().grad(x_hat)  # [n,F] or [B,n,F]

        # Chain rule: J = J_f @ ((I - x̂ x̂^T)/||x||)
        # Efficient form: J = (J_f - (J_f @ x̂) x̂^T) / ||x||
        if X.dim() == 1:
            # x_hat: [F], J_f: [n,F]
            proj = J_f @ x_hat                 # [n]
            J = (J_f - proj.unsqueeze(1) * x_hat.unsqueeze(0)) / r
            return J
        else:
            # X: [B,F], x_hat: [B,F], r: [B,1], J_f: [B,n,F]
            proj = torch.einsum('bnf,bf->bn', J_f, x_hat)       # [B,n]
            J = (J_f - proj.unsqueeze(-1) * x_hat.unsqueeze(1)) / r.unsqueeze(1)
            return J

    # --------- efficient VJP (v is always 1-D) ---------
    @torch.no_grad()
    def vjp(self, X: Tensor, v: Tensor) -> Tensor:
        """
        Vector–Jacobian product: returns J(X)^T @ v.

        X: [F] or [B, F]
        v: [n_components]   (always 1-D)
        Output: [F] or [B, F]
        """
        if v.dim() != 1 or v.numel() != self.n_components:
            raise ValueError(
                f"v must be 1-D with length n_components={self.n_components}, "
                f"got {tuple(v.shape)}"
            )

        x_hat, r = self._normalize(X)

        # First get u = (d y / d x̂)^T v using the base class (already fast & batched)
        u = super().vjp(x_hat, v)   # [F] or [B,F]

        # Then left-multiply by d(x̂)/dx = (I - x̂ x̂^T)/||x||
        # Using (I - x̂ x̂^T) u = u - x̂ (x̂^T u)
        if X.dim() == 1:
            dot = (x_hat * u).sum()         # scalar
            return (u - x_hat * dot) / r
        else:
            dot = (x_hat * u).sum(dim=1, keepdim=True)  # [B,1]
            return (u - x_hat * dot) / r                # broadcast over [B,1]