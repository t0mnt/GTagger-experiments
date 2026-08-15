"""fp32 precision islands inside autocast regions, for the Clifford primitives.

VENDORED from lgatr 2.0.0 (`lgatr/utils/autocast.py`, MIT). Copied rather than imported so
that this package stays transplantable back to DavidRuhe/clifford-group-equivariant-neural-
networks, which does not depend on lgatr -- the same reason every other file here carries a
`# from https://github.com/DavidRuhe/...` header. Behavioural parity with the installed
lgatr is pinned by tests/internal/test_cgenn_autocast.py, so the copy cannot drift silently.

WHY CGENN needs it. bf16 carries 8 mantissa bits against fp32's 24. That is fine for one
GEMM and wrong for the geometric product, which is a product OF products -- `cayley . W .
(x (x) y)` contracted over 256 paths per channel pair -- feeding `b()`/`q()`, which are
differences of like-magnitude terms and cancel. Running those in bf16 loses the
cancellation, not just a few digits. The decorator pins a FLOOR rather than disabling
autocast: the bulk of the network (the scalar MLPs phi_h/theta_h/psi_x/chi_x, the attention
GEMMs in the hybrids) still runs bf16 and still collects the bandwidth win, while the
Clifford ops run fp32.

NO-OP WHEN AMP IS OFF, which every shipped config is (`use_amp: false`): with no autocast
region active the decorator returns `func(*args, **kwargs)` unchanged, so decorating a
function is bit-identical until someone enables AMP. That is what makes landing this ahead
of the AMP fixtures safe.
"""

from collections.abc import Callable
from functools import wraps
from itertools import chain
from typing import Any, Literal

import torch

# Toggled by the naive_amp context manager; read at call time so torch.compile constant-folds it.
_NAIVE_AMP = False


def _autocast_active() -> bool:
    """Whether CPU or CUDA autocast is enabled."""
    return torch.is_autocast_enabled("cuda") or torch.is_autocast_enabled("cpu")


def autocast_dtype(device_type: str = "cuda") -> torch.dtype:
    """Dtype that autocast would cast to on ``device_type``."""
    return torch.get_autocast_dtype(device_type)


class naive_amp:
    """Disable all :class:`minimum_autocast_precision` pinning inside the block.

    While active, the fp32 islands are bypassed and the wrapped ops run in the surrounding
    autocast dtype (e.g. bf16). This is the control arm for measuring what the guards buy
    and what they cost: `with naive_amp(): ...` is unguarded bf16. Restores the previous
    state on exit; safe to nest. ``naive_amp(False)`` is a no-op that never overrides an
    outer ``naive_amp``.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._prev: list[bool] = []

    def __enter__(self) -> "naive_amp":
        global _NAIVE_AMP
        if self.enabled:
            self._prev.append(_NAIVE_AMP)
            _NAIVE_AMP = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        global _NAIVE_AMP
        if self.enabled:
            _NAIVE_AMP = self._prev.pop()
        return False


class minimum_autocast_precision:
    """Pin tensors to a minimum precision inside autocast regions.

    Used as a decorator: ``@minimum_autocast_precision(torch.float32)``. Inside
    autocast-enabled regions, floating-point inputs below ``min_dtype`` are cast up to
    ``min_dtype``, autocast is disabled for the call, and outputs are cast per ``output``.
    Outside autocast regions it is a no-op. Non-tensors, integer tensors and boolean tensors
    are left alone.

    Parameters
    ----------
    min_dtype
        Minimum dtype.
    output
        Which dtype the (floating-point tensor) outputs are cast to. ``"low"`` (default) uses
        the lowest precision among ``min_dtype`` and the input dtypes -- i.e. hand the result
        back at the ambient autocast precision, so the island does not widen the tensors
        downstream. ``"high"`` uses the highest-precision input dtype, for ops whose result
        feeds another island. ``None`` leaves outputs untouched. A ``torch.dtype`` forces one.
    """

    def __init__(
        self,
        min_dtype: torch.dtype = torch.float32,
        output: Literal["low", "high"] | torch.dtype | None = "low",
    ) -> None:
        self.min_dtype = min_dtype
        self.output = output

    def cast(self, var: Any) -> Any:
        """Upcast a floating-point tensor to at least ``min_dtype``."""
        if not isinstance(var, torch.Tensor):
            return var
        if not var.dtype.is_floating_point:
            return var
        if torch.finfo(var.dtype).bits >= torch.finfo(self.min_dtype).bits:
            return var
        return var.to(self.min_dtype)

    def _cast_out(self, var: Any, dtype: torch.dtype) -> Any:
        """Cast a single output to the requested dtype."""
        if not isinstance(var, torch.Tensor):
            return var
        if not var.dtype.is_floating_point:
            return var
        return var.to(dtype)

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def decorated_func(*args: Any, **kwargs: Any):
            # Skip in naive-AMP mode (run in the autocast dtype), or outside autocast regions.
            if _NAIVE_AMP or not _autocast_active():
                return func(*args, **kwargs)
            mod_args = [self.cast(arg) for arg in args]
            mod_kwargs = {key: self.cast(val) for key, val in kwargs.items()}
            # Fresh contexts (not `with self:`) -- keeps the decorator re-entrant-safe.
            with (
                torch.autocast(device_type="cuda", enabled=False),
                torch.autocast(device_type="cpu", enabled=False),
            ):
                outputs = func(*mod_args, **mod_kwargs)
            return self._apply_output_dtype(outputs, args, kwargs)

        return decorated_func

    def _apply_output_dtype(self, outputs: Any, args: tuple, kwargs: dict) -> Any:
        """Cast outputs per the ``output`` mode; see class docstring."""
        if self.output is None:
            return outputs
        if self.output in ["low", "high"]:
            in_dtypes = [
                arg.dtype
                for arg in chain(args, kwargs.values())
                if isinstance(arg, torch.Tensor) and arg.dtype.is_floating_point
            ]
            if not in_dtypes:
                # No floating-point inputs to derive "low"/"high" from; nothing to cast back to.
                return outputs
            # Plain loop instead of min/max(..., key=lambda) to avoid graph breaks in torch.compile
            if self.output == "low":
                candidates = [self.min_dtype] + in_dtypes
                out_dtype = candidates[0]
                for dt in candidates[1:]:
                    if torch.finfo(dt).bits < torch.finfo(out_dtype).bits:
                        out_dtype = dt
            else:
                out_dtype = in_dtypes[0]
                for dt in in_dtypes[1:]:
                    if torch.finfo(dt).bits > torch.finfo(out_dtype).bits:
                        out_dtype = dt
        else:
            out_dtype = self.output
        if isinstance(outputs, tuple):
            return tuple(self._cast_out(val, out_dtype) for val in outputs)
        return self._cast_out(outputs, out_dtype)
