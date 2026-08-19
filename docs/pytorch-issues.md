# PyTorch issue drafts (ledger item 8) — ready to file

Two independent inductor bugs hit during the CGENN compile program, each with a
minimal reproducer. File against pytorch/pytorch, label `module: inductor`.
Environment for both: NGC PyTorch container (torch 2.13 nightly-line build,
CUDA 13.0, H100 NVL), also reproduced expectations documented below.

## Issue 1: backend exception on `aten.nonzero.default` inside a compiled region

**Title**: Inductor backend exception (not a lowering) on aten.nonzero.default from
boolean-mask assignment under torch.compile(dynamic=True)

**Body**:

Compiling a module that applies BatchNorm to a boolean-masked subset and writes it
back (`out[mask] = norm(h[mask])`) fails in the inductor BACKEND with
`Exception: aten.nonzero.default` and falls back via a graph break. Dynamo itself
handles the data-dependent `nonzero` (unbacked symints); it is the backend that
throws. Expected: either a lowering for the masked pattern or a clean
graph break planned by dynamo — not a backend compiler exception, which the error
text itself flags as reportable ("Hint: Report an issue to the backend compiler
repo").

```python
import torch

class MaskedBN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.BatchNorm1d(8)

    def forward(self, h, mask_bool):
        out = h.clone()
        out[mask_bool] = self.norm(h[mask_bool])
        return out

m = MaskedBN().cuda()
f = torch.compile(m, dynamic=True)
h = torch.randn(64, 8, device="cuda", requires_grad=True)
mask = torch.rand(64, device="cuda") > 0.3
f(h, mask).sum().backward()   # W ... Backend compiler exception
                              # Explanation: Backend compiler `inductor` failed with
                              # aten.nonzero.default. Adding a graph break.
```

Observed log (from a real training run):

```
W0818 21:05:28 torch/_dynamo/exc.py:518 [4/0] Backend compiler exception
  Explanation: Backend compiler `inductor` failed with aten.nonzero.default. Adding a graph break.
  Hint: Report an issue to the backend compiler repo.
  Developer debug context: Backend: inductor / Exception:aten.nonzero.default
```

Severity: functional fallback (training proceeds through the break), so this is a
performance bug, not a correctness one — but the break lands inside a hot per-step
region and the exception path is undocumented.

## Issue 2: user-defined Triton kernel closure emitter drops the alias for a
JITFunction bound under a different name

**Title**: NameError in ast_to_ttir when a triton.jit function is called through a
module-level binding whose name differs from its def name (inductor
user-defined-kernel closure embedding)

**Body**:

`torch._inductor.utils.user_defined_triton_kernel_transitive_closure_source_code`
re-emits each called JITFunction's `src` — whose `def` line carries the ORIGINAL
function name — but, unlike its ConstexprFunction branch, writes no alias when the
kernel referenced it through a differently-named module binding. The re-emitted
closure then calls a name that does not exist and dies at `ast_to_ttir` with
`NameError`. The raw eager launch works (name resolution goes through module
`__globals__`); only the torch.compile path fails.

```python
import torch, triton, triton.language as tl
from torch.library import triton_op, wrap_triton

def _fwd_body(x_ptr, o_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(o_ptr + offs, tl.load(x_ptr + offs) * 2.0, mask=offs < N)

helper = triton.jit(_fwd_body)          # bound under a DIFFERENT name than the def

@triton.jit
def outer(x_ptr, o_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    helper(x_ptr, o_ptr, N, BLOCK)      # calls the binding name

@triton_op("repro::double", mutates_args=())
def double(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    wrap_triton(outer)[(triton.cdiv(x.numel(), 128),)](x, out, x.numel(), BLOCK=128)
    return out

x = torch.randn(1024, device="cuda")
double(x)                                # eager: works
torch.compile(lambda t: double(t))(x)    # NameError: '_fwd_body' is not defined
```

Workaround (what our repo ships): bind the JITFunction under the wrapped
function's own def name (`_fwd_body = triton.jit(_fwd_body)`). Suggested fix:
emit the same alias line the ConstexprFunction branch already emits.
