# PyTorch issue drafts (ledger item 8) — formatted for the torch.compile bug template

Two independent inductor bugs, each with a minimal reproducer, laid out to paste
straight into the pytorch/pytorch "torch.compile bug report" form (sections below match
its fields). Before filing, three template requirements are YOURS to complete:

1. **Versions**: run this ON THE CLUSTER (inside the NGC container — the environment the
   bugs were observed in) and paste the output into each issue's Versions field:

       curl -sL https://raw.githubusercontent.com/pytorch/pytorch/main/torch/utils/collect_env.py | python3

2. **AI policy**: PyTorch's AI_POLICY requires disclosure of AI-assisted content. These
   drafts are AI-assisted; review them yourself, confirm the reproducers run as described
   in your environment, and include a disclosure line (suggested: "This report was
   drafted with AI assistance and reviewed/reproduced by me.").

3. **Search first** (template requirement): check existing issues for
   `aten.nonzero.default inductor backend` and for
   `user_defined_triton_kernel_transitive_closure NameError` before posting; link any
   near-duplicates instead of filing anew if they match.

Neither issue is fuzzer-generated — the fuzzer checklist and label do not apply.

---

## Issue 1 — inductor backend exception on `aten.nonzero.default`

**Title**: Inductor backend exception (not a graph break planned by dynamo) on
aten.nonzero.default from boolean-mask assignment under torch.compile(dynamic=True)

### 🐛 Describe the bug

Compiling a module that applies BatchNorm to a boolean-masked subset and writes it back
(`out[mask] = norm(h[mask])`) fails in the inductor **backend** with
`Exception: aten.nonzero.default`, then falls back via a graph break. Dynamo itself
handles the data-dependent `nonzero` (unbacked symints); it is the backend that throws,
and the error text itself says "Hint: Report an issue to the backend compiler repo".
Expected: either a lowering for the masked pattern or a graph break planned by dynamo —
not a backend exception on a documented-supported op.

**Ablation**: `backend="inductor"` (default mode, `dynamic=True`) throws;
eager is fine. [Before filing, run the repro once with `backend="aot_eager"` and state
the result — expected clean, which isolates the failure to inductor lowering.]

**Minimal reproducer**:

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
f(h, mask).sum().backward()
```

Observed in a real training run (particle-physics GNN, per-step hot path) and reduced to
the above. Severity: functional fallback — training proceeds through the break — so a
performance bug, but the break lands inside the per-step region and the exception path
is undocumented. [Optional but recommended by the template: rerun the repro with
`TORCH_TRACE=/tmp/torch_trace` set, run `tlparse /tmp/torch_trace/*` and attach the full
artifact zip — not just index.html.]

### Error logs

```
W0818 21:05:28 torch/_dynamo/exc.py:518 [4/0] Backend compiler exception
  Explanation: Backend compiler `inductor` failed with aten.nonzero.default. Adding a graph break.
  Hint: Report an issue to the backend compiler repo.
  Developer debug context: Backend: inductor
    Exception:aten.nonzero.default
    Traceback:
      File ".../plaingraphgps.py", line 192, in torch_dynamo_resume_in_forward_at_191
        return out
/usr/local/lib/python3.12/dist-packages/torch/autograd/graph.py:829: UserWarning:
  Error detected in IndexPutBackward0. Traceback of forward call that caused the error:
  File ".../plaingraphgps.py", line 191, in torch_dynamo_resume_in_forward_at_191
    out[mask_bool] = self.norm(h[mask_bool])   # BatchNorm over the real nodes only
```

### Versions

[paste collect_env output from the cluster here]

---

## Issue 2 — user-defined Triton closure emitter drops the alias for a renamed JITFunction

**Title**: NameError in ast_to_ttir when a triton.jit function is called through a
module-level binding whose name differs from its def name (inductor
user_defined_triton_kernel_transitive_closure_source_code)

### 🐛 Describe the bug

`torch._inductor.utils.user_defined_triton_kernel_transitive_closure_source_code`
re-emits each called JITFunction's `src` — whose `def` line carries the ORIGINAL
function name — but, unlike its ConstexprFunction branch, writes no alias when the
calling kernel referenced it through a differently-named module binding. The re-emitted
closure then calls a name that does not exist in the emitted source and dies at
`ast_to_ttir` with `NameError`. The raw eager launch works (name resolution goes through
module `__globals__`); only the torch.compile path fails.

**Ablation**: eager launch of the same op works; the failure appears exactly when the
op is reached through `torch.compile` (inductor's user-defined-kernel embedding);
binding the JITFunction under the wrapped function's own def name makes compile succeed,
which isolates the fault to the missing alias in the closure emitter.

**Minimal reproducer**:

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

Workaround (shipped in our project): bind the JITFunction under the wrapped function's
own def name (`_fwd_body = triton.jit(_fwd_body)`). Suggested fix: emit the same alias
line the ConstexprFunction branch already emits.

### Error logs

```
torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
NameError('_fwd_body is not defined')
  (raised from ast_to_ttir while compiling the re-emitted user-defined kernel closure)
```

[Paste the full traceback from your run of the reproducer when filing.]

### Versions

[paste collect_env output from the cluster here]
