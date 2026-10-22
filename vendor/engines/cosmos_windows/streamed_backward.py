# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/streamed_backward.py
# Machine fallback paths cleared; no model data included.
from __future__ import annotations
import torch

class StreamedLayer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, layer, bind, evict, x, *params):
        ctx.layer = layer
        ctx.bind = bind
        ctx.evict = evict
        bind()
        with torch.no_grad():
            out = layer(x)
        evict()
        ctx.save_for_backward(x)
        ctx.n_params = len(params)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x = ctx.saved_tensors[0].detach().requires_grad_(True)
        params = [p for p in _params_of(ctx.layer)]
        ctx.bind()
        with torch.enable_grad():
            out = ctx.layer(x)
        grads = torch.autograd.grad(out, [x] + params, grad_out, allow_unused=True)
        ctx.evict()
        return (None, None, None, grads[0]) + tuple(grads[1:])

class StreamedLayerMulti(torch.autograd.Function):

    @staticmethod
    def forward(ctx, layer, bind, evict, call_fn, n_inputs, *tensors):
        ctx.layer = layer
        ctx.bind = bind
        ctx.evict = evict
        ctx.call_fn = call_fn
        ctx.n_inputs = n_inputs
        inputs = tensors[:n_inputs]
        bind()
        with torch.no_grad():
            out = call_fn(layer, *inputs)
        evict()
        ctx.save_for_backward(*inputs)
        return out

    @staticmethod
    def backward(ctx, *grad_outs):
        saved = [t.detach().requires_grad_(t.is_floating_point()) for t in ctx.saved_tensors]
        params = _params_of(ctx.layer)
        ctx.bind()
        with torch.enable_grad():
            out = ctx.call_fn(ctx.layer, *saved)
        outs = out if isinstance(out, tuple) else (out,)
        wrt = [s for s in saved if s.requires_grad] + params
        grads = torch.autograd.grad(outs, wrt, grad_outs, allow_unused=True)
        ctx.evict()
        gi = iter(grads)
        input_grads = [next(gi) if s.requires_grad else None for s in saved]
        param_grads = list(gi)
        return (None, None, None, None, None, *input_grads, *param_grads)

class StreamedLayerBatch(torch.autograd.Function):

    @staticmethod
    def forward(ctx, layer, bind, evict, call_fns, n_img, n_in, offload, *tensors):
        ctx.layer = layer
        ctx.bind = bind
        ctx.evict = evict
        ctx.call_fns = call_fns
        ctx.n_img = n_img
        ctx.n_in = n_in
        ctx.offload = offload
        grad_inputs = tensors[:n_img * n_in]
        bind()
        flat_out, n_out = ([], None)
        with torch.no_grad():
            for i in range(n_img):
                o = call_fns[i](layer, *grad_inputs[i * n_in:(i + 1) * n_in])
                o = o if isinstance(o, tuple) else (o,)
                n_out = len(o)
                flat_out.extend(o)
        evict()
        ctx.n_out = n_out
        if offload:
            ctx.dev = grad_inputs[0].device
            ctx.cpu_saved = [t.detach().to('cpu', copy=True).pin_memory() if t.is_floating_point() else t.detach().to('cpu', copy=True) for t in grad_inputs]
            ctx.req_grad = [t.requires_grad for t in grad_inputs]
        else:
            ctx.save_for_backward(*grad_inputs)
        return tuple(flat_out)

    @staticmethod
    def backward(ctx, *grad_outs):
        if ctx.offload:
            saved = [c.to(ctx.dev, non_blocking=True).requires_grad_(c.is_floating_point()) for c in ctx.cpu_saved]
        else:
            saved = [t.detach().requires_grad_(t.is_floating_point()) for t in ctx.saved_tensors]
        params = _params_of(ctx.layer)
        ctx.bind()
        outs = []
        with torch.enable_grad():
            for i in range(ctx.n_img):
                o = ctx.call_fns[i](ctx.layer, *saved[i * ctx.n_in:(i + 1) * ctx.n_in])
                outs.extend(o if isinstance(o, tuple) else (o,))
        wrt_inputs = [s for s in saved if s.requires_grad]
        grads = torch.autograd.grad(outs, wrt_inputs + params, grad_outs, allow_unused=True)
        ctx.evict()
        gin = iter(grads[:len(wrt_inputs)])
        input_grads = [next(gin) if s.requires_grad else None for s in saved]
        param_grads = list(grads[len(wrt_inputs):])
        return (None, None, None, None, None, None, None, *input_grads, *param_grads)

def apply_streamed_batch(layer, call_fns, image_inputs, bind=None, evict=None, offload=False):
    params = _params_of(layer)
    b = bind if bind is not None else layer.bind
    e = evict if evict is not None else layer.evict
    n_img = len(image_inputs)
    n_in = len(image_inputs[0])
    flat = [t for tup in image_inputs for t in tup]
    out = StreamedLayerBatch.apply(layer, b, e, call_fns, n_img, n_in, offload, *flat, *params)
    n_out = len(out) // n_img
    return [tuple(out[i * n_out:(i + 1) * n_out]) for i in range(n_img)]

def apply_streamed_multi(layer, call_fn, *grad_inputs, bind=None, evict=None):
    params = _params_of(layer)
    b = bind if bind is not None else layer.bind
    e = evict if evict is not None else layer.evict
    return StreamedLayerMulti.apply(layer, b, e, call_fn, len(grad_inputs), *grad_inputs, *params)

def _params_of(layer):
    return [p for p in layer.parameters() if p.requires_grad]

def apply_streamed(layer, x):
    params = _params_of(layer)
    return StreamedLayer.apply(layer, layer.bind, layer.evict, x, *params)

def _selftest():
    import torch.nn as nn
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    g = torch.Generator().manual_seed(2)

    class LoraLinear(nn.Module):

        def __init__(s, Wcpu, r=2):
            super().__init__()
            s.Wcpu = Wcpu
            D = Wcpu.shape[0]
            s.A = nn.Parameter(torch.randn(r, D, generator=g))
            s.B = nn.Parameter(torch.randn(D, r, generator=g))
            s.W = None

        def bind(s):
            s.W = s.Wcpu.to(dev)

        def evict(s):
            s.W = None

        def forward(s, x):
            return x @ s.W.T + x @ s.A.T @ s.B.T
    Wcpu = torch.randn(6, 6, generator=g)
    x = torch.randn(3, 6, generator=g).to(dev)
    m = LoraLinear(Wcpu).to(dev)
    (apply_streamed(m, x) ** 2).mean().backward()
    mr = LoraLinear(Wcpu).to(dev)
    mr.A.data.copy_(m.A.data)
    mr.B.data.copy_(m.B.data)
    mr.bind()
    (mr(x) ** 2).mean().backward()
    ok = torch.allclose(m.A.grad, mr.A.grad, atol=0.001) and torch.allclose(m.B.grad, mr.B.grad, atol=0.001)
    print(f"StreamedLayer(nn.Module): grads match resident = {ok} -> {('PASS' if ok else 'FAIL')}")
    return ok

def _selftest_multi():
    import torch.nn as nn
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    g = torch.Generator().manual_seed(5)

    class TwoStreamBlock(nn.Module):

        def __init__(s, Wcpu, r=2):
            super().__init__()
            s.Wcpu = Wcpu
            D = Wcpu.shape[0]
            s.A = nn.Parameter(torch.randn(r, D, generator=g))
            s.B = nn.Parameter(torch.randn(D, r, generator=g))
            s.W = None

        def bind(s):
            s.W = s.Wcpu.to(dev)

        def evict(s):
            s.W = None

        def forward(s, h, e, temb):
            lora = h @ s.A.T @ s.B.T
            return h @ s.W.T + e + temb + lora
    Wcpu = torch.randn(6, 6, generator=g)
    h = torch.randn(3, 6, generator=g).to(dev)
    e = torch.randn(3, 6, generator=g).to(dev)
    temb = torch.randn(3, 6, generator=g).to(dev)
    m = TwoStreamBlock(Wcpu).to(dev)
    call = lambda layer, hh, ee: layer(hh, ee, temb)
    h.requires_grad_(True)
    e.requires_grad_(True)
    (apply_streamed_multi(m, call, h, e) ** 2).mean().backward()
    mr = TwoStreamBlock(Wcpu).to(dev)
    mr.A.data.copy_(m.A.data)
    mr.B.data.copy_(m.B.data)
    mr.bind()
    hr = h.detach().clone().requires_grad_(True)
    er = e.detach().clone().requires_grad_(True)
    (mr(hr, er, temb) ** 2).mean().backward()
    ok = torch.allclose(m.A.grad, mr.A.grad, atol=0.001) and torch.allclose(m.B.grad, mr.B.grad, atol=0.001) and torch.allclose(h.grad, hr.grad, atol=0.001) and torch.allclose(e.grad, er.grad, atol=0.001)
    print(f"StreamedLayerMulti (2 inputs + param + const): grads match resident = {ok} -> {('PASS' if ok else 'FAIL')}")
    return ok

def _selftest_tuple():
    import torch.nn as nn
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    g = torch.Generator().manual_seed(7)

    class MoTBlock(nn.Module):

        def __init__(s, Wcpu, r=2):
            super().__init__()
            s.Wcpu = Wcpu
            D = Wcpu.shape[0]
            s.A = nn.Parameter(torch.randn(r, D, generator=g))
            s.B = nn.Parameter(torch.randn(D, r, generator=g))
            s.W = None

        def bind(s):
            s.W = s.Wcpu.to(dev)

        def evict(s):
            s.W = None

        def forward(s, und, gen, rot):
            u = und + und @ s.W.T + und @ s.A.T @ s.B.T + rot
            v = gen + gen @ s.W.T + rot
            return (u, v)
    Wcpu = torch.randn(6, 6, generator=g)
    und = torch.randn(3, 6, generator=g).to(dev).requires_grad_(True)
    gen = torch.randn(3, 6, generator=g).to(dev).requires_grad_(True)
    rot = torch.randn(3, 6, generator=g).to(dev)
    m = MoTBlock(Wcpu).to(dev)
    call = lambda layer, u, v: layer(u, v, rot)
    uo, vo = apply_streamed_multi(m, call, und, gen, bind=m.bind, evict=m.evict)
    ((uo ** 2).mean() + (vo ** 2).mean()).backward()
    mr = MoTBlock(Wcpu).to(dev)
    mr.A.data.copy_(m.A.data)
    mr.B.data.copy_(m.B.data)
    mr.bind()
    ur = und.detach().clone().requires_grad_(True)
    vr = gen.detach().clone().requires_grad_(True)
    uo2, vo2 = mr(ur, vr, rot)
    ((uo2 ** 2).mean() + (vo2 ** 2).mean()).backward()
    ok = torch.allclose(m.A.grad, mr.A.grad, atol=0.001) and torch.allclose(m.B.grad, mr.B.grad, atol=0.001) and torch.allclose(und.grad, ur.grad, atol=0.001) and torch.allclose(gen.grad, vr.grad, atol=0.001)
    print(f"StreamedLayerMulti (2 in -> 2 out, MoT-block shape): grads match resident = {ok} -> {('PASS' if ok else 'FAIL')}")
    return ok

def _selftest_batch():
    import torch.nn as nn
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    g = torch.Generator().manual_seed(11)

    class MoTBlock(nn.Module):

        def __init__(s, Wcpu, r=2):
            super().__init__()
            s.Wcpu = Wcpu
            D = Wcpu.shape[0]
            s.A = nn.Parameter(torch.randn(r, D, generator=g))
            s.B = nn.Parameter(torch.randn(D, r, generator=g))
            s.W = None

        def bind(s):
            s.W = s.Wcpu.to(dev)

        def evict(s):
            s.W = None

        def forward(s, und, gen, rot):
            u = und + und @ s.W.T + und @ s.A.T @ s.B.T + rot
            v = gen + gen @ s.W.T + rot
            return (u, v)
    N = 4
    Wcpu = torch.randn(6, 6, generator=g)
    unds = [torch.randn(3, 6, generator=g).to(dev).requires_grad_(True) for _ in range(N)]
    gens = [torch.randn(3, 6, generator=g).to(dev).requires_grad_(True) for _ in range(N)]
    rots = [torch.randn(3, 6, generator=g).to(dev) for _ in range(N)]
    m = MoTBlock(Wcpu).to(dev)
    call_fns = [lambda layer, u, v, _r=rots[i]: layer(u, v, _r) for i in range(N)]
    outs = apply_streamed_batch(m, call_fns, list(zip(unds, gens)), bind=m.bind, evict=m.evict)
    loss = sum(((uo ** 2).mean() + (vo ** 2).mean() for uo, vo in outs))
    loss.backward()
    mr = MoTBlock(Wcpu).to(dev)
    mr.A.data.copy_(m.A.data)
    mr.B.data.copy_(m.B.data)
    mr.bind()
    ur = [u.detach().clone().requires_grad_(True) for u in unds]
    vr = [v.detach().clone().requires_grad_(True) for v in gens]
    lref = sum(((lambda uo, vo: (uo ** 2).mean() + (vo ** 2).mean())(*mr(ur[i], vr[i], rots[i])) for i in range(N)))
    lref.backward()
    ok = torch.allclose(m.A.grad, mr.A.grad, atol=0.001) and torch.allclose(m.B.grad, mr.B.grad, atol=0.001) and all((torch.allclose(unds[i].grad, ur[i].grad, atol=0.001) for i in range(N))) and all((torch.allclose(gens[i].grad, vr[i].grad, atol=0.001) for i in range(N)))
    print(f"StreamedLayerBatch ({N} imgs, 1 bind/evict, shared LoRA): grads match resident-sequential = {ok} -> {('PASS' if ok else 'FAIL')}")
    for p in list(m.parameters()) + unds + gens:
        p.grad = None
    outs_o = apply_streamed_batch(m, call_fns, list(zip(unds, gens)), bind=m.bind, evict=m.evict, offload=True)
    sum(((uo ** 2).mean() + (vo ** 2).mean() for uo, vo in outs_o)).backward()
    ok_off = torch.allclose(m.A.grad, mr.A.grad, atol=0.001) and torch.allclose(m.B.grad, mr.B.grad, atol=0.001) and all((torch.allclose(unds[i].grad, ur[i].grad, atol=0.001) for i in range(N)))
    print(f"StreamedLayerBatch offload=True (RAM-offload activations): grads match = {ok_off} -> {('PASS' if ok_off else 'FAIL')}")
    return ok and ok_off
if __name__ == '__main__':
    import sys
    sys.exit(0 if _selftest() and _selftest_multi() and _selftest_tuple() and _selftest_batch() else 1)
