# Derived from private fork c5ee9ec9dd47786287e576e78fbb42e5f4c4eac3: cosmos_windows/validate_backward_stream.py
# Machine fallback paths cleared; no model data included.
import torch

def _run():
    g = torch.Generator().manual_seed(0)
    Wcpu = torch.randn(8, 8, generator=g)
    A0 = torch.randn(4, 8, generator=g)
    B0 = torch.randn(8, 4, generator=g)
    x0 = torch.randn(3, 8, generator=g)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    load = lambda: Wcpu.to(dev)
    x = x0.to(dev)
    A = A0.clone().to(dev).requires_grad_()
    B = B0.clone().to(dev).requires_grad_()

    class OnDemand(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, A, B):
            ctx.save_for_backward(x, A, B)
            W = load()
            out = x @ W.T + x @ A.T @ B.T
            del W
            return out

        @staticmethod
        def backward(ctx, go):
            x, A, B = ctx.saved_tensors
            W = load()
            gx = go @ W + go @ B @ A
            gA = (go @ B).T @ x
            gB = go.T @ (x @ A.T)
            del W
            return (gx, gA, gB)
    (OnDemand.apply(x, A, B) ** 2).mean().backward()
    Ar = A0.clone().to(dev).requires_grad_()
    Br = B0.clone().to(dev).requires_grad_()
    Wr = load()
    ((x @ Wr.T + x @ Ar.T @ Br.T) ** 2).mean().backward()
    ok = torch.allclose(A.grad, Ar.grad, atol=0.0001) and torch.allclose(B.grad, Br.grad, atol=0.0001)
    print(f"backward-stream lifecycle: gradA match={torch.allclose(A.grad, Ar.grad, atol=0.0001)} gradB match={torch.allclose(B.grad, Br.grad, atol=0.0001)} -> {('PASS' if ok else 'FAIL')}")
    return ok

def _run_multilayer(L=4, D=8):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    g = torch.Generator().manual_seed(1)
    Ws = [torch.randn(D, D, generator=g) for _ in range(L)]
    As = [torch.randn(2, D, generator=g) for _ in range(L)]
    Bs = [torch.randn(D, 2, generator=g) for _ in range(L)]
    x0 = torch.randn(3, D, generator=g)
    res = {'now': 0, 'max': 0}

    def load(i):
        res['now'] += 1
        res['max'] = max(res['max'], res['now'])
        return Ws[i].to(dev)

    def free():
        res['now'] -= 1

    class Layer(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, A, B, i):
            ctx.save_for_backward(x, A, B)
            ctx.i = i
            W = load(i)
            out = x @ W.T + x @ A.T @ B.T
            free()
            return out

        @staticmethod
        def backward(ctx, go):
            x, A, B = ctx.saved_tensors
            i = ctx.i
            W = load(i)
            gx = go @ W + go @ B @ A
            gA = (go @ B).T @ x
            gB = go.T @ (x @ A.T)
            free()
            return (gx, gA, gB, None)
    A1 = [a.clone().to(dev).requires_grad_() for a in As]
    B1 = [b.clone().to(dev).requires_grad_() for b in Bs]
    h = x0.to(dev)
    for i in range(L):
        h = Layer.apply(h, A1[i], B1[i], i)
    (h ** 2).mean().backward()
    Ar = [a.clone().to(dev).requires_grad_() for a in As]
    Br = [b.clone().to(dev).requires_grad_() for b in Bs]
    hr = x0.to(dev)
    for i in range(L):
        hr = hr @ Ws[i].to(dev).T + hr @ Ar[i].T @ Br[i].T
    (hr ** 2).mean().backward()
    ok = all((torch.allclose(A1[i].grad, Ar[i].grad, atol=0.001) and torch.allclose(B1[i].grad, Br[i].grad, atol=0.001) for i in range(L)))
    print(f"{L}-layer chain: all grads match={ok} | max layers resident={res['max']} -> {('PASS' if ok and res['max'] <= 1 else 'FAIL')}")
    return ok and res['max'] <= 1
if __name__ == '__main__':
    import sys
    sys.exit(0 if _run() and _run_multilayer() else 1)
