import torch

class ConjugateGradientOptim:
    def backward(self, model, loss):
        
        model.zero_grad()
        loss.backward(create_graph=True)
        
        self.params = [p for p in model.valueEstimationHead.parameters()]
        flat_params_list = []
        flat_grads_list = []
        for p in self.params:
            flat_params_list.append(p.view(-1))
            flat_grads_list.append(p.grad.view(-1))
            
        self.flat_params = torch.cat(flat_params_list)
        self.flat_grads = torch.cat(flat_grads_list)

        search_direction = self.conjugate_gradient(-self.flat_grads)

        max_constraint = 0.01
        sAs = torch.dot(search_direction, self.Av_func(search_direction))
        step_size = torch.sqrt(2 * max_constraint / (sAs + 1e-8))

        new_flat_params = self.flat_params + step_size * search_direction
        offset = 0
        for p in self.params:
            numel = p.numel()
            p.data.copy_(new_flat_params[offset:offset+numel].view_as(p))
            offset += numel

    def conjugate_gradient(self, b, max_iterations=10, tol=1e-10):
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rsold = torch.dot(r, r)
        
        for _ in range(max_iterations):
            Avp = self.Av_func(p)
            alpha = rsold / torch.dot(p, Avp)
            x += alpha * p
            r -= alpha * Avp
            rsnew = torch.dot(r, r)
            if rsnew < tol:
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew
        return x

    def Av_func(self, v):
        damping = 1e-2
        hvp = self.hessian_vector_product(self.flat_grads, self.params, v)
        return hvp + damping * v

    def hessian_vector_product(self, gradients, parameters, vector):
        hvp = torch.autograd.grad(
            gradients,
            parameters,
            grad_outputs=vector,
            retain_graph=True
        )
        hvp_flat = torch.cat([g.contiguous().view(-1) for g in hvp])
        return hvp_flat