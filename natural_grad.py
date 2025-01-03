import torch
from torch.optim.optimizer import Optimizer

class NaturalGradientDescent(Optimizer):
    """
    Natural gradient descent optimizer.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        tol (float, optional): tolerance for conjugate gradient solver (default: 1e-5)
        max_iter (int, optional): maximum number of iterations for conjugate gradient solver (default: 100
    """
    def __init__(self, params, lr=1e-3, tol=1e-5, max_iter=1000):
        if not 0.0 < lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "tol": tol, "max_iter": max_iter}
        super(NaturalGradientDescent, self).__init__(params, defaults)

    def _compute_fisher_vector_product(self, grad):
        """
        Compute Fisher vector product
        
        Args:
            grad (torch.Tensor): gradient tensor
        """
        original_shape = grad.shape
        # Flatten to 2D: (batch_size, -1)
        grad_flat = grad.reshape(original_shape[0], -1)
        # Compute Fisher matrix
        fisher_matrix = grad_flat @ grad_flat.T + 1e-5 * torch.eye(
            grad_flat.size(0), device=grad.device
        )
        return fisher_matrix, original_shape

    def _conjugate_gradient(self, fisher_matrix, grad, tol, max_iter):
        """
        Conjugate gradient method to solve for Fw^-1 @ grad
        
        Args:
            fisher_matrix (torch.Tensor): Fisher matrix
            grad (torch.Tensor): gradient tensor
            tol (float): tolerance for conjugate gradient solver
            max_iter (int): maximum number of iterations for conjugate gradient solver
        """
        original_shape = grad.shape
        grad_flat = grad.reshape(original_shape[0], -1)
        
        # Initialize solution and r vector
        x = torch.zeros_like(grad_flat)
        r = grad_flat.clone()
        p = r.clone()
        rs_old = torch.sum(r * r)

        # Solve for Fw^-1 @ grad using conjugate gradient
        for _ in range(max_iter):
            Ap = fisher_matrix @ p
            alpha = rs_old / (torch.sum(p * Ap) + 1e-8)
            x += alpha * p
            r -= alpha * Ap
            rs_new = torch.sum(r * r)

            # Check for convergence
            if torch.sqrt(rs_new) < tol:
                break

            # Update p vector
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new

        # Reshape solution back to original dimensions
        return x.reshape(original_shape)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure else None

        for group in self.param_groups:
            lr = group["lr"]
            tol = group["tol"]
            max_iter = group["max_iter"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.data
                fisher_matrix, original_shape = self._compute_fisher_vector_product(grad)
                natural_grad = self._conjugate_gradient(fisher_matrix, grad, tol, max_iter)
                
                # Update parameter using natural gradient
                param.data -= lr * natural_grad

        return loss