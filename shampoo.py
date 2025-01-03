import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

class Shampoo(Optimizer):
    def __init__(self, params, lr=1e-1, momentum=0, weight_decay=0, epsilon=1e-4, diag_cutoff=1e3, update_freq_sched=lambda x: 1, inv_p_root_device='cpu', svd_rank=None, newton_num_iters=5, newton_num_iters_max_sv=5):
        """
        Shampoo optimizer for training deep neural networks.

        Args:
            params (iterable): iterable of parameters to optimize or dicts defining parameter groups
            lr (float, optional): learning rate (default: 1e-1)
            momentum (float, optional): momentum factor (default: 0)
            weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            epsilon (float, optional): epsilon value for numerical stability (default: 1e-4)
            diag_cutoff (float, optional): diagonal cutoff value for preconditioner (default: 1e3)
            update_freq_sched (callable, optional): update frequency scheduler for preconditioner (default: lambda x: 1)
            inv_p_root_device (str, optional): device to compute the inverse square root of the preconditioner (default: 'cpu')
            svd_rank (int, optional): rank of the SVD approximation for the inverse square root of the preconditioner (default: None)
            newton_num_iters (int, optional): number of Newton iterations for the inverse square root of the preconditioner (default: 5)
            newton_num_iters_max_sv (int, optional): number of Newton iterations for the largest singular value (default: 5)
        """
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, epsilon=epsilon, diag_cutoff=diag_cutoff)
        super(Shampoo, self).__init__(params, defaults)
        self.update_freq_sched = update_freq_sched
        self.inv_p_root_device = inv_p_root_device
        self.svd_rank = svd_rank
        self.newton_num_iters = newton_num_iters
        self.newton_num_iters_max_sv = newton_num_iters_max_sv

    @torch.no_grad()
    def max_sv(self, G, error_tolerance=1e-6, device='cpu'):
        """
        Computes the largest singular value of G using iteratively.
        
        Args:
            G (torch.Tensor): input matrix
            error_tolerance (float, optional): error tolerance for convergence (default: 1e-6)
            device (str, optional): device to compute the largest singular value (default: 'cpu')
        """
        num_iters = self.newton_num_iters_max_sv
        G = G.to(device)
        n = G.size(0)
        v = torch.randn(n, device=device)
        sv = 0

        for _ in range(num_iters):
            v_hat = v / v.norm()
            v = G @ v_hat
            # old_sv = sv
            sv = v_hat @ v
            # if abs(old_sv - sv) < error_tolerance:
            #     break

        return sv
    
    @torch.no_grad()
    def matrix_pow_newton(self, G, power, error_tolerance=1e-5, ridge_epsilon=1e-6, device='cpu'):
        """
        Computes G^(-1/power) using Coupled Newton iterations.
        
        Args:
            G (torch.Tensor): input matrix
            power (float): power of the matrix
            error_tolerance (float, optional): error tolerance for convergence (default: 1e-5)
            ridge_epsilon (float, optional): ridge epsilon for numerical stability (default: 1e-6)
            device (str, optional): device to compute the matrix power (default
        """
        dev = G.device
        G = G.to(device)
        num_iters = self.newton_num_iters

        mat_size = G.size(0)
        identity = torch.eye(mat_size, device=device)
        max_eigenvalue = self.max_sv(G, error_tolerance=error_tolerance, device=device)
        ridge_epsilon = ridge_epsilon * max(max_eigenvalue, 1e-16)
        damped_mat_g = G + ridge_epsilon * identity

        alpha = -1.0 / power
        z = (1 + power) / (2 * torch.linalg.norm(damped_mat_g))
        mat_m = damped_mat_g * z
        mat_h = identity * z**(1 / power)
        # error = torch.max(torch.abs(mat_m - identity))

        for _ in range(num_iters):
            # if error < error_tolerance:
            #     break
            mat_m_i = (1 - alpha) * identity + alpha * mat_m
            mat_h = mat_h @ mat_m_i
            mat_m = mat_m_i.matrix_power(power) @ mat_m
            # error = torch.max(torch.abs(mat_m - identity))

        return mat_h.to(dev)
    
    @torch.no_grad()
    def matrix_pow_svd(self, matrix, power, rank=None, device='cpu'):
        """
        Computes the matrix power using the SVD decomposition.
        
        Args:
            matrix (torch.Tensor): input matrix
            power (float): power of the matrix
            rank (int, optional): rank of the SVD approximation (default: None)
            device (str, optional): device to compute the matrix power (default
        """
        if matrix.device == device:
            matrix_dev = matrix
        else:
            matrix_dev = matrix.to(device)
        
        rank = min(rank, min(matrix_dev.size())) if rank is not None else min(matrix_dev.size())
        u, sigma, v = torch.svd_lowrank(matrix_dev, q=rank)  # Use low-rank SVD for speed
        return (u @ sigma.pow(power).diag() @ v.t()).to(matrix.device)

    @torch.no_grad()
    def block_diagonal_approx(self, matrix, block_size=32):
        """
        Computes the block-diagonal approximation of the matrix.
        
        Args:
            matrix (torch.Tensor): input matrix
            block_size (int, optional): block size for the approximation (default: 32)
        """
        n = matrix.size(0)
        if n <= block_size:
            return matrix
        num_blocks = (n + block_size - 1) // block_size
        padded_size = num_blocks * block_size
        padding = padded_size - n
        matrix_padded = F.pad(matrix, (0, padding, 0, padding))
        matrix_padded = matrix_padded.view(num_blocks, block_size, num_blocks, block_size)
        eye = torch.eye(num_blocks, device=matrix.device).view(num_blocks, 1, num_blocks, 1)
        matrix_padded = matrix_padded * eye
        matrix_padded = matrix_padded.view(padded_size, padded_size)
        return matrix_padded[:n, :n]
            
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Receive loss function f_t: R^(n1x...xnk) -> R
                grad = p.grad.data          # Gt
                n_dim = grad.ndimension()   # k
                grad_size = grad.size()     # (n1, ..., nk)
                momentum = group['momentum']
                weight_decay = group['weight_decay']

                # Initialize the preconditioner matrix
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    if momentum > 0:
                        state['momentum_buffer'] = grad.clone()

                    # Create preconditioner matrices for each dimension of the gradient
                    for i, dim in enumerate(grad.size()):
                        if (dim > group['diag_cutoff']):
                            # Initialize the preconditioner H_0^i = epsilon * I_ni
                            state[f'precond_{i}'] = group['epsilon'] * torch.ones(dim, device=grad.device)
                            state[f'inv_precond_{i}'] = torch.zeros(dim, device=grad.device)
                        else:
                            # Initialize the preconditioner H_0^i = epsilon * I_ni
                            state[f'precond_{i}'] = group['epsilon'] * torch.eye(dim, device=grad.device)
                            state[f'inv_precond_{i}'] = torch.zeros(dim, dim, device=grad.device)

                update_freq = self.update_freq_sched(state['step'])

                if momentum > 0:
                    # Apply momentum to the gradient
                    grad = grad * (1 - momentum) + state["momentum_buffer"] * momentum

                if weight_decay > 0:
                    # Add weight decay to the gradient [L2 regularization: L<-L+λw**2 => grad<-grad+2λw]
                    # grad = grad + p.data * weight_decay
                    grad.add_(p.data, alpha=weight_decay)

                # Update the weights for each dimension of the gradient
                for i, dim in enumerate(grad.size()):
                    precond = state[f'precond_{i}']

                    ## Block-diagonal approximation
                    # if (dim > group['diag_cutoff']):
                    #     precond = self.block_diagonal_approx(precond, block_size=4)

                    inv_precond_pow = state[f'inv_precond_{i}']

                    grad = grad.transpose(0, i).contiguous()
                    grad_size_t = grad.size()
                    grad = grad.view(dim, -1)

                    if (dim > group['diag_cutoff']):
                        # Use diagonal variant of shampoo
                        # precond += torch.diagonal(grad @ grad.t())
                        precond.add_(torch.diagonal(grad @ grad.t()))
                        if state['step'] % update_freq == 0:
                            inv_precond_pow = precond.pow(-1 / (2*n_dim)).clone()
                    else:
                        # Update preconditioner matrix
                        # precond += grad @ grad.t()
                        precond.addmm_(grad, grad.t())
                        if state['step'] % update_freq == 0:
                            # inv_precond_pow = self.matrix_pow_svd(precond, -1 / (2*n_dim), rank=self.svd_rank, device=self.inv_p_root_device).clone()
                            inv_precond_pow = self.matrix_pow_newton(precond, 2*n_dim, device=self.inv_p_root_device).clone()

                    # Compute the preconditioned gradient
                    if i == n_dim - 1:
                        if (dim > group['diag_cutoff']):
                            grad = grad.t() @ inv_precond_pow.diag()
                        else:
                            grad = grad.t() @ inv_precond_pow
                        grad = grad.view(grad_size)
                    else:
                        if (dim > group['diag_cutoff']):
                            grad = inv_precond_pow.diag() @ grad
                        else:
                            grad = inv_precond_pow @ grad
                        grad = grad.view(grad_size_t)

                # Update the state
                state['step'] += 1
                state['momentum_buffer'] = grad

                # Update the weights                
                p.data = p.data - grad * group['lr']

        return loss