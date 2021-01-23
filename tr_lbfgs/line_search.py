"""
LineSearch.py

Helper class for line search methods.

Credit: Schuyler Krawczuk, UCSC
"""
from scipy.optimize import line_search
import numpy as np


class LineSearch:
    def __init__(self, cost_function, grad_function, max_iterations, **kwargs):
        self.cost = lambda x: cost_function(x, **kwargs)
        self.grad_cost = lambda x: grad_function(x, **kwargs)
        self.max_iterations = max_iterations

    def minimize(self, x0, tol, verbose=False):
        """
        :param x0: initial guess
        :param tol: stopping tolerance for p magnitude
        :param c1: Wolfe condition lower bound
        :param c2: Wolfe condition upper bound
        :param a_max: maximum step size
        :param verbose: interval of logging update
        :return: x for each iteration
        """
        x = np.zeros((self.max_iterations+1, x0.shape[0]))
        x[0] = x0
        for i in range(self.max_iterations):
            p = self.step_direction(x[i])
            # a = line_search(self.cost, self.grad_cost, x[i], p.squeeze())[0]
            a = self.step_length(p, x[i])

            try:
                x[i+1] = x[i] + a * p
            except TypeError:
                print('iteration {}: Not a descent direction'.format(i))
                return x[:i+1]

            if np.linalg.norm(p, np.inf) <= tol:
                break
            if verbose:
                if i % verbose == 0:
                    print('iter', i, 'of', self.max_iterations)

        return x[:i+1]

    def step_direction(self, x):
        """
        steepest descent direction
        """
        return -self.grad_cost(x)

    def step_length(self, p, x, c1=0.1, c2=0.9, a_max=1e6, max_iters=50):
        a0 = 0
        a_ast = 0.5 * a_max

        phi0 = self.cost(x)
        dphi0 = self.grad_cost(x).T @ p
        phi_prev = phi0

        for i in range(1, max_iters):
            phi = self.cost(x + a_ast * p)

            if np.all(phi > phi0 + c1 * a_ast * dphi0) or (np.all(phi >= phi_prev) and i > 1):
                a_ast = self.zoom(a0, a_ast, c1, c2, x, p, phi0, dphi0)
                break

            dphi = self.grad_cost(x + a_ast * p).T @ p
            if np.all(abs(dphi) <= -c2 * dphi0):
                break

            if np.all(dphi >= 0):
                a_ast = self.zoom(a_ast, a0, c1, c2, x, p, phi0, dphi0)
                break

            a0 = a_ast
            a_ast = 0.5 * (a_ast + a_max)
            phi_prev = phi

        return a_ast

    def zoom(self, a_low, a_high, c1, c2, x, p, phi0, dphi0, max_iters=50):
        phi_low = self.cost(x + a_low * p)

        for i in range(max_iters):
            a = 0.5 * (a_low + a_high)
            phi = self.cost(x + a * p)
            if np.all(phi > phi0 + c1 * a * dphi0) or np.all(phi >= phi_low):
                a_high = a
            else:
                dphi = self.grad_cost(x + a * p).T @ p

                if np.all(np.abs(dphi) <= -c2 * dphi0):
                    break

                if np.all(dphi * (a_high - a_low) >= 0):
                    a_high = a_low

                a_low = a
                phi_low = phi
        return a
