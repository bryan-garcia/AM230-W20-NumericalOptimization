"""
step_length.py

	Description:
		Algorithm 3.5 in Numerical Optimization by Nocedal and Wright.
		
"""
import numpy as np

def step_length( f, grad, x, p, c1 = 0.9, c2 = 0.1, a_max = 1e6, max_iter = 50000):
	"""
	step_length:

		Description:
			Computes a step_length that satisfies the Wolfe conditions.
		Parameters:
			f : function
				Cost function
			grad : function
				Gradient of f
			x : nd.array
				Current point
			p : nd.array
				Search direction
			c1 : float (optional)
				Nocedal & Wright c1 constant
			c2 : float (optional)
				Nocedal & Wright c2 constant
			a_max : float (optional)
				Maximum step size
		Returns:
			a_star : float
				Step length that satisfies Wolfe conditions
	"""

	def Zoom(x0, p, alpha_low, alpha_high, c1, c2):
		i = 0
		MAX_ITER = 10

		fx0 = f(x0);
		gx0 = np.dot( grad(x0).T, p )

		while True:
		    
		    alpha_x = 0.5*(alpha_low + alpha_high)
		    alpha = alpha_x
		    xx = x0 + alpha_x*p
		    fxx = f(xx)
		    gxx = grad(xx)
		    fs = fxx
		    gs = gxx
		    gxx = np.dot( gxx.T, p )
		    x_low = x0 + alpha_low*p
		    f_low = f(x_low)
		    
		    if ((fxx > fx0 + c1*alpha_x*gx0) or (fxx >= f_low)):
		        alpha_high = alpha_x
		    else:
		        if abs(gxx) <= -c2*gx0:
		            alpha = alpha_x
		            return alpha
		        if gxx*(alpha_high - alpha_low) >= 0:
		            alpha_high = alpha_low

		        alpha_low = alpha_x;
		    
		    i = i+1
		    if i > MAX_ITER:
		        alpha = alpha_x
		        return alpha

		"""'''''''''''''''''''''''''''''''''''''''''''''''''''''''"""
		"""'''''''''''''''''''''''''''''''''''''''''''''''''''''''"""

	MAX_ITER = 3
	alpha_p = 0
	alpha_x = 1

	fx0 = f(x)
	gx0 = grad(x)

	gx0 = np.dot(gx0.T, p)
	fxp = fx0;

	i=1;

	while True:
	    xx = x + alpha_x*p
	    fxx = f(xx)
	    gxx = grad(xx)
	    
	    gxx = np.dot(gxx.T,p)
	    if (fxx > fx0 + c1*alpha_x*gx0) or ((i > 1) and (fxx >= fxp)):
	        alpha = Zoom(x, p, alpha_p, alpha_x, c1, c2)
	        return alpha
	    
	    if abs(gxx) <= -c2*gx0:
	        alpha = alpha_x
	        return alpha
	    
	    if gxx >= 0:
	        alpha = Zoom(x, p, alpha_x, alpha_p, c1, c2)
	        return alpha
	    
	    alpha_p = alpha_x
	    fxp = fxx

	    if i > MAX_ITER:
	        alpha = alpha_x
	        return alpha

	    r = 0.8
	    alpha_x = alpha_x + (a_max-alpha_x)*r
	    i = i+1







