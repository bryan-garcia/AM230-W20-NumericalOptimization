"""
HW_2.py
AM 230 - Numerical Optimization
Programmer: Bryan Garcia
Date: 1 February 2020
"""

import matplotlib
matplotlib.use("tkagg")

import numpy as np
from numdifftools import Hessian, Gradient
from scipy.optimize import line_search, fmin, rosen, rosen_der, rosen_hess
from scipy.io import loadmat
import matplotlib.pyplot as plt
import step_length as SL

"""
Cost function, gradient, and Hessian for Problem 2
"""
def rosenbrock( x ):
	x1, x2 = x
	return ( 100 * ( x2 - x1 ** 2 ) ** 2 + ( 1 - x1 ) ** 2 )

def grad_rosenbrock( x ):
	x1, x2 = x
	return np.array( [ 400 * ( x1 ** 3 ) - 400 * ( x1 * x2 ) + 2 * x1 - 2,
						200 * x2 - 200 * ( x1 ** 2 ) ] )
def hessian_rosenbrock( x ):
	x1, x2 = x
	h_11 = 1200 * ( x1 ** 2 ) - 400 * x2 + 2
	h_12 = -400 * x1
	h_21 = -400 * x1
	h_22 = 200
	return np.array( [ np.array( [ h_11, h_12 ] ) ,	np.array( [ h_21, h_22 ] ) ] )

"""
Cost function, gradient, and Hessian for Problem 3
"""
def P3_func( x ):
	x1, x2, x3 = x
	return ( 1 / 4 * x1 ** 4 + 1 / 2 * ( x2 - x3 ) ** 2 + 1 / 2 * x2 ** 2 )

def P3_grad( x ):
	x1, x2, x3 = x
	return np.array( [ x1 ** 3 , 2 * x2 - x3 , x3 - x2 ] )

def P3_hessian( x ):
	x1, x2, x3 = x
	h11 = 3 * x1 ** 2
	h12 = 0 
	h13 = 0
	h21 = 0
	h22 = 2
	h23 = -1
	h31 = 0
	h32 = -1
	h33 = 1
	return np.array( [ np.array( [ h11 , h12 , h13] ) , 
					   np.array( [ h21 , h22 , h23] ) ,
					   np.array( [ h31 , h32 , h33] ) ] )

def steepest_descent( f , grad , x0 , tol = 1e-8 , max_iter = 1000 , 
	full_output = True , error_analysis = None , grad_analysis = None , verbose = False ):
	"""
	steepest_descent
		
		Steepest Descent method for optimization of a given cost function
		at some initial point.

		Parameters:
			f : function
				Cost function to minimize.

			grad : function
				Gradient of the given cost function f.

			x0 : array-like
				Vector corresponding to the initial condition. This
				will generate the initial search direction.

			tol : float (optional)
				Tolerance for convergence checking

			max_iter : int (optional)
				Max number of iterations to apply steepest descent.

			full_output : bool (optional)
				When set to True, the method returns all computed points
				xk during the steepest descent search.

			error_analysis : array-like (optional)
				User provided vector corresponding to a minimizer of the 
				cost function f. To be used when convergence analysis is
				to be performed.

			verbose : bool (optional)
				When set to true, prompt user during application of 
				steepest descent.

		Returns:
			output : list
				A list of two entries is returned if error_analysis is set
				to True.

				First entry of output corresponds to xk iterates generated
				during steepest descent (single float if full_output is False).
				
				Second entry of output corresponds to error values generated
				during steepest descent ()

				If error_analysis is set to False, a single np.array object 
				is returned corresponding to the minimized point.
			
			None : NoneType
				Should steepest descent fail, None is returned

	"""

	# Flag for error checking
	converged = False

	# Unpack user provided initial point
	xk = np.array( [ x for x in x0 ] )

	# Check for full_output or error_analysis flags
	if full_output: 
		xvals = []
	if error_analysis is not None:
		error = [] if full_output else error_analysis
	if grad_analysis is not None:
		gvals = []

	# Beginning steepest descent...
	for k in range( int( max_iter ) + 1):

		# Collecting current iterate
		if full_output:
			xvals.append( np.array( xk ) )
			# Compute error at current iteration
			if error_analysis is not None:
				error.append( np.linalg.norm( error_analysis - xk ) )

		elif error_analysis is not None:
			error = np.abs( error_analysis - xk )

		# Generate search direction
		pk = -grad( xk )
		if grad_analysis is not None:
			gvals.append( np.linalg.norm( pk ) )

		# Check for convergence
		if np.linalg.norm( pk ) < tol:
			if verbose:
				print( "steepest_descent converged to " \
					+ "{} in {} iterations...".format( xk , k + 1 ) )
			converged = True
			break

		# Performing line_search for Wolfe friendly step-size
	#try:
		alpha = SL.step_length( f , grad , xk , pk )
		# print( alpha )
	# except:
	# 	print( 'xk: {}, {}'.format(xk[0], xk[1]) )

		# Checking if 
		# if alpha[0] is None:
		# 	alpha = alpha[1]
		# else:
		# 	alpha = alpha[0]

		# Obtaining next iterate x_k+1
		xk += alpha * pk

		# print( np.linalg.norm( pk ) )  
	# Checking convergence flag (fails if num of iterations are over max_iter)
	if not converged:
		print( "Did not converge to a solution with desired tolerance...")

	# Prepare output 
	output = []
	if full_output:
		output.append( np.asarray( xvals ) )
	else:
		output.append( xk )
	if error_analysis is not None:
		output.append( np.asarray( error ) )
	if grad_analysis is not None:
		output.append( np.asarray( gvals ) )

	return output if len( output ) > 1 else output[0]

def newtons_method( f , grad , hessian , x0 , tol = 1e-8 , max_iter = 1e8 , 
	full_output = True,  error_analysis = None , grad_analysis = None , verbose = False ):
	"""
	newtons_method
		
		Newton's method for optimization of a given cost function
		at some initial point.

		Parameters:
			f : function
				Cost function to minimize.

			grad : function
				Gradient of the given cost function f.

			hessian : function
				Hessian of the given cost function f.

			x0 : array-like
				Vector corresponding to the initial condition. This
				will generate the initial search direction.

			tol : float (optional)
				Tolerance for convergence checking

			max_iter : int (optional)
				Max number of iterations to apply.

			full_output : bool (optional)
				When set to True, the method returns all computed points
				xk during the optimization.

			error_analysis : array-like (optional)
				User provided vector corresponding to a minimizer of the 
				cost function f. To be used when convergence analysis is
				to be performed.

			verbose : bool (optional)
				When set to true, prompt user during optimization.

		Returns:
			output : list
				A list of two entries is returned if error_analysis is set
				to True.

				First entry of output corresponds to xk iterates generated
				during optimization (single float if full_output is False).
				
				Second entry of output corresponds to error values generated
				during optimization ()

				If error_analysis is set to False, a single np.array object 
				is returned corresponding to the minimized point.
			
			None : NoneType
				Should optimization fail, None is returned

	"""
	
	# Flag for error checking
	coverged = False

	# Unpack user provided initial point
	xk = np.array( [ x for x in x0 ] )

	# Check for full_output or error_analysis flags
	if full_output: 
		xvals = []
	if error_analysis is not None:
		error = [] if full_output else error_analysis
	if grad_analysis is not None:
		gvals = []

	# Beginning steepest descent...
	for k in range ( int( max_iter + 1 ) ):

		# Collecting current iterate
		if full_output:
			xvals.append( np.array( xk ) )
			# Compute error at current iteration
			if error_analysis is not None:
				error.append( np.linalg.norm( error_analysis - xk ) )

		elif error_analysis is not None:
			error = np.abs( error_analysis - xk )

		# Compute gradient at xk
		grad_xk = grad( xk )
		if grad_analysis:
			gvals.append( np.linalg.norm( grad_xk) )

		# Checking if cost function sufficiently minimized
		if np.linalg.norm( grad_xk ) < tol:
			if verbose:
				print( "newtons_method converged to point " \
					+ "{} in {} iterations...".format( xk, k + 1 ) )
			converged = True
			break

		# Computing Hessian at xk
		hessian_xk = hessian( xk )

		# Prompt user if verbose is set to True
		if verbose:
			print("Hessian is positive definite at current xk value " \
				+ "{}...".format( xk ) ) \
			if np.dot( xk, np.dot( hessian_xk, xk ) ) > 0 \
		 		else print("Hessian is not positive definite at current " \
		 			+ "xk value {}...".format( xk ) )

		# Solve for search direction pk
		pk = -np.linalg.solve( hessian_xk, grad_xk )

		# Obtain x_k+1 iterate
		xk += pk


	# Checking convergence flag (fails if num of iterations are over max_iter)
	if not converged:
		print( "Did not converge to a solution...")
		return None

	# Prepare output 
	output = []
	if full_output:
		output.append( np.asarray( xvals ) )
	else:
		output.append( xk )
	if error_analysis is not None:
		output.append( np.asarray( error ) )
	if grad_analysis is not None:
		output.append( np.asarray( gvals ) )

	return output if len( output ) > 1 else output[0]


def Problem_2( ):
	"""
	Relevant code for Problem 2.
	"""

	# Setting initial point and optimal point
	x_initial = np.array( [ 1.1 , 1.1] )
	x_optimal = np.array( [ 1 , 1 ] )

	# Get xk iterations and corresponding error values for steepest_descent
	steepest_descent_xvals, steepest_descent_error = \
	steepest_descent( rosenbrock, grad_rosenbrock, x_initial, 
		full_output = True , error_analysis = x_optimal , verbose = False ) 

	# Get xk iterations and corresponding error values for newtons_method
	newtons_xvals, newtons_error = \
	newtons_method( rosenbrock, grad_rosenbrock, hessian_rosenbrock, x_initial,
		full_output = True , error_analysis = x_optimal , verbose = False ) 

	# Plotting relevant data
	plt.subplot( 221 )
	plt.title( r"Steepest Descent: $x_0 = $ ({}, {})".format( x_initial[0] , x_initial[1] ) )
	plt.plot( np.arange( len( steepest_descent_error ) ) , np.log( steepest_descent_error ) , label = 'Steepest Descent' )
	plt.xlabel( r"$K$ number of iterations", fontsize = 12)
	plt.ylabel( r'$log|x^{*} - x_k|$', fontsize = 12)
	plt.grid( True )
	plt.legend()

	kiter = np.arange( len ( newtons_error ) )
	# pfit = np.polyfit( kiter , np.log( newtons_error ) , deg = 2 )
	plt.subplot( 222 )
	plt.title( r"Newton's Method: $x_0 = $ ({}, {})".format( x_initial[0] , x_initial[1] ) )
	plt.plot( kiter , np.log( newtons_error ) , color = 'orange' , label = 'Newton\'s Method' )
	# plt.plot( kiter, pfit[0] * kiter ** 2 + pfit[1] * kiter + pfit[2] , linestyle = '--', color = 'red' , label = 'Quadratic fit')
	plt.xlabel( r"$K$ number of iterations", fontsize = 12)
	plt.ylabel( r'$log|x^{*} - x_k|$', fontsize = 12)
	plt.grid( True )
	plt.legend()

	# Setting initial point and optimal point
	x_initial = np.array( [ 1.1 , 1.2] )
	x_optimal = np.array( [ 1 , 1 ] )

	# Get xk iterations and corresponding error values for steepest_descent
	steepest_descent_xvals, steepest_descent_error = \
	steepest_descent( rosenbrock, grad_rosenbrock, x_initial, 
		full_output = True , error_analysis = x_optimal , verbose = False ) 

	# Get xk iterations and corresponding error values for newtons_method
	newtons_xvals, newtons_error = \
	newtons_method( rosenbrock, grad_rosenbrock, hessian_rosenbrock, x_initial,
		full_output = True , error_analysis = x_optimal , verbose = False ) 

	# Plotting relevant data
	plt.subplot( 223 )
	plt.title( r"Steepest Descent: $x_0 = $ ({}, {})".format( x_initial[0] , x_initial[1] ) )
	plt.plot( np.arange( len( steepest_descent_error ) ) , np.log( steepest_descent_error ) , label = 'Steepest Descent' )
	plt.xlabel( r"$K$ number of iterations", fontsize = 12)
	plt.ylabel( r'$log|x^{*} - x_k|$', fontsize = 12)
	plt.grid( True )
	plt.legend()

	kiter = np.arange( len ( newtons_error ) )
	# pfit = np.polyfit( kiter , np.log( newtons_error ) , deg = 2 )
	plt.subplot( 224 )
	plt.title( r"Newton's Method: $x_0 = $ ({}, {})".format( x_initial[0] , x_initial[1] ) )
	plt.plot( kiter , np.log( newtons_error ) , color = 'orange' , label = 'Newton\'s Method' )
	# plt.plot( kiter, pfit[0] * kiter ** 2 + pfit[1] * kiter + pfit[2] , linestyle = '--', color = 'red' , label = 'Quadratic fit')
	plt.xlabel( r"$K$ number of iterations", fontsize = 12)
	plt.ylabel( r'$log|x^{*} - x_k|$', fontsize = 12)
	plt.grid( True )
	plt.legend()

	plt.tight_layout()
	plt.show()

def Problem_3( ):
	"""
	Relevant code for Problem 3
	"""

	# Setting initial point and optimal point
	x_initial = np.array( [ 1.0 , 1.0 , 1.0 ] )
	x_optimal = np.array( [ 0 , 0 , 0 ] )

	# Get xk iterates and corresponding error values for newtons_method
	newtons_xvals, newtons_error = \
	newtons_method( P3_func , P3_grad , P3_hessian , x_initial, 
		full_output = True , error_analysis = x_optimal, verbose = True )

	# np.array listing number of iterations
	kiter = np.arange( len( newtons_error ) )

	plt.subplot( 111 )
	plt.title( r"Convergence Analysis: $x_0 = $ ({}, {}, {})".format( x_initial[0] , x_initial[1] , x_initial[2] ) )
	plt.plot( kiter , np.log( newtons_error ) , color = 'blue' , label = 'Newton\'s Method' )

	# Performing linear regression to illustrate linear convergence
	pfit = np.polyfit( kiter , np.log( newtons_error ) , deg = 1 )
	plt.plot( kiter , pfit[0] * kiter + pfit[1] , linestyle = '--' , color = 'salmon' , label = 'Linear fit with slope {}'.format( pfit[0] ) )
	plt.xlabel( r"$K$ number of iterations", fontsize = 12)
	plt.ylabel( r'$log|x^{*} - x_k|$', fontsize = 12)
	plt.grid( True )
	plt.legend()

	plt.tight_layout()
	plt.show()

def Problem_4( ):
	"""
	Relevant code for Problem 4
	"""

	"""
	PDF, cost function, gradient, and Hessian for Problem 4
	"""
	def P4_PDF( label , data , theta ):
		return 1 / ( 1 + np.exp( -label * np.dot( theta , data ) ) )

	def P4_cost( theta_vect ):
		ll_sum = 0
		for k in range( len ( DATA ) ):
			ll_sum += np.log( P4_PDF( LABELS[ k ] , DATA[ k ] , theta_vect ) )
		return 1 / 2 * np.linalg.norm( theta_vect ) ** 2 - ll_sum

	def P4_gradient( theta_vect ):
		_sum = 0
		for k in range( len( DATA ) ):
			L = LABELS[ k ]
			X = DATA[ k ]
			_sum += ( L * X * np.exp( -L * np.dot( theta_vect , X ) ) ) / ( ( 1 + np.exp( -L * np.dot( theta_vect , X ) ) ) )
		return theta_vect - _sum

	# Read in raw .mat files as dictionaries
	data_dict = loadmat( 'DATA.mat' )
	label_dict = loadmat( 'LABELS.mat' )

	# Extracting relevant data for completion of Problem 4
	DATA = np.asarray( [ d for d in data_dict[ 'DATA' ] ] )
	LABELS = np.asarray( [ l for l in label_dict[ 'LABELS' ] ] )
	theta_initial = np.asarray( [ 1.6 , 2.1 ] )

	steepest_descent_output = steepest_descent( P4_cost , P4_gradient , theta_initial , \
		full_output = True, grad_analysis = True)
	newtons_method_output = newtons_method( P4_cost , P4_gradient , Hessian( P4_cost ) , theta_initial , \
		full_output = True, grad_analysis = True)

	print( "=" * 50 )
	print( "Steepest Descent converged to theta = ({}, {})".format(steepest_descent_output[0][-1][0], steepest_descent_output[0][-1][1]))
	print( "Newton's Method converged to theta = ({}, {})".format(newtons_method_output[0][-1][0], newtons_method_output[0][-1][1]))
	print( "=" * 50 )

	plt.subplot(211)
	plt.ylabel( r'$log \vert \vert \nabla f \vert \vert$', fontsize = 12)
	plt.xlabel( r"$K$ number of iterations", fontsize = 12)
	plt.title( r"Convergence to $\theta = $ ({}, {})".format(steepest_descent_output[0][-1][0], steepest_descent_output[0][-1][1]))
	plt.plot( [ n for n in range(len(steepest_descent_output[1])) ], np.log(steepest_descent_output[1]), label = 'Steepest Descent' )
	plt.grid(True)
	plt.legend()

	plt.subplot(212)
	plt.ylabel( r'$log \vert \vert \nabla f \vert \vert$', fontsize = 12)
	plt.xlabel( r"$K$ number of iterations", fontsize = 12)
	plt.title( r"Convergence to $\theta = $ ({}, {})".format(newtons_method_output[0][-1][0], newtons_method_output[0][-1][1]))
	plt.plot( [ n for n in range(len(newtons_method_output[1])) ], np.log(newtons_method_output[1]), color = 'orange', label = "Newton's Method")
	plt.grid(True)
	plt.legend()

	plt.tight_layout()
	plt.show()

if __name__ == "__main__":
	Problem_2()
	Problem_3()
	Problem_4()












