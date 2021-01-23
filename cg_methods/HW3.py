"""
HW3.py

Author: Bryan Garcia
Date: 18 Feb. 2020

Description: Associated code for AM 230 Homework 3. Contains code illustrating
conjugate gradient methods for numerical optimization.

"""

import matplotlib
matplotlib.use( "tkagg" )
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import line_search, fmin_cg
from numdifftools import Gradient
from line_search import LineSearch

"""
Cost Functions, residual, gradient definition, and matrix construction helper.
"""

def P3_cost( x , A , b ):
	"""
	P3_cost

	Description: 
		Cost function of the form: 
		f( x ) = 1 / 2 * ( x`A x ) - b`x

	Parameters:
		x : nd.array
			Vector to be evaluated
		
		matrix_type : str (optional)
			Type of matrix A to be constructed.

			options:
				'0' : uniform distribution of eigenvalues (default)
				'1' : clustered eigenvalues

	Return:
		Scalar value
	"""

	return ( 1 / 2 ) * ( np.dot( np.dot( x.T , A ) , x ) ) - np.dot( b.T , x )

def P3_residual( x , A , b):
	"""
	P3_residual:

		Description:
			The equivalent linear system to solve. That is:
				r_k = Ax_k - b 

		Parameters:
			x : nd.array
				Vector to be evaluated

		Return:
			nd.array
	"""
	return np.dot( A , x ) - b


def P4_cost( x ):
	"""
	P4_cost:

		Description:
			Cost function associated with problem 4 on the HW_3 pdf.

		Parameters:
			x : nd.array
				Vector to be evaluated
		Return:
			float scalar cost
	"""
	if x.shape == ( x.shape[0] , 1 ):
		xi = x.flatten( )
	else:
		xi = x

	N = xi.shape[ 0 ]
	cost = 0
	for i in range( N - 1 ):
		cost += 100 * ( xi[ i ] ** 2 - xi[ i + 1 ] ) ** 2 + ( xi[ i ] - 1 ) ** 2
	return cost

def P4_grad( x ):
	if x.shape == ( x.shape[0] , 1 ):
		xi = x.flatten( )
	else:
		xi = x

	N = xi.shape[ 0 ]
	grad = np.zeros( N )

	grad[ 0 ] = 400 * xi[ 0 ] * ( xi[ 0 ] ** 2 - xi[ 1 ] ) + 2 * ( xi[ 0 ] - 1)
	grad[ N - 1 ] = -200 * ( xi[ N - 2 ] ** 2 - xi[ N - 1] )

	for i in range( 1 , N - 1 ):
		grad[ i ] = -200 * ( xi[ i - 1 ] ** 2 - xi[ i ] ) \
			+ 400 * xi[ i ] * ( xi[ i ] ** 2 - xi[ i + 1 ] ) \
			+ 2 * ( xi[ i ] - 1)

	return grad

def get__matrix( mtype = '0' , n = 10e3 ):
	"""
	get__matrix:

		Description:
			Constructs a randomly generated positive-definite symmetrix matrix
			of dimension n x n.

		Parameters:
			mtype : string
				Specification of type of matrix to construct
					Options: 
						'0' : Uniform distribution of eigenvalues (default)
						'1' : Clustered distribution of eigenvalues
			n : integer
				Dimension specification for n x n matrix

		Return:
			nxn Matrix as type nd.array
	"""
	n = int ( n )

	if mtype not in [ '0' , '1' ]:
		mtype = '0'

	if mtype == '0':
		eigenvalues = np.linspace( 10 , n , num = n )
	elif mtype == '1':
		n_rand = np.random.randint( 1 , n + 1 )
		eigenvalues = [ i for i in np.linspace( 9 , 11 , num = n_rand ) ] \
		+ [ i for i in np.linspace( 999, 1001 , num = n - n_rand ) ]

	Q_matrix = np.linalg.qr( np.random.rand( n , n ) )[0]
	Diag_eig_matrix = np.diagflat( eigenvalues )

	return np.dot( np.dot( Q_matrix.T , Diag_eig_matrix) , Q_matrix ), eigenvalues


"""
Optimization algorithms: CG, FR_CG, FR_CG (with restart)
"""

def conjugate_gradient( f , x_0 , A, b, tol = 1e-05, error_analysis = None ):
	"""
	conjugate_gradient:

		Description:
			Linear conjugate gradient method for optimization.

		Parameters:
			f : function
				Cost function to minimize
			x_0 : nd.array
				Initial point to begin CG method
			tol: float (optional)
				Tolerance of optimizer

		Return:
			x_k : nd.array
				Solution that minimizes cost function
			k : int
				Number of iterations
	"""
	k = 0
	x_k = x_0
	r_k = P3_residual( x_k , A , b )
	p_k = -r_k

	redux = []
	error_norm = []

	while np.linalg.norm( r_k ) > tol:

		redux.append( np.linalg.norm(r_k) )
		r_k_old = r_k

		if error_analysis is not None:
			error_norm.append( np.dot( np.dot( ( x_k - error_analysis ).T , A ) , \
			 ( x_k - error_analysis ) ) )

		alpha_k = np.dot( r_k.T , r_k ) / np.dot( p_k.T , np.dot( A , p_k ) )
		x_k = x_k + alpha_k * p_k

		r_k = r_k_old + alpha_k * np.dot( A , p_k )
		beta_k = np.dot( r_k.T , r_k ) / np.dot( r_k_old.T , r_k_old )
		p_k = -r_k + beta_k * p_k
		
		k += 1

	output = [ x_k , r_k , np.asarray( redux ) ]

	if error_analysis is not None:
		output.append( error_norm )
	return output , k 

def FR_CG( f , grad , x_0 , tol = 1e-05 , mode = '0', nu = 0.25, max_iter = 5000):

	if x_0.shape == ( x_0.shape[ 0 ] , 1 ):
		x_0 = x_0.flatten( )

	# Given x_0...
	x_current = x_0

	# Evaluate f_0...
	f_current = f( x_current )

	# Evaluate grad_0...
	grad_current = grad( x_current )

	# Set p_0 = - grad_0...
	p_current = - grad_current

	# k <- 0...
	k = 0

	redux = []

	# While grad_k != 0...
	while ( np.linalg.norm( grad_current , ord = np.inf) ) > tol:

		redux.append( np.linalg.norm( grad_current ) )
		
		# Compute alpha_k...

		alpha_current = line_search( f, grad, x_current, p_current )[0]

		if alpha_current is None:
			alpha_current = LineSearch( f, grad, 1000).step_length( p_current, x_current )

		# Set x_{ k + 1 } = x_k + alpha_k * p_k...
		x_next = x_current + alpha_current * p_current

		# Evaluate grad_{ k + 1 }...
		grad_next = grad( x_next )

		# Evaluate beta_FR with grad_{ k + 1} and grad_{ k }
		beta_FR = np.linalg.norm( grad_next ) / np.linalg.norm( grad_current )
		if mode == '1':
			if k > 0 and abs( np.dot( grad_current.T, grad_past ) ) / ( np.linalg.norm( grad_past )  ** 2 ) >= nu:
				beta_FR = beta_FR * 0

		elif mode == '2':
			beta_FR = np.dot( grad_next.T, grad_next - grad_current) / ( np.linalg.norm( grad_current ) ** 2 )

		# Evaluate p_{ k + 1 }...
		p_next = - grad_next + beta_FR * p_current

		# Update k...
		k += 1

		# Loop stuff for next iteration...
		x_current = np.copy( x_next )
		p_current = np.copy( p_next )
		grad_past = np.copy( grad_current )
		grad_current = np.copy( grad_next )

		if k >= max_iter:
			break

	return (x_current, grad_current, np.asarray( redux ) ), k



def Problem_3():
	"""
	Problem_3:

		Description/requirements:
			Perform conjugate gradient optimization on the cost function
			P3_cost. Optimize with both uniform and clustered eigenvalue
			style matrix types and compare the performance of CG.

			Extra points if you can explain your numerical findings using 
			the theoretical convergence results discussed in the lecture.
	"""
	DIM_P3 = 1000

	x = np.random.rand( DIM_P3 , 1)
	A_uniform, eig_uniform = get__matrix( mtype = '0', n = DIM_P3 )
	A_clustered, eig_clustered = get__matrix( mtype = '1', n = DIM_P3 )
	b = np.random.rand( DIM_P3 , 1 )

	x_uniform, k0 = conjugate_gradient( P3_residual , x , A_uniform , b )
	x_clustered, k1 = conjugate_gradient( P3_residual , x , A_clustered , b )
	print( '\n"''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''"')
	print( 'CG with uniform eigenvalues ran for {} iterations and ended with {} residual.'.format(k0, P3_residual(x_uniform[0], A_uniform, b)))
	print( 'CG with clustered eigenvalues ran for {} iterations and ended with {} residual.'.format(k1, P3_residual(x_clustered[0], A_clustered, b)))
	print( '"''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''"\n')


	kiter0 = np.arange(k0)
	plt.plot(kiter0, np.log10(x_uniform[2][:k0]), color = 'cornflowerblue', label = 'Uniform Eigenvalues: {} iteration convergence'.format(k0))
	plt.legend()
	plt.grid()
	plt.title("Convergence of Linear CG: Uniformly Distributed Eigenvalues")
	plt.ylabel(r"$\log_{10}( \vert \vert \nabla f \vert \vert )$")
	plt.xlabel(r"Iteration number $k$")
	plt.tight_layout()
	plt.show()

	kiter1 = np.arange(k1)
	plt.plot(kiter1, np.log10(x_clustered[2][:k1]), color = 'deepskyblue', label = 'Clustered Eigenvalues: {} iteration convergence'.format(k1))
	plt.legend()
	plt.grid()
	plt.title("Convergence of Linear CG: Clustered Eigenvalues")
	plt.ylabel(r"$\log_{10}( \vert \vert \nabla f \vert \vert )$")
	plt.xlabel(r"Iteration number $k$")
	plt.tight_layout()
	plt.show()

	x_uniform, k0 = conjugate_gradient( P3_residual , x , A_uniform , b , error_analysis = x_uniform[0] )
	uniform_analysis = []
	uniform_K = max( eig_uniform ) / min ( eig_uniform )
	uniform_bound_const = ( np.sqrt( uniform_K ) - 1 ) / ( np.sqrt( uniform_K ) + 1 )
	initial_uniform_norm = np.dot( np.dot( (x - x_uniform[0] ).T, A_uniform) , (x - x_uniform[0] ) ).flatten()

	for k in range( len ( x_uniform[-1] ) ):
		uniform_analysis.append( 2 * ( uniform_bound_const ** k ) * initial_uniform_norm )

	# plt.subplot(211)
	plt.plot( np.arange(k0) , np.log10( np.asarray(uniform_analysis) ), color = 'firebrick', label = "Theoretical Error Bound")
	plt.plot( np.arange(k0) , np.log10( np.asarray(x_uniform[-1]).flatten() ), color = 'cornflowerblue', label = "Obtained Error")
	plt.legend()
	plt.grid()
	plt.title("Linear Conjugate Gradient Analysis: Uniformly Distributed Eigenvalues")
	plt.ylabel(r"$\log_{10}( \vert \vert x_k -x^{*} \vert \vert_{A} )$")
	plt.xlabel(r"Iteration number $k$")
	plt.tight_layout()
	plt.show()

	x_clustered, k0 = conjugate_gradient( P3_residual , x , A_clustered, b , error_analysis = x_clustered[0] )
	clustered_analysis = []
	clustered_K = max( eig_clustered ) / min ( eig_clustered)
	clustered_bound_const = ( np.sqrt( clustered_K ) - 1 ) / ( np.sqrt( clustered_K ) + 1 )
	initial_clustered_norm = np.dot( np.dot( (x - x_clustered[0] ).T, A_clustered) , (x - x_clustered[0] ) ).flatten()

	for k in range( len ( x_clustered[-1] ) ):
		clustered_analysis.append( 2 * ( clustered_bound_const ** k ) * initial_clustered_norm )

	# plt.subplot(212)
	plt.plot( np.arange(k0) , np.log10( np.asarray(clustered_analysis) ), color = 'orangered', label = "Theoretical Error Bound")
	plt.plot( np.arange(k0) , np.log10( np.asarray(x_clustered[-1]).flatten() ), color = 'deepskyblue', label = "Obtained Error" )
	plt.legend()
	plt.grid()
	plt.title("Linear Conjugate Gradient Analysis: Clustered Eigenvalues")
	plt.ylabel(r"$\log_{10}( \vert \vert x_k -x^{*} \vert \vert_{A} )$")
	plt.xlabel(r"Iteration number $k$")
	plt.tight_layout()
	plt.show()

def Problem_4():
	"""
	Problem_4:

		Description/requirements:
			Numerically solve the cost function defined by P4_cost using 
			nonlinear conjugate gradient algorithms. The algorithms to be 
			tested are FR_CG and FR_CG (with restart).

			Compare the performance of each of these methods.
	"""

	DIM_P4 = 100

	x = np.random.rand( DIM_P4 , 1)
	fr, k0 = FR_CG( P4_cost, P4_grad, x, mode = '0')
	fr_restart, k1 = FR_CG( P4_cost, P4_grad, x, mode = '1')
	pr, k2 = FR_CG( P4_cost, P4_grad, x, mode = '2')
	print( '\n"''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''"')
	print( 'FR_CG ran for {} iterations and ended with {} residual.'.format(k0, P4_cost(fr[0])))
	print( 'FR_Restart ran for {} iterations and ended with {} residual.'.format(k1, P4_cost(fr_restart[0])))
	print( 'PR_CG ran for {} iterations and ended with {} residual.'.format(k2, P4_cost(pr[0])))
	print( '"''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''"\n')
	k = min(k0, k1, k2)
	kiter0 = np.arange(k0)
	kiter1 = np.arange(k1)
	kiter2 = np.arange(k2)
	plt.plot(kiter0, np.log10(fr[2][:k0]), label = 'FR')
	plt.plot(kiter1, np.log10(fr_restart[2][:k1]), label = 'FR_Restart')
	plt.plot(kiter2, np.log10(pr[2][:k2]), label = 'PR')
	plt.legend()
	plt.grid()
	plt.title("Convergence Analysis of CG methods")
	plt.ylabel(r"$\log_{10}( \vert \vert \nabla f \vert \vert )$")
	plt.xlabel(r"Iteration number $k$")
	plt.show()


if __name__ == "__main__":
	Problem_3()
	Problem_4()













