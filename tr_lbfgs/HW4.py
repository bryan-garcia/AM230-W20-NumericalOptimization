"""
HW4.py

Programmer: Bryan Garcia
AM 230 - Numerical Optimization

Description:
	Homework set 4 code corresponding to Problem 2 and Problem 4.
	That is, implementation of trust-region methods and LBFGS.
"""

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
from numdifftools import Gradient, Hessian
from line_search import LineSearch
from scipy.optimize import minimize, minimize_scalar

def get__matrix( mtype = '0' , n = 1e2, seed = None ):
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

	if seed:
		np.random.seed( seed )
	Q_matrix = np.linalg.qr( np.random.rand( n , n ) )[0]
	Diag_eig_matrix = np.diagflat( eigenvalues )

	return np.dot( np.dot( Q_matrix.T , Diag_eig_matrix) , Q_matrix ), eigenvalues

class P2_Analysis:

	def __init__( self, Q, dim = 3,method = 'trcp', eps = 1e-6 ):
		self.method = method
		self.dim = dim
		self.eps = eps
		self.Q = Q
		self.norm_reduction = []
		self.hessian = Hessian( self.cost )
		self.msolver = self.get_solver( method )

	def get_solver( self, method ):
		solver_lookup = { 'trcp' : self.get_cauchy, \
						  'dogleg' : self.dogleg }

		if method not in solver_lookup.keys():
			method = 'trcp'
		return solver_lookup[ method ]

	def set_solver( self, method ):
		self.msolver = self.get_solver( method )
		self.method = method

	def cost( self, xk, nfev_count = False ):
		if nfev_count:
			self.nfev += 1
		return np.log( 1 + np.dot( xk.T, np.dot( self.Q, xk ) ) )

	def gradient( self, xk ):
		return 1 / ( 1 + np.dot( xk.T, np.dot( self.Q, xk ) ) ) \
			* ( 2 * np.dot( self.Q, xk ) )

	def tessian( self, xk ):
		return 2 * ( self.Q @ xk ) * ( -( 1 + xk.T @ self.Q @ xk ) ** ( -2 ) \
					* ( 2 * xk.T @ self.Q ) ) + 2 * self.Q * ( 1 + xk.T @ self.Q @ xk ) ** -1

	def model( self, xk, pk, Bk ):
		return self.cost( xk ) + np.dot( self.gradient( xk ).T, pk ) \
			+ 1 / 2 * np.dot( pk.T, np.dot( Bk, pk ) )

	def get_rho( self, xk, pk, Bk, nfev = False):
		actual_reduction = self.cost( xk, nfev ) - self.cost( xk + pk, nfev )
		predicted_reduction = self.model( xk, np.zeros( pk.shape ), Bk ) \
							- self.model( xk, pk, Bk )
		return actual_reduction / predicted_reduction

	def get_tau( self, pk, Bk, dk ):
		condition_value = np.dot( pk.T, np.dot( Bk, pk ) ).squeeze()
		if condition_value > 0.0:
			tau_k = min ( ( np.linalg.norm( pk ) ** 2 ) \
						/ ( condition_value ), dk / np.linalg.norm( pk ) )
		else:
			tau_k = dk / np.linalg.norm( pk )	
		return tau_k
	
	def get_cauchy( self, xk, pk, Bk, dk ):
		# pk_search = - ( dk / np.linalg.norm( pk ) ) * pk
		pk_cauchy = - self.get_tau( pk, Bk, dk ) * pk # pk_search
		return pk_cauchy

	def dogleg( self, xk, pk, Bk, dk ):
		condition_value = pk.T @ Bk @ pk

		if condition_value <= 0.0:
			pk_dogleg = - dk / np.linalg.norm( pk ) * pk
		else:
			pb = -np.linalg.solve( Bk, pk )
			pu = -( pk.T @ pk ) / ( pk.T @ ( Bk @ pk ) ) * pk 
			if pu.T @ ( pb - pu ) <= 0.0:
				pk_dogleg = self.get_cauchy( xk, pk, Bk, dk )
			else:
				if np.linalg.norm( pb ) <= dk:
					pk_dogleg = pb
				else:
					if np.linalg.norm( pu ) >= dk:
						pk_dogleg = - dk / np.linalg.norm( pk ) * pk
					else:
						a = np.linalg.norm( pb - pu ) ** 2
						b = 2 * ( pb - pu ).T @ pu
						c = np.linalg.norm( pu ) ** 2 - dk ** 2
						tau = 1 + ( -b + np.sqrt( b ** 2 - 4 * a * c ) ) / ( 2 * a )
						pk_dogleg = pu + ( tau - 1 ) * ( pb - pu )

		return np.copy( pk_dogleg )

	def TR_update( self, xk, pk, dk, rk, \
		nu_low = 0.25, nu_hi = 0.125, rscale = 0.25, dmax = 100 ):
		"""
		TR_update

			Description:
				Helper function to determine trust-region and xk
				values.
		"""
		if rk < nu_low:
			dk *= rscale
		else:
			if rk > 0.75 and np.linalg.norm( pk ) <= dk:
				dk = min( 2 * dk, dmax)
		if rk > nu_hi:
			xk += pk

		return np.copy( xk ), dk

	def TR_search( self, x0, dmax = 1000, nu = 0.15, analysis = False ):
		"""
		TR_search

			Description:
				Main loop of trust-region search for Cauchy-point and
				Dogleg methods.
		"""
		self.nfev = 0
		norm_tracker = []
		xk = np.copy( x0 )
		dk = dmax / 100
		max_iter = x0.shape[0] * 100

		for k in np.arange( max_iter + 1 ):
			pk = self.gradient( xk )
			norm_tracker.append( np.linalg.norm( pk ) )
			if norm_tracker[-1] <= self.eps:
				break
			# Bk = get__matrix( n = self.dim, seed = self.dim )[ 0 ]
			Bk = self.tessian( xk ) 
			pk = self.msolver( xk, pk, Bk, dk )
			rk = self.get_rho( xk, pk, Bk, analysis )

			xk, dk = self.TR_update( xk, pk, dk, rk )

		self.norm_reduction.append( [ norm_tracker, self.method ] )
		return xk, k

	def TR_analysis( self ):
		for run in self.norm_reduction:
			#plt.subplot( plot_number )
			plt.plot( np.arange( len( run[ 0 ] ) ), run[ 0 ], label = "Method: " + run[ 1 ] )
			#plot_number += 1
		plt.yscale( 'log' )
		plt.grid()
		plt.legend()
		plt.title('Trust region analysis: Cauchy-point, Dogleg')
		plt.ylabel('Log-error norm')
		plt.xlabel('Iterations (k)')
		plt.tight_layout()
		plt.show()

	def TR_Benchmark( self, x0 ):
		sol, iters = self.TR_search( x0, analysis = True )
		print('\n' + "=" * 75)
		print( self.msolver )
		print( "\n{} TR-search minimized cost to {}\nIterations: {}"\
			.format(self.method, self.cost( sol ), iters ) )
		print("=" * 75 + '\n')

		self.set_solver( method = 'dogleg' )

		sol, iters = self.TR_search( x0, analysis = True )
		print('\n' + "=" * 75)
		print( self.msolver )
		print( "\n{} TR-search minimized cost to {}\nIterations: {}"\
			.format(self.method, self.cost( sol ), iters ) )
		print("=" * 75 + '\n')

		self.TR_analysis()

class P4_Analysis:

	def __init__( self, dim = 3, eps = 1e-8 ):
		self.dim = dim
		self.eps = eps
	def cost( self, xk, alpha = 100 ):

		N = len( xk.squeeze() )
		x = xk.squeeze()

		c = 0.0
		for i in np.arange(0, N // 2 ):
			c += alpha * ( x[ 2 * (i) - 1 ] ** 2 - x[ 2 * (i)] ) ** 2\
				+ ( x[ 2 * (i) - 1 ] - 1 ) ** 2
		return c

	def gradient( self, xk, alpha = 100 ):
		
		g = np.zeros( xk.shape ).flatten()
		x = xk.squeeze()
		N = len( x )
		for i in np.arange(0, N // 2):
			g[ 2*i ] = 2 * alpha * ( x[ 2 * i ] - x[ 2 * i - 1] ** 2 )
			g[ 2*i - 1 ] = - 4 * alpha * x[ 2*i - 1] * ( x[ 2 * i ] \
				- x[ 2 * i - 1 ] ** 2 ) - 2 * ( 1 - x[ 2 *i-1])
		g.shape = xk.shape

		# g = self.gtest( xk )
		# g = np.reshape( g, xk.shape )
		return g

	def optimize( self, x0, m = 3 ):

		def LBFGS_update(xk, S, Y, B0, k, m ):
			
			q = self.gradient( xk )

			if m >= k + 1:
				lower_iter = 0
			else:
				lower_iter = k - m


			alpha = []

			for i in np.arange( len( S ) )[::-1]:
				muk = 1 / ( Y [ i ].T @ S[ i ] )
				alpha.append( muk * S[ i ].T @ q)
				try:
					q = q - alpha[ len(S) - i  - 1 ] * Y[ i ]
				except:
					import pdb
					pdb.set_trace()
					print( 'debug')
			z = B0 @ q

			for i in np.arange( len( S ) ):
				muk = 1 / ( Y [ i ].T @ S[ i ] )
				beta = muk * Y[ i ].T @ z
				z = z + S[ i ] @ ( alpha[ len(S) - i  - 1 ] - beta )

			return -z

		max_iter = x0.shape[0] * 100
		
		xk = np.copy( x0 )

		sk_storage = []
		yk_storage = []

		norm_tracker = []

		iteration_count = 0

		for k in np.arange( max_iter ):

			pk = self.gradient( xk )

			print( np.linalg.norm( pk ) )
			norm_tracker.append( np.linalg.norm( pk) )
			if norm_tracker[-1] <= self.eps:
				break

			H0 = np.eye( len( xk.squeeze() ) )
			# if k > 0:
			# 	temp_k = 1*k
			# 	k = k % (m-1)
			# 	H0 *= ( sk_storage[ k - 1 ].T @ yk_storage[ k - 1] )\
			# 		/ ( yk_storage[ k - 1 ].T @ yk_storage[ k - 1] )
			# 	k = 1*temp_k

			pk = LBFGS_update( xk, sk_storage, yk_storage, H0, k, m )

			xk2 = xk + LineSearch( self.cost, self.gradient, 1000)\
				.step_length( pk, xk ) * pk

			if len( sk_storage ) == m:
				sk_storage.pop( 0 )
				yk_storage.pop( 0 )

			sk = xk2 - xk
			yk = self.gradient( xk2 ) - self.gradient( xk )
			sk_storage.append( sk )
			yk_storage.append( yk )

			xk = xk2
			iteration_count += 1

		self.m = m 
		self.norm_reduction = [ norm_tracker ]
		self.iterations = iteration_count
		return xk

	def LBFGS_analysis( self ):
		for run in self.norm_reduction:
			#plt.subplot( plot_number )
			plt.plot( np.arange( len( run ) ), run, label = "m = {}".format( self.m ) )
			#plot_number += 1
		plt.yscale( 'log' )
		plt.grid()
		plt.legend()
		plt.title('L-BFGS Analysis with dimension-size = {}'.format( self.dim))
		plt.ylabel('Log-error norm')
		plt.xlabel('Iterations (k) ( Total: {} )'.format( self.iterations ) )
		plt.tight_layout()
		plt.show()


def Problem_2():
	P2_dim = 100
	np.random.seed( P2_dim )
	x0 = np.random.rand( P2_dim, 1 )
	Q = get__matrix( n = P2_dim )[ 0 ]
	P2 = P2_Analysis(Q, dim = P2_dim)
	P2.TR_Benchmark( x0 )

def Problem_4():
	P4_dim = 1000
	maxcor = 5
	# x0 = -np.random.rand( P4_dim, 1 )
	x0 = -np.eye( P4_dim, 1 )
	P4 = P4_Analysis( dim = P4_dim )

	P4.optimize( x0, m = maxcor )
	P4.LBFGS_analysis()

	_options = { 'disp' : 1, 'maxcor' : maxcor, 'iprint' : 1}
	minimize( P4.cost, x0, jac = P4.gradient, method = 'L-BFGS-B', options = _options )

if __name__ == "__main__":
	# Problem_2()
	Problem_4()

























































































