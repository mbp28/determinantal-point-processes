class seq_model.py(object):

	def __init__(self, seq_length, n_labels, A, Q, Q1, G):
		"""
		A = N x N sparse edge potential matrix; A(i,j) is potential from label i to j
		Q = 1 x N node potential vector (applies to all except (possibly) first node)
		Q1 = 1 x N initial node quality vector (if omitted, uses Q)
		G = N x D similarity features (only needed for 2nd-order modes)

		"""

		self.seq_length = seq_length # Alex calls it T
		self.n_labels = n_labels # Alex calls it N
		self.A = # 
		self.Q = np.ones(N)
		self.Q1 = np.exp(-1000*((np.arange(100) / N) - 0.5)**2)
		self.G = 

