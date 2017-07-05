class seq_model.py(object):

	def __init__(self, seq_length, n_labels, sim_dim, k):
		"""
		A = N x N sparse edge potential matrix; A(i,j) is potential from label i to j
		Q = 1 x N node potential vector (applies to all except (possibly) first node)
		Q1 = 1 x N initial node quality vector (if omitted, uses Q)
		G = N x D similarity features (only needed for 2nd-order modes)

		"""

		self.seq_length = seq_length # Alex calls it T
		self.n_labels = n_labels # Alex calls it np
		self.sim_dim = sim_dim
		self.k = k

		temp = np.tile(np.arange(seq_length) / seq_length, (seq_length,1))
		self.A = np.exp(-1e4 * (temp - temp.T) **2)
		self.Q = np.ones(N)
		self.Q1 = np.exp(-1000*((np.arange(self.seq_length) / self.seq_length) - 0.5) **2)
		temp = np.tile(np.arange(seq_length) / seq_length, (sim_dim, 1))
		temp2 = np.tile(np.arange(sim_dim) / sim_dim, (seq_length, 1))
		self.G = np.exp(-1(temp.T - temp2) ** 2)

		

