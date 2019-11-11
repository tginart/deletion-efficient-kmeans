import numpy as np

class Kmeans(object):
    '''
    In-house implementation of k-means via Lloyd-Max iterations
    This is a research prototype and is not necessarily well-optimized
    '''
    def __init__(self,
                    k,
                    termination='fixed',
                    iters=10,
                    tol=10**-3):
        '''
        Constructor
        INPUT:
        k - # of centroids/clusters
        iters - # of iterations to run
        termination - {'fixed', 'loss', 'centers'}
            if 'fixed' - runs for fixed # of iterations
            if 'loss' - runs until loss converges
            if 'centers' -runs until centers converge
        tol - numeric tolerance to determine convergence
        '''
        # set parameters
        self.k = k
        self.iters = iters
        self.tol = tol
        self.termination = termination
        # initialize placeholder values
        self._init_placeholders()

    def run(self, X):
        '''
        Run clustering algorithm
        INPUT:
        X - numpy matrix, n-by-d, each row is a data point
        OUTPUT: (3-tuple)
        centroids - k-by-d matrix of centroids
        assignments - Vector of length n, with datapoint to center assignments
        loss - The loss of the final partition
        '''
        self._set_data(X)
        self._lloyd_iterations()
        return self.centroids, self.assignments, self.loss

    def delete(self, del_idx):
        '''
        Delete point associated with key del_idx
        NOTE: del_idx must be int in {0,n-1}
            After deleting any key other than n-1,
            the (n-1)-th datapoint's key is automatically
            swapped with del_idx to
        '''
        self.data = np.delete(self.data, del_idx, 0)
        self.n = self.n-1
        self._init_placeholders()
        return self.run(self.data)

    def _init_placeholders(self):
        self.loss = np.Infinity
        self.empty_clusters = []
        self.kpp_inits = set()
        self.centroids = None
        self.assignments = None
        self.model = None

    def _set_data(self, X):
        self.data = X
        self.n, self.d = X.shape

    def _lloyd_iterations(self):
        self._init_centroids()
        for _ in range(self.iters):
            loss_prev = self.loss
            centers_prev = self.model
            self._assign_clusters()
            self._assign_centroids()
            prev = loss_prev if self.termination == 'loss' else centers_prev
            if self._check_termination(prev):
                break
            
    def _check_termination(self, prev):
        if self.termination == 'loss':
            return (1 - self.loss/prev) < self.tol
        elif self.termination == 'center':
            return np.linalg.norm(self.centroids - prev) < self.tol
        else:
            return False

    def _init_centroids(self):
        '''
        Kmeans++ initialization
        Returns vector of initial centroids
        '''
        first_idx = np.random.choice(self.n)
        self.centroids = self.data[first_idx,:]
        for kk in range(1,self.k):
            P = self._get_selection_prob()
            nxt_idx = np.random.choice(self.n,p=P)
            self.kpp_inits.add(nxt_idx)
            self.centroids = np.vstack([self.centroids,self.data[nxt_idx,:]])

    def _get_selection_prob(self):
        '''
        Outputs vector of selection probabilites
        Equal to Distance^2 to nearest centroid
        '''
        #handle edge case in centroids shape by unsqueezing
        if len(self.centroids.shape) == 1:
            self.centroids = np.expand_dims(self.centroids, axis=0)

        #probability is square distance to closest centroid
        D = np.zeros([self.n])
        for i in range(self.n):
            d = np.linalg.norm(self.data[i,:] - self.centroids, axis=1)
            D[i] = np.min(d)
        P = [dist**2 for dist in D]
        P = P / sum(P)
        return P  

    def _assign_centroids(self):
        '''
        Computes centroids in Lloyd iterations
        '''
        self.centroids = np.zeros([self.k,self.d])
        c = np.zeros([self.k])
        for i in range(self.n):
            a = self.assignments[i]
            c[a] += 1
            self.centroids[a,:] += self.data[i,:]
            
        for j in range(self.k):
            self.centroids[j,:] = self.centroids[j,:] / c[j]

        for j in self.empty_clusters: 
            self._reinit_cluster(j)
        self.empty_clusters = []
        
    def _assign_clusters(self):
        '''
        Computes clusters in Lloyd iterations
        '''
        assert (self.k, self.d) == self.centroids.shape, "Centers wrong shape"
        self.assignments = np.zeros([self.n]).astype(int)
        self.loss = 0
        for i in range(self.n):
            d = np.linalg.norm(self.data[i,:] - self.centroids, axis=1)
            d1 = np.linalg.norm(self.data[i,:] - self.centroids, axis=1,ord=1)
            self.assignments[i] = int(np.argmin(d))
            self.loss += np.min(d)**2
        self.loss = self.loss / self.n
        self.empty_clusters = self._check_4_empty_clusters()

    def _check_4_empty_clusters(self):
        empty_clusters = []
        for kappa in range(self.k):
            if len(np.where(self.assignments == kappa)[0]) == 0:
                empty_clusters.append(kappa)
        return empty_clusters

    def _reinit_cluster(self, j):
        '''
        Gets a failed centroid with idx j (empty cluster)
        Should replace with new k++ init centroid
        in:
            j is idx for centroid, 0 <= j <= n
        out:
            j_prime is idx for next centroid
        side-effects:
            centroids are update to reflect j -> j'
        '''
        P = self._get_selection_prob()
        j_prime = np.random.choice(self.n,p=P)
        self.kpp_inits.add(j_prime)
        self.centroids[j,:] = self.data[j_prime,:]
        return j_prime

class QKmeans(Kmeans):
    def __init__(self,
                    k,
                    eps,
                    termination='fixed',
                    iters=10, 
                    gamma=0.2,
                    tol=10**-3):
        '''
        Constructor for quantized k-means solved via Lloyd iterations
        k - # of centroids/clusters
        eps - epsilon parameter in quantizing epsilon net
        termination - {'fixed', 'centers'}
        iters - # of iterations to run
        gamma - momentum correct parameter for class imbalance
        tol - numerical convergence tolerance 
        '''
        assert termination != 'loss','Termination should be fixed or centers' 
        Kmeans.__init__(self, k, termination=termination, iters=iters, tol=tol)
        self.eps = eps
        self.gamma = gamma
        
    def run(self, X):
        '''
        X - numpy matrix, n-by-d, each row is a data point
        OUTPUT: (3-tuple)
        centroids - k-by-d matrix of centroids
        assignments - Vector of length n, with datapoint to center assignments
        loss - The loss of the final partition
        '''
        self._set_data(X)
        self._init_placeholders_q()
        self._init_metadata()
        self._quant_lloyd_iterations()
        return self.model, self.model_assignment, self.minloss

    def delete(self, del_idx):
        if not self._certify_invariance(del_idx):
            self._init_placeholders_q()
            return super(QKmeans, self).delete(del_idx)
        else:
            return self.model, self.model_assignment, self.minloss

    def metadata(self):
        '''
        Returns the metadata
        '''
        return self.c_record, self.phase_record

    def _init_metadata(self):
        self.analog_c_meta = np.zeros( [self.iters+1, self.k, self.d])
        self.q_c_meta = np.zeros( [self.iters+1, self.k, self.d])
        self.phase_record = np.zeros([self.iters, self.d])
        self.clustersizes_record = np.zeros([self.iters, self.k])
        self.early_term = -1
        self.momentum = self.gamma * self.n / self.k

    def _init_placeholders_q(self):
        self.minloss = np.Infinity
        self.momentum = None
        self.model_assignment = None
        self.model = None
        self.analog_c_record = None
        self.c_record = None
        self.phase_record = None
        self.clustersizes_record = None

    def _certify_invariance(self, del_idx):
        '''
        Computes a certficate of invariance under deletion for del_idx
        INPUT:
        del_idx - int index of row of data matrix to delete
        OUTPUT: 
        Succesful - Bool flag if deletion succesful
            Updates metadata automatically on success
        '''
        pt2del = self.data[del_idx,:]
        if del_idx in self.kpp_inits:
            return False
        
        for i in range(self.iters):
            if i >= self.early_term and self.early_term >= 0:
                break
                
            analog_centroids  = self.analog_c_meta[i+1,:,:]
            phase = self.phase_record[i,:]
            clustersizes = self.clustersizes_record[i,:]
            d = np.linalg.norm(pt2del - analog_centroids, axis=1)
            assignment_idx = int(np.argmin(d))
            centroid = analog_centroids[assignment_idx,:]
            centroid_prev = self.q_c_meta[i,assignment_idx,:]
            clustersize = clustersizes[assignment_idx]
            
            if clustersize < self.momentum:
                centroid = self._momentum_correction(
                    centroid, centroid_prev, clustersize)
                clustersize = self.momentum
                
            perturbed_centroid = centroid - pt2del/clustersize
            quant = self._Q(centroid, self.eps, phase)
            quant_perturbed = self._Q(perturbed_centroid, self.eps, phase)
            
            if not all(quant == quant_perturbed):
                return False
            
            self.clustersizes_record[i-1,assignment_idx] -= 1
            self.analog_c_meta[i-1,assignment_idx] = perturbed_centroid
                
        self.data[del_idx,:] = np.zeros(self.d)
        self.n -= 1
        self.momentum = self.gamma * self.n / self.k
        return True

    def _quant_lloyd_iterations(self):
        self._init_centroids()
        self.analog_c_meta[0] = self.centroids
        #no need to quantize initial centroids
        self.q_c_meta[0] = self.centroids 
        self._assign_clusters()
        self._phase_shift()
        for i in range(self.iters):
            prev = self.model
            self._iterate(i)
            if self.minloss <= self.loss:
                self.early_term = i
                break
            elif self._check_termination(prev):
                self.early_term = i
                self._save_model()
                break
            else:
                self._save_model()

    def _save_model(self):
        self.minloss = self.loss
        self.model = self.centroids
        self.model_assignment = self.assignments

    def _iterate(self,i):
        self._assign_centroids()
        self._quantize(i)
        self._assign_clusters()

    def _quantize(self,i):
        #record analog clusters
        self.analog_c_meta[i+1,:,:] = self.centroids

        #compute the clustersizes
        clustersizes = {j : 0 for j in range(self.k)}
        for j in range(self.n):
            a = self.assignments[j]
            clustersizes[a] += 1
  
        #record the clustersizes and apply momentum correction
        for j in range(self.k):
            self.clustersizes_record[i,j] =  clustersizes[j]
            if (clustersizes[j]  < self.momentum):
                self.centroids[j] = self._momentum_correction(
                    self.centroids[j], self.q_c_meta[i,j], clustersizes[j])
        #quantize centroids
        self._phase_shift()
        self.centroids = self._Q(self.centroids, self.eps, self.phase)

        #record random phase and quantized centroids
        self.phase_record[i,:] = self.phase
        self.q_c_meta[i+1] = self.centroids

    def _phase_shift(self):
        self.phase = np.random.random([self.d])

    def _momentum_correction(self,centroid_cur, centroid_prev, clustersize):
        lag = (self.momentum-clustersize)/self.momentum
        lagged_centroid = (clustersize/self.momentum)*centroid_cur
        lagged_centroid += lag*centroid_prev
        return lagged_centroid

    def _Q(self, centroids, eps, phase):
        centroids = centroids * 1/eps        
        centroids = centroids + (phase-0.5)
        centroids = np.round(centroids)
        centroids = centroids - (phase-0.5)
        centroids = centroids * eps
        return centroids

class DCnode(Kmeans):
    '''
    A k-means subproblem for the divide-and-conquer tree
    in DC-k-means algorithm
    '''
    def __init__(self, k, iters):
        Kmeans.__init__(self, k, iters=iters)
        self.children = []
        self.parent = None
        self.time = 0
        self.loss = 0
        self.node_data = set()
        self.data_prop = set()

    def _run_node(self, X):
        self._set_node_data(X)
        self._lloyd_iterations()

    def _set_node_data(self, X):
        self.data = X[list(self.node_data)]
        self._set_data(self.data)

class DCKmeans():
    def __init__(self, ks, widths, iters=10):
        '''
        Constructor for quantized k-means solved via Lloyd iterations
        ks - list of k parameter for each layer of DC-tree
        widths - list of width parameter (number of buckets) for each layer
        iters - # of iterations to run 
            (at present, only supports fixed iteration termination)
        '''
        self.ks = ks
        self.widths = widths
        self.dc_tree = self._init_tree(ks,widths,iters)
        self.data_partition_table = dict()
        self.data = dict()
        self.dels = set()
        self.valid_ids = []
        self.d = 0
        self.n = 0
        self.h = len(self.dc_tree)
        for i in range(self.h):
            self.data[i] = None
            
    def run(self, X, assignments=False):
        '''
        X - numpy matrix, n-by-d, each row is a data point
        assignments (optional) - bool flag, computes assignments and loss
            NOTE: Without assignments flag, this only returns the centroids
        OUTPUT:
        centroids - k-by-d matrix of centroids
            IF assignments FLAG IS SET ALSO RETURNS:
        assignments - Vector of length n, with datapoint to center assignments
        loss - The loss of the final partition
        '''
        self._init_data(X)
        self._partition_data(X)
        self._run()
        if assignments:
            assignment_solver = Kmeans(self.ks[0])
            assignment_solver._set_data(X)
            assignment_solver.centroids = self.centroids
            assignment_solver._assign_clusters()
            self.assignments = assignment_solver.assignments
            self.loss = assignment_solver.loss
            return self.centroids, self.assignments, self.loss
        return self.centroids
    
    def delete(self, del_idx):
        idx = self.valid_ids[del_idx]
        self.valid_ids[del_idx] = self.valid_ids.pop()
        self.dels.add(idx)
        node = self.dc_tree[-1][self.data_partition_table[idx]]
        node.node_data.remove(idx)
        l = self.h-1
        self.n -= 1
        while True:
            node._run_node(self.data[l])
            if node.parent == None:
                self.centroids = node.centroids
                break
            data_prop = list(node.data_prop)
            for c_id in range(len(node.centroids)):
                idx = data_prop[c_id]
                self.data[l][idx] = node.centroids[c_id]
            node = node.parent
            l -= 1

    def _init_data(self,X):
        self.n = len(X)
        self.valid_ids = list(range(self.n))
        self.d = len(X[0])
        data_layer_size = self.n
        for i in range(self.h-1,-1,-1):
            self.data[i] = np.zeros((data_layer_size,self.d))
            data_layer_size = self.ks[i]*self.widths[i] 
        
    def _partition_data(self, X):
        self.d = len(X[0])
        num_leaves = len(self.dc_tree[-1])
        for i in range(len(X)):
            leaf_id = np.random.choice(num_leaves)
            leaf = self.dc_tree[-1][leaf_id]
            self.data_partition_table[i] = leaf_id
            leaf.node_data.add(i)
            self.data[self.h-1][i] = X[i]

    def _run(self):
        for l in range(self.h-1,-1,-1):
            c = 0
            for j in range(self.widths[l]):
                subproblem = self.dc_tree[l][j]
                subproblem._run_node(self.data[l])
                if subproblem.parent == None:
                    self.centroids = subproblem.centroids
                else:
                    for c_id in range(len(subproblem.centroids)):
                        subproblem.data_prop.add(c)
                        subproblem.parent.node_data.add(c)
                        self.data[l-1][c] = subproblem.centroids[c_id] 
                        c += 1

    def _init_tree(self, ks, widths, iters):
        tree = [[DCnode(ks[0],iters)]] # root node
        for i in range(1,len(widths)):
            k = ks[i]
            assert widths[i] % widths[i-1] == 0, "Inconsistent widths in tree"
            merge_factor = int(widths[i] / widths[i-1])
            level = []
            for j in range(widths[i-1]):
                parent = tree[i-1][j]
                for _ in range(merge_factor):
                    child = DCnode(k,iters=10)
                    child.parent = parent
                    parent.children.append(child)
                    level.append(child)
            tree.append(level)
        return tree
