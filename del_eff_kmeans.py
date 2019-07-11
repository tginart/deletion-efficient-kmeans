import numpy as np
import time

class Kmeans(object):
    '''
    In-house implementation of k-means via Lloyd-Max iterations
    See 'run' routine for user-facing method call 
    '''
    def __init__(self, k, iters=10, tol=-1):
        '''
        Constructor
        INPUT:
        k - # of centroids/clusters
        iters - # of iterations to run
        '''
        self.k = k
        self.iters = iters
        self.loss = np.Infinity
        self.tol = tol
        self.empty_clusters = []
        self.kpp_inits = set()
        self.n = None
        self.d = None
        self.data = None
        self.centroids = None
        self.assignments = None
        self.model = None

    @staticmethod
    def run(X, k, iters=10, scale=True):
        '''
        User facing routine that runs k-means without exposing class internals
        INPUT:
        X - numpy matrix, n-by-d, each row is a data point
        k - # of clusters/centroids
        iters - # of Lloyd-Max iterations
        OUTPUT: (3-tuple)
        centroids - k-by-d matrix of centroids
        assignments - Vector of length n, with data point to centroid assignments
        loss - The loss of the final partition
        '''
        kmeans = Kmeans(k, iters=iters)
        kmeans.set_data(X, minmax_scaling=scale)
        kmeans.lloyd_iterations()
        return kmeans.centroids, kmeans.assignments, kmeans.loss

    def lloyd_iterations(self):
        self._init_centroids()
        for _ in range(self.iters):
            loss_prev = self.loss
            self._assign_clusters()
            self._assign_centroids()
            if self.tol > 0 and (1 - self.loss/loss_prev) < self.tol:
                break
        
    def set_data(self, X, minmax_scaling=True):
        '''
        Sets data to be clustered
        X - numpy matrix, n-by-d, each row is a data point
        minmax_scaling - bool flag for applying scaling to data
        '''
        self.data = Kmeans._minmax_scale(X) if minmax_scaling else X
        self.n, self.d = X.shape

    def _init_centroids(self):
        '''
        Kmeans++ initialization
        Returns vector of initial centroids
        '''
        first_idx = np.random.choice(self.n)
        self.centroids = self.data[first_idx,:]
        #self.kpp = set()
        for kk in range(1,self.k):
            P = self._get_selection_prob()
            nxt_idx = np.random.choice(self.n,p=P)
            #print(nxt_idx)
            self.kpp_inits.add(nxt_idx)
            #print(self.kpp_inits)
            self.centroids = np.vstack([self.centroids,self.data[nxt_idx,:]])

    def _get_selection_prob(self):
        '''
        in:
        X is a data matrix
        C is a matrix of centroids, i-th row is a centroid for the i-th assignment
        out:
        P is  vector of selection probabilites, equal to Distance^2 to nearest centroid
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

    def _assign_clusters(self):
        '''
        Used in kpp init routine
        in:
        X is a data matrix, each row is a data point
        C is a matrix of centroids, i-th row is a centroid for the i-th assignment
        out:
        A is an assignmnets vector, i-th entry is the cluster assignment of i-th data point
        '''
        assert (self.k, self.d) == self.centroids.shape, "Centroids matrix has incorrect shape"
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
        
        
    def _assign_centroids(self):
        '''
        in:
        X is a data matrix, each row is a data point
        A is an assignments vector, i-th entry is the cluster assignment of i-th data point    
        out:
        C is a matrix of centroids, i-th row is a centroid for the i-th assignment
        '''
        self.centroids = np.zeros([self.k,self.d])
        c = np.zeros([self.k])
        for i in range(self.n):
            a = self.assignments[i]
            c[a] += 1
            self.centroids[a,:] += self.data[i,:]
            
        for j in range(self.k):
            #print(f"clustersize for j {c[j]}")
            self.centroids[j,:] = self.centroids[j,:] / c[j]

        for j in self.empty_clusters: 
            self._reinit_cluster(j)
        self.empty_clusters = []
        
        #print(self.centroids)
            
    @staticmethod
    def _minmax_scale(X, tol=1e-20):
        numerator = X - np.min(X,axis=0)
        denominator = np.max(X,axis=0) - np.min(X,axis=0) + tol
        return numerator / denominator

class QKmeans(Kmeans):
    def __init__(self, k, eps, iters, L, rephase, gamma, adaptive=False):
        '''
        Constructor for quantized k-means solved via Lloyd-Max iterations
        k - # of centroids/clusters
        eps - epsilon parameter in quantizing epsilon net
        iters - # of iterations to run
        L - radial bound of data
        '''
        Kmeans.__init__(self, k, iters=iters)
        self.eps = eps
        self.L = L
        self.loss = 0
        self.minloss = np.Infinity
        self.gamma = gamma
        self.rephase = rephase
        self.momentum = None
        self.model_assignment = None
        self.model = None
        self.analog_c_record = None
        self.c_record = None
        self.phase_record = None
        self.clustersizes_record = None
        self.max_vec = None
        self.min_vec = None
        self.adaptive = False

    @staticmethod
    def run_quantized(X, k, eps, 
                        iters=10, 
                        L=1, 
                        scale=True, 
                        gamma=0.8, 
                        rephase=True):
        '''
        User facing routine that runs k-means without exposing class internals
        INPUT:
        X - numpy matrix, n-by-d, each row is a data point
        k - # of clusters/centroids
        iters - # of Lloyd-Max iterations
        OUTPUT: (4-tuple)
        centroids - k-by-d matrix of centroids
        assignments - Vector of length n, with data point to centroid assignments
        loss - The loss of the final partition
        qkmeans - the created qkmeans object instance 
        '''
        qkmeans = QKmeans(k, eps, iters, L, gamma, rephase)
        qkmeans.set_data(X, minmax_scaling=scale)
        qkmeans.set_metadata()
        qkmeans.quant_lloyd_iterations()
        #qkmeans.model = qkmeans.centroids
        return qkmeans.model, qkmeans.model_assignment, qkmeans.minloss

    def metadata(self):
        '''
        Returns the metadata
        '''
        return self.c_record, self.phase_record

    def train_model(self, X, scale=True):
        '''
        Takes an initialized model and runs it on dataset X
        '''
        self.set_data(X, minmax_scaling=scale)
        self.set_metadata()
        self.quant_lloyd_iterations()
        return self.model, self.model_assignment, self.minloss

    def delete(self, del_idx, remove_from_db=True, verbose=False):
        '''
        Deletion op
        INPUT:
        del_idx - int index of row of data matrix to delete
        OUTPUT: 
        Succesful - Bool flag if deletion succesful 
        Row - returns deleted point as numpy row
        '''
        pt2del = self.data[del_idx,:]
        
        #print(del_idx)
        #print(self.kpp_inits)
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
            quant = self._quantize(centroid, self.eps, phase)
            quant_perturbed = self._quantize(perturbed_centroid, self.eps, phase)
            
            if not all(quant == quant_perturbed):
                return False
            
            if remove_from_db:
                self.clustersizes_record[i-1,assignment_idx] -= 1
                self.analog_c_meta[i-1,assignment_idx] = perturbed_centroid
                
        if remove_from_db:
            self.data[del_idx,:] = np.zeros(self.d)
            self.n -= 1
            self.momentum = self.gamma * self.n / self.k

        return True

    def _phase_shift(self):
        self.phase = np.random.random([self.d])

    def set_metadata(self):
        self.analog_c_meta = np.zeros( [self.iters+1, self.k, self.d])
        self.q_c_meta = np.zeros( [self.iters+1, self.k, self.d])
        self.phase_record = np.zeros([self.iters, self.d])
        self.clustersizes_record = np.zeros([self.iters, self.k])
        self.early_term = -1
        self.momentum = self.gamma * self.n / self.k


    def quant_lloyd_iterations(self):
        self._init_centroids()
        self.analog_c_meta[0] = self.centroids
        #no need to quantize initial centroids
        self.q_c_meta[0] = self.centroids 
        self._assign_clusters()
        self._phase_shift()
        for i in range(self.iters):
            self._iterate(i)
            if self.minloss > self.loss and self.early_term < 0:
                #print(
                self._save_model()
            else:
                self.early_term = i
                #print(self.early_term)
                break
                    
    
    def _save_model(self):
        #print('save')
        self.minloss = self.loss
        self.model = self.centroids
        self.model_assignment = self.assignments

    def _iterate(self,i):
        #self._stabilize_clusters()
        self._assign_centroids()
        self.quantize(i)
        self._assign_clusters()
   

    def quantize(self,i):
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
            #print(f"clustersize for {j} is {clustersizes[j]}")
            if (clustersizes[j]  < self.momentum):
                self.centroids[j] = self._momentum_correction(self.centroids[j], self.q_c_meta[i,j],
                                                              clustersizes[j])
            
        #print(f"clustersize record {self.clustersizes_record}")
        
        #quantize centroids
        if self.rephase:
            self._phase_shift()
        self.centroids = self._quantize(self.centroids, self.eps, self.phase)

        #record random phase and quantized centroids
        self.phase_record[i,:] = self.phase
        self.q_c_meta[i+1] = self.centroids


    def _momentum_correction(self,centroid_cur, centroid_prev, clustersize):
        lag = (self.momentum-clustersize)/self.momentum
        lagged_centroid = (clustersize/self.momentum)*centroid_cur
        lagged_centroid += lag*centroid_prev
        return lagged_centroid


    def _quantize(self, centroids, eps, phase):
        centroids = centroids * 1/eps        
        centroids = centroids + (phase-0.5)
        centroids = np.round(centroids)
        centroids = centroids - (phase-0.5)
        centroids = centroids * eps
        return centroids
                    
class DCnode(Kmeans):
    def __init__(self, k, iters):
        Kmeans.__init__(self, k, iters=iters)
        self.children = []
        self.parent = None
        self.time = 0
        self.loss = 0
        self.node_data = set()
        self.data_prop = set()

    def run_node(self, X):
        self.set_node_data(X)
        self.lloyd_iterations()

    def set_node_data(self, X):
        self.data = X[list(self.node_data)]
        isLeaf = False#(self.children == [])
        self.set_data(self.data,minmax_scaling=isLeaf)

class DCKmeans():
    def __init__(self, ks, widths, iters):
        self.ks = ks
        self.widths = widths
        self.dc_tree = self._init_tree(ks,widths,iters)
        self.data_partition_table = dict()
        self.data = dict()
        self.dels = set()
        self.d = 0
        self.n = 0
        self.h = len(self.dc_tree)
        for i in range(self.h):
            self.data[i] = None
            
    def run(self,X):
        self.init_data(X)
        self.partition_data(X)
        self._run()
        return self.centroids

    def init_data(self,X):
        self.n = len(X)
        self.d = len(X[0])
        data_layer_size = self.n
        for i in range(self.h-1,-1,-1):
            self.data[i] = np.zeros((data_layer_size,self.d))
            data_layer_size = self.ks[i]*self.widths[i] 
        
    def partition_data(self, X):
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
                subproblem.run_node(self.data[l])
                if subproblem.parent == None:
                    self.centroids = subproblem.centroids
                else:
                    for c_id in range(len(subproblem.centroids)):
                        subproblem.data_prop.add(c)
                        subproblem.parent.node_data.add(c)
                        self.data[l-1][c] = subproblem.centroids[c_id] 
                        c += 1
                        

    def delete(self, idx):
        assert not idx in self.dels, "trying to redel an idx"
        self.dels.add(idx)
        node = self.dc_tree[-1][self.data_partition_table[idx]]
        node.node_data.remove(idx)
        l = self.h-1
        while True:
            node.run_node(self.data[l])
            if node.parent == None:
                self.centroids = node.centroids
                break
            data_prop = list(node.data_prop)
            for c_id in range(len(node.centroids)):
                idx = data_prop[c_id]
                self.data[l][idx] = node.centroids[c_id]
            node = node.parent
            l -= 1
            

    def _init_tree(self, ks, widths,iters):
        tree = [[DCnode(ks[0],iters)]] # root note
        for i in range(1,len(widths)):
            k = ks[i]
            assert widths[i] % widths[i-1] == 0, "Inconsistent widths in hierarchy"
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
