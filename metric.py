import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import cloudpickle
import copy

class Metric():
    """ This class stores the metric (tensor) as a SymPy tensor. Other quantities can
        be derived from the metric tensor, like the Christoffel symbols, Riemann tensor,
        Ricci tensor, Ricci scalar, Einstein tensor, geodesic equations, killing vectors
        and spacetime interval representation.
        
        The metric tensor requires first to input the variables as strings (it is preferred 
        not to represent variables as 'theta' but rather as 'θ'.) and from there input the 
        metric tensor components as SymPy expressions. Once that is done, quantities can be
        derived from the metric by running the 'init_' modules (e.g. metric.init_chr initializes
        and stores the Christoffel symbols in metric.chr.). 

        One can also access common metrics such as:
            - schwarzschild metric,
            - frw (Friedmann-Robertson-Walker) metric,
            - [...]

        which can be accessed by 
            >>> metric = Metric().schwartzschild()
    """
    def __init__(self, variables, name=None):
        self.variables = sp.symbols(variables)  # Variables representing each coordinate
        self.dim = len(variables)          # Dimension of the metric
        self.name = name              # Name of the metric
        self.tensor = sp.Matrix(np.zeros((self.dim,self.dim))) # Tensor representation
        self.inv = None     # Metric (tensor) inverse
        self.chr = None     # Christoffel symbols
        self.geo = None     # Geodesic equations
        self.riemann = None # Riemann tensor
        self.ricci = None   # Ricci tensor
        self.r = None       # Ricci scalar
        self.einstein_tensor = None # Einstein tensor

    def st_interval(self):
        """ Metric as spacetime interval equation (ds^2 = g_{uv}dx^u dx^v). """
        dvariables = self.variables # Infinitesimal variables
        for idx, variable in enumerate(self.variables):
            dvariables[idx] = sp.symbols('d'+ str(variable))
        ds = 0
        for i, dx in enumerate(dvariables):
            for j, dy in enumerate(dvariables):
                ds += self.tensor[i,j]*dx*dy
        return sp.Eq(sp.symbols('ds')**2, ds)

    def init_chr(self):
        """ Christoffel symbols of the metric. The first index is the upper index. """
        self.inv = self.tensor.inv() # Metric inverse
        chr = sp.MutableDenseNDimArray(np.zeros((self.dim,)*3)) # Initializing symbols
        dg = sp.MutableDenseNDimArray(np.zeros((self.dim,)*3)) # derivative of metric w.r.t. variables
        for mu in range(self.dim):
            dg[:,:,mu] = sp.diff(self.tensor, self.variables[mu])
        for nu in range(self.dim):
            chr[:,:,nu] = 1/2*( self.inv*dg[:,:,nu] + self.inv*dg[:,nu,:] - self.inv*(sp.Matrix(dg[:,nu,:]).transpose()))
        self.chr = sp.simplify(chr) # store christoffel symbols in object
        
    def init_geo(self):
        """ Geodesic equations of the metric. """
        if isinstance(self.chr, type(None)): # Initialize Christoffel symbols (if not already done)
            self.init_chr()
        variables = sp.zeros(self.dim,1)
        parameter = sp.symbols('𝛕') # Parameter labelling each point on the world-line
        for idx in range(self.dim):
            variables[idx] = sp.Function(self.variables[idx])(parameter) # making each variable a function of parameter
        geodesics = sp.zeros(self.dim,1) # Store geodesic expressions in sympy vector
        for eq in range(self.dim): # geodesic equations
            for i, x in enumerate(variables):
                for j, y in enumerate(variables):
                    geodesics[eq] += self.chr[eq,i,j]*sp.diff(x, parameter)*sp.diff(y, parameter)
        self.geo = sp.simplify(sp.Eq(sp.diff(variables, parameter, 2) + geodesics, sp.zeros(self.dim,1)))

    def init_riemann(self):
        """ Riemann tensor of the metric, which is a 4-index tensor. """
        riemann = sp.MutableDenseNDimArray(np.zeros((self.dim,)*4)) # Inizializing 4-index tensor
        dchr = sp.MutableDenseNDimArray(np.zeros((self.dim,)*4)) # Derivative of Christoffel symbols
        if isinstance(self.chr, type(None)):
            self.init_chr() # Initialize Christoffel symbols (if not already done)
        for mu in range(self.dim):
            dchr[:,:,:,mu] = sp.diff(self.chr, self.variables[mu])
        for sigma in range(self.dim):
            for rho in range(self.dim):
                riemann[rho,sigma,:,:] = dchr[rho,:,sigma,:].transpose() - dchr[rho,:,sigma,:] \
                        + sp.tensorcontraction(sp.tensorproduct(self.chr[rho,:,:], self.chr[:,:,sigma]),(1,2)) \
                        - (sp.tensorcontraction(sp.tensorproduct(self.chr[rho,:,:], self.chr[:,:,sigma]),(1,2))).transpose()
        self.riemann = sp.simplify(riemann)

    def init_ricci(self):
        """ Ricci tensor of the metric, which is a (dim x dim) tensor. """
        self.ricci = sp.MutableDenseNDimArray(np.zeros((self.dim,)*2))
        if isinstance(self.riemann, type(None)):
            self.init_riemann() # Initialize Riemann tensor (if not already done)
        for mu in range(self.dim):
            self.ricci += self.riemann[mu,:,mu,:] # Contracting first (upper) and third (lower) indices
        self.ricci = sp.Matrix(sp.simplify(self.ricci))

    def init_r(self):
        """ Ricci scalar of the metric. """
        if isinstance(self.ricci, type(None)):
            self.init_ricci() # Initialize Ricci tensor (if not already done)
        self.r = sp.trace(sp.simplify(sp.Matrix(self.ricci)*self.inv))

    def init_einstein(self):
        """ Einstein tensor of the metric, which is a (dim x dim) tensor. """
        if isinstance(self.r, type(None)):
            self.init_r() # Initialize Ricci scalar (if not already done)
        self.einstein_tensor = sp.simplify(self.ricci - 1/2*self.r*self.tensor) 
    
    def killing_vec(self):
        """ Check for (trivial) killing vectors. Each killing vector is stored as a column vector associated with variable order.
            Note: a column vector full of zeros is not a killing vector."""
        k = sp.zeros(self.dim) # storing each killing vector as a column vector (order associated with variable order)
        for idx, variable in enumerate(self.variables):
            new_metric = self.tensor.subs(variable, variable + sp.symbols('constant')) # making the change of variables
            if new_metric == self.tensor: # if the metric does not depend on the variable
                k[idx,idx] = 1
        return k

    def isvacuum(self):
        """ Does the metric satisfy Einstein's vacuum equation: R_uv = 0 for all u,v? """
        if isinstance(self.ricci, type(None)):
            self.init_ricci()
        return self.ricci == sp.zeros(self.dim)

    def __eq__(self, other):
        """ Two metric are equivalent if they have the same tensor representation. """
        if not isinstance(other, Metric):
            # don't attempt to compare against unrelated types
            raise TypeError("Cannot compare unrelated types.")
        return self.tensor == other.tensor

    @staticmethod
    def schwarzschild():
        """ Unpickling the Schwarzschild metric stored in 'schwartzschild.txt'. """
        with open('common_metrics/schwartzschild.txt', 'rb') as f:
            sch_metric = cloudpickle.load(f)
        return sch_metric

    @staticmethod
    def frw():
        """ Unpickling Friedmann-Robertson-Walker metric stored in 'frw.txt'. """
        with open('common_metrics/frw.txt', 'rb') as f:
            frw_metric = cloudpickle.load(f)
        return frw_metric

def save_metric(metric, name):
    """ Pickles a Metric instance and stores it in name.txt."""
    if isinstance(metric, Metric):
        with open("common_metrics/" + str(name)+'.txt', 'wb') as f:
            cloudpickle.dump(metric, f)
    else:
        raise TypeError("Invalid data type, must be a 'Metric' instance.")