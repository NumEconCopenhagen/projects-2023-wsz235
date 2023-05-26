from types import SimpleNamespace
import warnings 
import numpy as np
from scipy import optimize
import pandas as pd 
import matplotlib.pyplot as plt

class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        if np.isclose(par.sigma, 0.0):
            H = np.minimum(HF, HM)
        elif np.isclose(par.sigma, 1.0):
            H = HM**(1-par.alpha)*HF**par.alpha
        else:
            H = ((1-par.alpha)*HM**((par.sigma-1)/par.sigma) + par.alpha * HF**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1))

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
    

        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]
        opt.HFHM_Ratio = HF[j]/HM[j]
    

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt
    
    def solve_continuous(self,do_print=False):
        """ solve model continuously """
        
        # setting up initial parameters
        par = self.par
        sol = self.par
        opt = SimpleNamespace()

        # creating objective function for optimize
        def obj(x):
            return -self.calc_utility(x[0],x[1],x[2],x[3])
        
        # setting up constraints, 24 hour max for H and L
        constraints = { 'type': 'ineq', 'fun': lambda x: 24 - x[0] - x[1],
                        'type': 'ineq', 'fun': lambda x: 24 - x[2] - x[3]}

        # setting bounds, 24 hours for each 
        bounds = ((0,24),(0,24),(0,24),(0,24))

        # setting initial guess
        initial_guess = [12, 12, 12, 12]

        #Optimizing
        solution = optimize.minimize(obj, initial_guess, method='SLSQP',bounds = bounds, constraints = constraints,tol=1e-10)
        
        # saving results
        opt.LM = solution.x[0]
        opt.HM = solution.x[1]
        opt.LF = solution.x[2]
        opt.HF = solution.x[3]

        if do_print:
            print(solution.message)
            print(f'LM: {opt.LM:.4f}')
            print(f'HM: {opt.HM:.4f}')
            print(f'LF: {opt.LF:.4f}')
            print(f'HF: {opt.HF:.4f}')

        return opt

    def solve_wF_vec(self,discrete=False):
        """ solve model for vector of female wages """

        par = self.par
        sol = self.sol

        #loop over the vector of female wage and change the value of wF to whereever we are in the vector 
        for i, wF in enumerate(par.wF_vec):
            par.wF = wF

            #solve the model with the discrete solver if keyword argument above is true
            if discrete == True:
                opt = self.solve_discrete()
            
            #use the contiuous solver if keyword argument above is false 
            elif discrete == False:                
                opt = self.solve_continuous()

            #store the resulting values 
            sol.LM_vec[i] = opt.LM
            sol.HM_vec[i] = opt.HM
            sol.LF_vec[i] = opt.LF
            sol.HF_vec[i] = opt.HF

        return sol.LM_vec, sol.HM_vec, sol.LF_vec, sol.HF_vec

    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
    
    def estimate(self,alpha=None,sigma=None):
        """ estimate alpha and sigma """

        par = self.par
        sol = self.sol

        # define an objective function to be minimized
        def objective_function(x):

            #initial parameters
            par.alpha = x[0]
            par.sigma = x[1]

            # solve optimal choice set, account for different wF
            self.solve_wF_vec()

            # run regression for beta_0 and beta_1
            self.run_regression()

            return (par.beta0_target - sol.beta0)**2 + (par.beta1_target - sol.beta1)**2

        # create bounds and initial guess
        bounds = [(0, 1), (0, 5)]
        initial_guess = [0.5, 1]

        # call the solver to find the optimal alpha and sigma here
        solution = optimize.minimize(objective_function, initial_guess, method='Nelder-Mead', bounds=bounds)

        # create a dictionary to store the results
        sol.alpha = solution.x[0]
        sol.sigma = solution.x[1]

        return sol
        