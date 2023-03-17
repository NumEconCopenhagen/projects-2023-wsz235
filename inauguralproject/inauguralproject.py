


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

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

    def H(sigma, HM, HF):
        if sigma = 0:
            return np.minimum(HM, HF)
        elif sigma = 1:
            return HM**(1-alpha)*HF**alpha
        else:
            return ((1-alpha)*HM**((alpha-1)/alpha) + alpha*HF**((alpha-1)/alpha))**(alpha/(alpha-1))