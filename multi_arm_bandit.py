import torch
from tqdm import tqdm
from torch.distributions import Beta
class MultiArmBandit:
    def __init__(self, config) -> None:
        # self.seed = config.seed
        # self.iterations = config.iterations
        # self.horizon = config.horizon
        # self.arms = config.arms
        self.__dict__ = config.__dict__
        self.betas = torch.ones((self.arms,2))
        
    def beta_pdf(self, distribution, x):
        return distribution.log_prob(x).exp() # dim (arms, points)
    
    def sample(self, betas):
        distributions = [Beta(beta[0], beta[1]) for beta in betas] # dim = (arms)
        x = torch.linspace(1/self.points,1,self.points)
        f = torch.stack([self.beta_pdf(d,x) for d in distributions], dim=0) # dim (arms, points)
        F = f.cumsum(dim=1)/self.points # dim (arms, points)
        F_bar = torch.prod(F,dim=0) # dim (points)
        G = (f * x).cumsum(dim=1)/self.points # dim (arms, points)
        p_star = ((f * F_bar)/F).sum(dim=1)/self.points # dim (arms, points)
        m_same_arm = (((x * f)/F) * F_bar).sum(dim=1)/(self.points * p_star) # dim (arms, )
        m_different_arm = (((f * F_bar) / (F * F)) * G).sum(dim=1) / (self.points * p_star) # dim (arms, )
        maap = torch.where(torch.eye(self.arms) > 0, m_same_arm, m_different_arm) # dim (arms, arms)
        rho_star = torch.dot(p_star, m_same_arm) # dim (1,)
        delta = rho_star - (betas[:,0]/(betas[:,0] + betas[:,1])) # dim(arms,)
        log_weighted_different_arm = m_different_arm * torch.log((betas[:,0] + betas[:,1])/betas[:,0]) +  (1-m_different_arm) * torch.log((1-m_different_arm) * ((betas[:,0] + betas[:,1])/betas[:,1]))
        g = p_star *  log_weighted_different_arm # dim (arms, )
        return delta, g
    
    def act(self, delta, gain):
        #inputs: self.arms, delta-> dim=(arms,), gain-> dim=(arms,)
        q = torch.linspace(1/self.points,1,self.points) # dim (points,)
        left_shift_delta = torch.roll(delta,-1) # dim (arms,)
        left_shift_gain = torch.roll(gain, -1) # dim (arms,)
        delta_aa = torch.outer(delta, q) + torch.outer(left_shift_delta, (1-q)) # dim (arms, arms, points)
        gain_aa = torch.outer(gain, q) + torch.outer(left_shift_gain, (1-q)) # dim (arms, arms, points)
        delta_aa = torch.tril(delta_aa[:-1,:-1,:]) # dim (arms, arms, points)
        gain_aa = torch.tril(gain_aa[:-1,:-1,:]) # dim (arms, arms, points)
        ir = delta_aa ** 2 / gain_aa
        
        
        
        pass
    
    def update_params(self):
        pass
    
    def regret(self, action):
        pass
    
    def sample_act_regret(self, betas):
        delta, gain = self.sample(betas)
        optimalAction = self.act(delta, gain)
        betas = self.update_params()
        regret = self.regret(optimalAction)
        return torch.cumsum(regret), betas

    def cum_regret_over_horizon(self):
        cumRegret = torch.zeros((self.horizon), dtype=float)
        betas = self.betas
        for i in range(self.horizon):
            cumRegret[i], betas = self.sample_act_regret(betas)
        return torch.unsqueeze(cumRegret,-1) # dim (1, T)
    def run(self):
        cumRegret = torch.zeros((self.iterations, self.horizon), dtype=float) # dim = (iter, T)
        for i in tqdm(range(self.iterations)):
            cumRegret[i] = self.cum_regret_over_horizon() 
        avgCumRegret = torch.mean(cumRegret, dim=0) # dim = (1,T)
        return avgCumRegret 