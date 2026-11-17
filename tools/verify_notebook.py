"""Run mathematical checks against the actual notebook cells; no dataset download."""
from pathlib import Path
import math
import unittest
import nbformat
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.mixture import GaussianMixture

nb=nbformat.read(Path(__file__).resolve().parents[1]/'VaDE_CIFAR10.ipynb',as_version=4)
ns=dict(torch=torch, nn=nn, F=F, math=math, np=np,
        linear_sum_assignment=linear_sum_assignment,
        normalized_mutual_info_score=normalized_mutual_info_score,
        adjusted_rand_score=adjusted_rand_score,CFG={'n_clusters':10})
for cell in nb.cells:
    if cell.cell_type!='code': continue
    if cell.source.startswith('class Encoder'):
        exec(cell.source.split('\nmodel = VaDE')[0], ns)
    if cell.source.startswith('def clustering_metrics'):
        exec(cell.source, ns)
VaDE=ns['VaDE']
torch.set_num_threads(2)

class NotebookChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.net=VaDE()
        with torch.no_grad():
            self.net.mu_c.normal_(0,0.5)
            self.net.logvar_c.uniform_(-1,1)
            self.net.pi_logits.normal_()
        self.x=torch.rand(4,3,32,32)
    def test_gmm_probabilities(self):
        z=np.random.default_rng(4).normal(size=(200,10)).astype('float32')
        gmm=GaussianMixture(10,covariance_type='diag',random_state=4).fit(z)
        with torch.no_grad():
            self.net.mu_c.copy_(torch.from_numpy(gmm.means_))
            self.net.logvar_c.copy_(torch.from_numpy(np.log(gmm.covariances_)))
            self.net.pi_logits.copy_(torch.from_numpy(np.log(gmm.weights_)))
            gamma,_=self.net.responsibilities(torch.from_numpy(z))
        np.testing.assert_allclose(gamma.numpy(),gmm.predict_proba(z),atol=2e-5,rtol=2e-5)
    def test_loss_matches_distribution_kl(self):
        for likelihood in ['gaussian','bce']:
            torch.manual_seed(123)
            loss,terms=self.net.loss(self.x,likelihood,0.1)
            torch.manual_seed(123)
            mu,lv=self.net.encoder(self.x)
            z=mu+torch.randn_like(mu)*(0.5*lv).exp()
            logits=self.net.decoder(z)
            zg=mu+torch.randn_like(mu)*(0.5*lv).exp()
            gamma,lg=self.net.responsibilities(zg)
            q=torch.distributions.Normal(mu[:,None],(0.5*lv[:,None]).exp())
            p=torch.distributions.Normal(self.net.mu_c[None],(0.5*self.net.logvar_c[None]).exp())
            klz=(gamma*torch.distributions.kl_divergence(q,p).sum(-1)).sum(1).mean()
            klc=torch.distributions.kl_divergence(
                torch.distributions.Categorical(probs=gamma),
                torch.distributions.Categorical(logits=self.net.pi_logits)).mean()
            if likelihood=='gaussian':
                rec=-torch.distributions.Normal(logits.sigmoid(),math.sqrt(0.1)).log_prob(self.x).flatten(1).sum(1).mean()
            else:
                rec=F.binary_cross_entropy_with_logits(logits,self.x,reduction='sum')/len(self.x)
            torch.testing.assert_close(loss,rec+klz+klc,rtol=1e-5,atol=1e-4)
            torch.testing.assert_close(terms['kl_z'],klz)
            torch.testing.assert_close(terms['kl_c'],klc)
    def test_gradients_and_inference(self):
        for likelihood in ['gaussian','bce']:
            self.net.zero_grad()
            loss,_=self.net.loss(self.x,likelihood)
            loss.backward()
            for name,param in self.net.named_parameters():
                self.assertIsNotNone(param.grad,name)
                self.assertTrue(torch.isfinite(param.grad).all(),name)
            for name in ['pi_logits','mu_c','logvar_c']:
                self.assertGreater(getattr(self.net,name).grad.abs().sum().item(),0,name)
        self.net.eval()
        self.assertTrue(torch.equal(self.net.predict(self.x),self.net.predict(self.x)))
        mu,_=self.net.encoder(self.x)
        self.assertEqual(tuple(self.net.decoder(mu).shape),(4,3,32,32))
        gamma,_=self.net.responsibilities(mu)
        torch.testing.assert_close(gamma.sum(1),torch.ones(4))
    def test_acc_permutation_and_collapse(self):
        y=np.tile(np.arange(10),3)
        m,_=ns['clustering_metrics'](y,(y+3)%10)
        self.assertEqual(m['ACC'],1)
        m,_=ns['clustering_metrics'](y,np.zeros_like(y))
        self.assertEqual(m['ACC'],0.1)
        self.assertEqual(m['occupied_clusters'],1)

if __name__=='__main__': unittest.main(verbosity=2)
