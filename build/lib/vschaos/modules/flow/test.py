import unittest
import torch
import torch.distributions as distrib
# Different flows to test
from flow.flow import NormalizingFlow
from flow.basic import PlanarFlow, RadialFlow
from flow.order import ReverseFlow, ShuffleFlow
from flow.layers import BatchNormFlow, PReLUFlow, AffineFlow, AffineLUFlow, ActNormFlow
from flow.ar import MaskedCouplingFlow, MAFlow, IAFlow, ContextIAFlow

eps = 1e-5
batch_size = 32
d_inputs = 11
n_hidden = 64
mask = torch.arange(0, d_inputs) % 2
mask = mask.unsqueeze(0)

class TestFlow(unittest.TestCase):
    
    def assertFlowInversion(self, flow, name):
        # Create random input
        x = torch.randn(batch_size, d_inputs)
        # Transform the input
        logdet = flow.log_abs_det_jacobian(x)
        y = flow(x)
        # Inverse the output
        z = flow._inverse(y)
        inv_logdet = flow.log_abs_det_jacobian(z)
        # Test that it is a bijection
        self.assertTrue((x - z).abs().max() < eps, name + ' is not a bijection.')
        self.assertTrue((logdet - inv_logdet).abs().max() < eps, name + ' has incoherent determinants.')    
        
    """
    Single iteration tests    
    """
    
    def testReverseFlow(self):
        flow = ReverseFlow(d_inputs)
        self.assertFlowInversion(flow, 'ReverseFlow')
    
    def testShuffleFlow(self):
        flow = ShuffleFlow(d_inputs)
        self.assertFlowInversion(flow, 'ShuffleFlow')
    
    def testPReLUFlow(self):
        flow = PReLUFlow(d_inputs)
        self.assertFlowInversion(flow, 'PReLUFlow')
    
    def testAffineFlow(self):
        flow = AffineFlow(d_inputs)
        self.assertFlowInversion(flow, 'AffineFlow')
    
    def testAffineLUFlow(self):
        flow = AffineLUFlow(d_inputs)
        self.assertFlowInversion(flow, 'AffineLUFlow')
    
    def testPlanarFlow(self):
        flow = PlanarFlow(d_inputs)
        self.assertFlowInversion(flow, 'PlanarFlow')
    
    def testRadialFlow(self):
        flow = RadialFlow(d_inputs)
        self.assertFlowInversion(flow, 'RadialFlow')
        
    """
    Multiple iteration tests    
    """
    
    def testBatchNormFlow(self):
        flow = BatchNormFlow(d_inputs)
        flow.train()
        # First run
        self.assertFlowInversion(flow, 'BatchNormFlow')
        # Second run
        self.assertFlowInversion(flow, 'BatchNormFlow - Second run')
        # Eval run
        flow.eval()
        self.assertFlowInversion(flow, 'BatchNormFlow - Eval run')

    def testActNorm(self):
        flow = ActNormFlow(d_inputs)
        # First run
        self.assertFlowInversion(flow, 'ActNormFlow')
        # Second run
        self.assertFlowInversion(flow, 'ActNormFlow - Second run')
        
    """
    Sequential tests    
    """
    
    def testSequential(self):
        # Sets of flows
        blocks = { 
                'glow': [ ActNormFlow, AffineLUFlow, MaskedCouplingFlow ],
                'maf': [ MAFlow ], 
                'iaf': [ IAFlow ], 
                'iaf_c': [ ContextIAFlow ]
                }
        flow = NormalizingFlow(d_inputs, blocks['glow'], 1, distrib.Normal(0, 1))
        # First run
        self.assertFlowInversion(flow, 'Sequential')
        # Second run
        self.assertFlowInversion(flow, 'Sequential - Second run')
    
    def testSequentialRepeat(self):
        # Sets of flows
        blocks = { 
                'glow': [ ActNormFlow, AffineLUFlow, MaskedCouplingFlow ],
                'maf': [ MAFlow ], 
                'iaf': [ IAFlow ], 
                'iaf_c': [ ContextIAFlow ]
                }
        flow = NormalizingFlow(d_inputs, blocks['glow'], 16, distrib.Normal())
        # First run
        self.assertFlowInversion(flow, 'Sequential')
        # Second run
        self.assertFlowInversion(flow, 'Sequential - Second run')
        # Eval run
        flow.eval()
        self.assertFlowInversion(flow, 'BatchNormFlow - Eval run')

if __name__ == "__main__":
    unittest.main()
