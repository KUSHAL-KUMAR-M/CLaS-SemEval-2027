import unittest
import torch

from steering.subspace import fit_subspace, intervene, validate_artifact
from steering.subspace_hooks import hf_hook, vllm_hook, install_hf, clear_vllm


class SubspaceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.u = torch.linalg.qr(torch.randn(12, 2)).Q
        self.centers = torch.tensor([[-2., 0.], [2., 0.], [0., 2.]])
        self.variance = torch.ones(2)
        self.h = torch.randn(4, 12)

    def test_zero_strength_identity(self):
        result = intervene(self.h, self.u, self.centers, self.variance, 1, strength=0)
        torch.testing.assert_close(result, self.h)

    def test_preserves_complement_and_moves_to_centroid(self):
        result = intervene(self.h, self.u, self.centers, self.variance, 1,
                           strength=1, adaptive=False, max_relative_norm=100)
        torch.testing.assert_close((result-self.h) @ (torch.eye(12)-self.u@self.u.T), torch.zeros_like(self.h), atol=1e-5, rtol=0)
        torch.testing.assert_close(result @ self.u, self.centers[1].expand(4,-1))

    def test_norm_cap(self):
        result = intervene(self.h, self.u, self.centers, self.variance, 1,
                           strength=100, adaptive=False, max_relative_norm=.01)
        self.assertTrue(((result-self.h).norm(dim=-1) <= .01001*self.h.norm(dim=-1)).all())

    def test_gate_avoids_edit_when_target_is_confident(self):
        far_centers = self.centers * 10
        # Near, but not exactly at the centroid: a fixed edit would move it.
        h = ((far_centers[1] + torch.tensor([.1, .1])) @ self.u.T).unsqueeze(0)
        result = intervene(h, self.u, far_centers, self.variance, 1)
        torch.testing.assert_close(result, h)

    def test_fit_removes_shared_content(self):
        content = torch.randn(16, 1, 12)
        offsets = torch.stack([self.centers[i] @ self.u.T for i in range(3)])
        x = content.unsqueeze(0) + offsets[:,None,None,:]
        fitted = fit_subspace(x, 2)
        torch.testing.assert_close(fitted['basis'][0] @ fitted['basis'][0].T, self.u@self.u.T, atol=1e-5, rtol=1e-5)
        artifact = dict(fitted, format_version=1, model='test', languages=['a','b','c'], layers=[0])
        validate_artifact(artifact, 'test')
        with self.assertRaises(ValueError):
            validate_artifact(artifact, 'other')

    def state(self):
        return dict(method='fixed', basis=self.u, centers=self.centers, variance=self.variance,
                    target=1, strength=.5, threshold=.8, temperature=1., max_relative_norm=.1)

    def test_vllm_edits_full_residual_and_preserves_residual_operand(self):
        hidden, residual = self.h, torch.randn_like(self.h)
        h2, r2 = vllm_hook(self.state(), None, None, (hidden, residual))
        expected = intervene(hidden+residual, self.u, self.centers, self.variance, 1,
                             strength=.5, adaptive=False)
        torch.testing.assert_close(h2+r2, expected)
        self.assertIs(r2, residual)
        with self.assertRaises(RuntimeError):
            vllm_hook(self.state(), None, None, hidden)

    def test_hf_last_position_only(self):
        h = torch.randn(2, 5, 12)
        result = hf_hook(self.state(), 'last', None, None, (h, 'cache'))
        torch.testing.assert_close(result[0][:,:-1], h[:,:-1])
        self.assertFalse(torch.equal(result[0][:,-1], h[:,-1]))
        self.assertEqual(result[1], 'cache')

    def test_hook_removal_restores_model(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.layers = torch.nn.ModuleList([torch.nn.Linear(12,12, bias=False)])
        model = Model()
        artifact = dict(layers=[0], basis=self.u[None], centers=self.centers[None],
                        variance=self.variance[None], means=torch.randn(3,1,12))
        expected = model.model.layers[0](self.h)
        handles = install_hf(model, artifact, 0, 1, [0], self.state(), 'all')
        self.assertFalse(torch.equal(model.model.layers[0](self.h), expected))
        for handle in handles:
            handle.remove()
        torch.testing.assert_close(model.model.layers[0](self.h), expected)


if __name__ == '__main__':
    unittest.main()
