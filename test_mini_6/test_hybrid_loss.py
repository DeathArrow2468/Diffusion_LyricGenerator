import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from loss_func_1.config_loss import HybridLossConfig
from loss_func_1.hybrid_loss import HybridLoss
from loss_func_1.hybrid_batch import HybridCollateFn
from diffusion_model_3.model import BidirectionalDiffusionTransformer


class ToyTokenizer:
    bos_token_id = None
    pad_token_id = 0
    eos_token_id = 0

    def convert_tokens_to_ids(self, token):
        return {'[SEP]': 1, '[MASK]': 2}[token]

    def __len__(self):
        return 32

    def add_special_tokens(self, values):
        pass

    def save_pretrained(self, path):
        Path(path, "tokenizer-test.json").write_text("{}")

    def encode(self, text, **kwargs):
        return [3 + int(x) for x in text.split()]


class HybridTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = HybridLossConfig(half_life_start=1, half_life_end=1)
        self.loss = HybridLoss(self.config)

    def test_document_weights_and_sparse_absolute_distance(self):
        labels = torch.tensor([[-100, 1, 2, 3, 4]])
        q = self.loss.distance_weights(labels, torch.tensor([0]), 0, 10)
        torch.testing.assert_close(q, torch.tensor([[0., .505, .265, .145, .085]]))
        labels = torch.tensor([[-100, 1, -100, 3]])
        q = self.loss.distance_weights(labels, torch.tensor([0]), 0, 10)
        torch.testing.assert_close(q, torch.tensor([[0., .77, 0., .23]]))

    def test_empty_and_invalid_targets_rejected(self):
        for labels, boundary in [(torch.full((1, 3), -100), 0), (torch.tensor([[1, -100, -100]]), 0)]:
            with self.assertRaises(ValueError):
                self.loss.distance_weights(labels, torch.tensor([boundary]), 0, 10)

    def test_uniform_single_target_and_distant_numerics(self):
        labels = torch.tensor([[-100, -100, 1, 2], [-100, -100, -100, 2]])
        objective = HybridLoss(HybridLossConfig(rho=1))
        q = objective.distance_weights(labels, torch.tensor([0, 0]), 0, 10)
        torch.testing.assert_close(q, torch.tensor([[0., 0., .5, .5], [0., 0., 0., 1.]]))
        objective = HybridLoss(HybridLossConfig(half_life_start=1e-9, half_life_end=1e-9))
        q = objective.distance_weights(labels, torch.tensor([0, 0]), 0, 10)
        self.assertTrue(torch.isfinite(q).all())
        torch.testing.assert_close(q.sum(-1), torch.ones(2))

    def test_schedule_and_config_validation(self):
        objective = HybridLoss(HybridLossConfig(half_life_start=2, half_life_end=16))
        self.assertEqual(objective.half_life(5, 10), 9)
        self.assertEqual(objective.half_life(20, 10), 16)
        for kwargs in [dict(alpha=2), dict(rho=-1), dict(half_life_start=0), dict(noise_min=2)]:
            with self.assertRaises(ValueError):
                HybridLossConfig(**kwargs)

    def test_per_example_reduction_and_logit_gradient(self):
        logits = torch.randn(2, 5, 7, requires_grad=True)
        labels = torch.tensor([[-100, 1, 2, 3, 4], [-100, -100, 3, -100, -100]])
        boundary = torch.tensor([0, 0])
        q = self.loss.distance_weights(labels, boundary, 0, 10)
        result = self.loss.distance_loss(logits, labels, boundary, 0, 10)
        manual = torch.stack([sum(-q[b,i]*logits[b,i].log_softmax(-1)[labels[b,i]]
            for i in range(5) if labels[b,i] != -100) for b in range(2)]).mean()
        torch.testing.assert_close(result, manual)
        result.backward()
        expected = logits.detach().softmax(-1)
        expected.scatter_add_(-1, labels.clamp_min(0).unsqueeze(-1), -torch.ones(2,5,1))
        expected *= q.unsqueeze(-1) / 2
        torch.testing.assert_close(logits.grad, expected)

    def test_ar_shift(self):
        logits = torch.randn(2, 4, 7)
        labels = torch.tensor([[-100, 1, 2, 3], [-100, -100, 4, -100]])
        expected = (-logits[0,:3].log_softmax(-1)[torch.arange(3), torch.tensor([1,2,3])].mean()
                    -logits[1,1].log_softmax(-1)[4]) / 2
        torch.testing.assert_close(self.loss.ar_loss(logits, labels), expected)

    def test_collation_and_zero_noise_fallback(self):
        config = HybridLossConfig(noise_min=0, noise_max=0, max_target_tokens=2)
        batch = HybridCollateFn(ToyTokenizer(), 12, config)([
            dict(prompt='1 2', lyrics='3 4 5 6 7'), dict(prompt='1', lyrics='2')])
        self.assertEqual(batch['input_ids'].shape, (2, 12))
        self.assertTrue((batch['labels'].ne(-100).sum(-1) == 1).all())
        masked = batch['labels'].ne(-100)
        self.assertTrue((batch['input_ids'][masked] == 2).all())
        torch.testing.assert_close(batch['labels'][masked], batch['ar_input_ids'][masked])
        self.assertTrue((batch['attention_mask'].sum(-1) - batch['prefix_end'] - 1 <= 2).all())
        self.assertTrue((batch['input_ids'][~batch['attention_mask'].bool()] == 0).all())

    def tiny_model(self):
        config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                            attention_dropout=0.0)
        config._attn_implementation = 'sdpa'
        base = Qwen2ForCausalLM(config)
        with patch('diffusion_model_3.model.AutoModelForCausalLM.from_pretrained', return_value=base):
            return BidirectionalDiffusionTransformer('test', 32)

    def test_real_qwen_attention_no_future_leakage(self):
        model = self.tiny_model().eval()
        x = torch.tensor([[1, 3, 4, 5]])
        y = torch.tensor([[1, 3, 4, 8]])
        attention = torch.ones_like(x)
        with torch.no_grad():
            a = model(x, attention, mode='ar')
            b = model(y, attention, mode='ar')
            torch.testing.assert_close(a[:,:3], b[:,:3])
            a = model(x, attention, torch.zeros(1))
            b = model(y, attention, torch.zeros(1))
            self.assertGreater((a[:,0] - b[:,0]).abs().max().item(), 1e-6)
            attention[:,-1] = 0
            torch.testing.assert_close(model(x, attention, torch.zeros(1))[:,:3],
                                       model(y, attention, torch.zeros(1))[:,:3])

    def test_two_branch_cpu_optimizer_step(self):
        model = self.tiny_model()
        batch = HybridCollateFn(ToyTokenizer(), 12, self.config)([
            dict(prompt='1', lyrics='2 3 4 5'), dict(prompt='2', lyrics='3 4')])
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = model.base_model.get_input_embeddings().weight.detach().clone()
        ar = self.loss.ar_loss(model(batch['ar_input_ids'], batch['ar_attention_mask'], mode='ar'), batch['ar_labels'])
        (.7*ar).backward()
        den = self.loss.distance_loss(model(batch['input_ids'], batch['attention_mask'], batch['timesteps']),
            batch['labels'], batch['prefix_end'], 0, 10)
        (.3*den).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertGreater(model.time_embedder.mlp[2].weight.grad.abs().sum().item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before, model.base_model.get_input_embeddings().weight))

    def test_training_entry_point_checkpoint_and_partial_accumulation(self):
        from training_4 import training
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory, 'data.jsonl')
            data.write_text('\n'.join(json.dumps(dict(prompt='1', lyrics='2 3 4')) for _ in range(5)))
            settings = SimpleNamespace(DATASET_PATH=str(data), CHECKPOINT_DIR=directory,
                MODEL_NAME='test', STRUCTURAL_TOKENS=[], SEP_TOKEN='[SEP]', MASK_TOKEN='[MASK]',
                MAX_SEQ_LEN=12, MICRO_BATCH_SIZE=2, GRAD_ACCUM_STEPS=2,
                LEARNING_RATE=1e-3, EPOCHS=1, WARMUP_STEPS=0)
            model = self.tiny_model()
            with patch.object(training, 'config', settings), \
                 patch.object(training.AutoTokenizer, 'from_pretrained', return_value=ToyTokenizer()), \
                 patch.object(training, 'BidirectionalDiffusionTransformer', return_value=model):
                training.train(device='cpu')
            checkpoint = torch.load(Path(directory, 'best_model.pt'), weights_only=True)
            self.assertEqual(checkpoint['completed_steps'], 2)
            self.assertEqual(checkpoint['total_steps'], 2)
            self.assertEqual(checkpoint['loss_config']['rho'], .1)
            self.assertIn('scheduler_state_dict', checkpoint)


if __name__ == '__main__':
    unittest.main()
