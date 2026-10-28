"""Small models exercise the real HF generate loop, without model weights."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast

from upstream import modeling_unlimitedocr as ocr
from upstream._sparse_base_impl import DeepseekV2ForCausalLM


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList()
        self.calls = []
        self.cache_flags = []
        self.fail = False

    def forward(self, input_ids, past_key_values, position_ids, **kwargs):
        if self.fail:
            raise RuntimeError("decoder failure")
        self.calls.append((input_ids.clone(), position_ids.clone(), kwargs.get("images")))
        self.cache_flags.append(kwargs.get("use_cache"))
        step = getattr(past_key_values, "test_step", 0)
        past_key_values.test_step = step + 1
        keys = torch.zeros(1, 1, input_ids.shape[1], 2)
        past_key_values.update(keys, keys.clone(), 0)
        hidden = torch.zeros(1, input_ids.shape[1], self.config.hidden_size)
        hidden[..., 0] = step
        return BaseModelOutputWithPast(last_hidden_state=hidden, past_key_values=past_key_values)


class Head(nn.Module):
    def __init__(self, tokens):
        super().__init__()
        self.tokens = tokens
        self.lengths = []

    def forward(self, hidden):
        self.lengths.append(hidden.shape[1])
        scores = torch.full((*hidden.shape[:2], 16), -10.0)
        token = self.tokens[min(int(hidden[0, -1, 0]), len(self.tokens) - 1)]
        scores[..., token] = 10
        return scores


class TinyOCR(ocr.UnlimitedOCRForCausalLM):
    def __init__(self, tokens):
        config = ocr.UnlimitedOCRConfig(vocab_size=16, hidden_size=4, num_hidden_layers=0,
                                       bos_token_id=1, eos_token_id=2, pad_token_id=2,
                                       sparse_attention_config={"page_token_id": 3})
        PreTrainedModel.__init__(self, config)
        self.model = Decoder(config)
        self.lm_head = Head(tokens)
        self.config._ring_window = 17
        self.config.sliding_window = 23
        self.eval()


TOKENIZER = SimpleNamespace(eos_token_id=2, decode=lambda ids, **kwargs: " ".join(map(str, ids)))


def generate(model, **kwargs):
    return model._uocr_generate(TOKENIZER, torch.tensor([1, 3, 4]), None, None, None,
                               max_new_tokens=kwargs.pop("max_new_tokens", 8), **kwargs)


def test_generation_mixin_inheritance_and_actual_generate(monkeypatch):
    assert GenerationMixin in DeepseekV2ForCausalLM.__bases__
    assert ocr.UnlimitedOCRForCausalLM.can_generate()
    calls = []
    original = GenerationMixin.generate

    def spy(self, *args, **kwargs):
        calls.append(kwargs)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(GenerationMixin, "generate", spy)
    model = TinyOCR([3, 5, 2])
    result = generate(model)
    assert len(calls) == 1
    assert result["token_ids"] == [3, 5]
    assert result["finish_reason"] == "stop"
    assert result["num_pages_out"] == result["expected_pages"] == 1
    assert 0 < result["prefill_time"] <= result["time"]
    assert model.lm_head.lengths == [1, 1, 1]
    assert [c[0].shape[1] for c in model.model.calls] == [3, 1, 1]
    assert [c[1].tolist() for c in model.model.calls] == [[[0, 1, 2]], [[3]], [[4]]]
    assert model.config._ring_window == 17 and model.config.sliding_window == 23
    assert not ocr._DASH_INFER["on"]


def test_images_only_in_prefill_and_request_cache_is_fresh(monkeypatch):
    model = TinyOCR([3, 5, 2])
    original = model.generate
    processors = []

    def spy(**kwargs):
        processor = kwargs["logits_processor"][0]
        assert processor.guard_state == {'marker_cum': 0, 'scanned': 0, 'page_len': 0}
        processors.append(processor)
        return original(**kwargs)

    monkeypatch.setattr(model, "generate", spy)
    for _ in range(2):
        result = model._uocr_generate(TOKENIZER, torch.tensor([1, 3, 4]),
                                     (torch.zeros(1), torch.zeros(1)), torch.zeros(3),
                                     torch.zeros(1), max_new_tokens=8)
        assert result["token_ids"] == [3, 5]
    assert [c[2] is not None for c in model.model.calls] == [True, False, False] * 2
    assert processors[0] is not processors[1]
    assert processors[0].guard_state is not processors[1].guard_state


def test_failure_restores_dash_and_window_state():
    model = TinyOCR([3])
    model.model.fail = True
    with pytest.raises(RuntimeError, match="decoder failure"):
        generate(model)
    assert model.config._ring_window == 17 and model.config.sliding_window == 23
    assert not ocr._DASH_INFER["on"]


def test_model_generation_defaults_cannot_override_ocr_contract(monkeypatch):
    model = TinyOCR([3, 5, 2])
    model.generation_config.use_cache = False
    model.generation_config.do_sample = True
    model.generation_config.num_beams = 2
    model.generation_config.top_k = 1
    model.generation_config.top_p = 0.1
    model.generation_config.repetition_penalty = 2.0
    original = model._prepare_generation_config
    effective = []

    def spy(*args, **kwargs):
        config, model_kwargs = original(*args, **kwargs)
        effective.append(config)
        return config, model_kwargs

    monkeypatch.setattr(model, "_prepare_generation_config", spy)
    result = generate(model)
    assert result["token_ids"] == [3, 5]
    assert result["finish_reason"] == "stop"
    assert model.model.cache_flags == [True, True, True]
    assert [c[0].shape[1] for c in model.model.calls] == [3, 1, 1]
    assert effective[0].use_cache and not effective[0].do_sample
    assert effective[0].num_beams == 1 and effective[0].top_k is None
    assert effective[0].top_p == 1.0 and effective[0].repetition_penalty == 1.0
    assert model.generation_config.use_cache is False


@pytest.mark.parametrize("options,expected", [({"max_new_tokens": 2}, [3, 5]),
                                             ({"max_length": 4}, [3]),
                                             ({"max_new_tokens": 0}, [])])
def test_length_boundaries(options, expected):
    model = TinyOCR([3, 5, 6, 7])
    result = generate(model, **options)
    assert result["token_ids"] == expected
    assert result["finish_reason"] == "length"
    if options.get("max_new_tokens") == 0:
        assert len(model.model.calls) == 1
        assert model.lm_head.lengths == []
        assert result["prefill_time"] is None


def test_eos_first_retains_first_token_timing_but_not_eos():
    result = generate(TinyOCR([2]), page_count_guard=False)
    assert result["token_ids"] == [] and result["finish_reason"] == "stop"
    assert result["prefill_time"] is not None


def test_page_and_ngram_processor_matches_original_helpers():
    processor = ocr._UOCRLogitsProcessor(3, 2, 3, 2, 3, 2, 10, {8})
    state = {'marker_cum': 0, 'scanned': 0, 'page_len': 0}
    for generated in [[], [3], [3, 5], [3, 5, 5], [3, 5, 5, 5], [3, 5, 5, 5, 3]]:
        scores = torch.arange(16, dtype=torch.float32)
        expected = ocr._uocr_no_repeat_ngram(scores.clone(), generated, 2, 10, {8})
        expected = ocr._uocr_page_count_guard(expected, generated, 2, 3, 2, 3, state)
        actual = processor(torch.tensor([[1, 3, 4] + generated]), scores[None])
        torch.testing.assert_close(actual[0], expected)


def test_sampling_has_no_implicit_top_k_or_top_p(monkeypatch):
    model = TinyOCR([3])
    observed = []

    def spy(**kwargs):
        observed.append(kwargs["generation_config"])
        return torch.tensor([[1, 3, 4, 2]])

    monkeypatch.setattr(model, "generate", spy)
    generate(model, temperature=0.7)
    config = observed[0]
    assert config.do_sample and config.temperature == 0.7
    assert config.top_k == 0 and config.top_p == 1.0


def test_stream_contains_neither_prompt_nor_eos(monkeypatch):
    tokens, endings = [], []

    class TextOutput:
        def __init__(self, *args, **kwargs):
            pass

        def put(self, value):
            tokens.extend(value.tolist())

        def end(self):
            endings.append(True)

    monkeypatch.setattr(ocr, "TextStreamer", TextOutput)
    result = generate(TinyOCR([3, 5, 2]), stream=True)
    assert tokens == result["token_ids"] == [3, 5]
    assert endings == [True]


@pytest.mark.parametrize("batch,options", [(2, {}), (1, {"num_beams": 2}),
                                          (1, {"do_sample": True, "num_return_sequences": 2})])
def test_standard_generation_rejects_batch_and_beams(batch, options):
    model = TinyOCR([3, 5, 2])
    with pytest.raises(ValueError, match="batch size 1"):
        model.generate(torch.tensor([[1, 3, 4]] * batch), max_new_tokens=2,
                       pad_token_id=2, eos_token_id=2, **options)
    assert not model.model.calls
