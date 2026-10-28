import json
from collections import OrderedDict

import pytest
import torch
from safetensors.torch import save_file

from unlimited_ocr import cpu_checkpoint as checkpoint
from unlimited_ocr import cpu_runtime as runtime


@pytest.fixture
def source(tmp_path, monkeypatch):
    import upstream.modeling_unlimitedocr as upstream

    class Tiny(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.embed = torch.nn.Embedding(12, 8)
            self.proj = torch.nn.Linear(8, 6)
            self.output = torch.nn.Linear(6, 4, bias=False)
            self.norm = torch.nn.LayerNorm(4)
            self.register_buffer("positions", torch.arange(4))

        def forward(self, ids):
            return self.norm(self.output(self.proj(self.embed(ids))))

    monkeypatch.setattr(upstream, "UnlimitedOCRForCausalLM", Tiny)
    monkeypatch.setitem(runtime.POLICY, "selected", "avx2")
    torch.backends.quantized.engine = "fbgemm"
    torch.manual_seed(23)
    model = Tiny(None)
    state = {k: v.to(torch.bfloat16) if v.is_floating_point() else v
             for k, v in model.state_dict().items()}
    path = tmp_path / "source"
    path.mkdir()
    save_file(state, path / "weights.safetensors")
    for name in checkpoint.ASSETS:
        (path / name).write_text('{"sparse_attention_config": {}}' if name == "config.json" else '{}')
    (path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {k: "weights.safetensors" for k in state}}))
    return path


def test_round_trip_without_source_or_requantization(source, tmp_path, monkeypatch):
    expected = runtime.load_model(source)
    ids = torch.tensor([[1, 3, 7]])
    destination = tmp_path / "export"
    assert checkpoint.export_quantized_model(source, destination) == destination
    assert checkpoint.is_quantized_checkpoint(destination)
    assert set(p.name for p in destination.iterdir()) == {
        *checkpoint.ASSETS, checkpoint.MANIFEST_FILENAME, checkpoint.STATE_FILENAME}
    state = torch.load(destination / checkpoint.STATE_FILENAME, weights_only=True, mmap=True)
    assert isinstance(state, OrderedDict)
    assert state._metadata["proj"]["version"] == 4
    assert state["proj._packed_params._packed_params"][0].is_quantized
    source.rename(tmp_path / "source-unavailable")

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline load must not requantize or read safetensors")

    monkeypatch.setattr(runtime, "pack_linear", forbidden)
    monkeypatch.setattr(runtime, "safe_open", forbidden)
    original_load = torch.load

    def checked_load(path, **options):
        assert options == dict(map_location="cpu", weights_only=True, mmap=True)
        return original_load(path, **options)

    monkeypatch.setattr(checkpoint.torch, "load", checked_load)
    loaded = runtime.load_model(destination)
    torch.testing.assert_close(loaded(ids), expected(ids), rtol=0, atol=0)
    assert loaded.embed.weight.dtype == torch.bfloat16
    assert loaded.embed(ids).dtype == torch.float32
    assert loaded._cpu_weight_format == "int8"
    assert loaded._cpu_quantized_weights
    assert not loaded.training
    assert not any(p.requires_grad or p.is_meta for p in loaded.parameters())
    torch.testing.assert_close(loaded.positions, expected.positions)


def test_generic_uses_dequantized_weights_without_fbgemm(source, tmp_path, monkeypatch):
    destination = checkpoint.export_quantized_model(source, tmp_path / "export")
    packed = runtime.load_model(destination)
    monkeypatch.setitem(runtime.POLICY, "selected", "generic")

    def forbidden(*args, **kwargs):
        raise AssertionError("Generic checkpoint load must not instantiate FBGEMM modules")

    monkeypatch.setattr(checkpoint.Int8Linear, "__init__", forbidden)
    with pytest.warns(UserWarning, match="per-operation"):
        loaded = runtime.load_model(destination)
    assert not loaded._cpu_quantized_weights
    assert loaded._cpu_weight_format == "int8_dequantize"
    assert loaded.proj.weight.is_quantized
    assert not list(loaded.proj.parameters())
    x = torch.randn(2, 8)
    w, b = packed.proj._weight_bias()
    torch.testing.assert_close(loaded.proj(x), torch.nn.functional.linear(x, w.dequantize(), b), rtol=0, atol=0)
    assert not any(p.requires_grad or p.is_meta for p in loaded.parameters())


@pytest.mark.parametrize("options", [{"quantized": False}, {"quantized": False, "bf16_storage": True}])
def test_reference_modes_rejected(source, tmp_path, options):
    destination = checkpoint.export_quantized_model(source, tmp_path / "export")
    with pytest.raises(ValueError, match="reference weights"):
        runtime.load_model(destination, **options)


@pytest.mark.parametrize("contents", [None, "{", '{}', '{"format": "other", "version": 1}'])
def test_invalid_manifest_rejected(source, tmp_path, contents):
    destination = checkpoint.export_quantized_model(source, tmp_path / "export")
    manifest = destination / checkpoint.MANIFEST_FILENAME
    if contents is None:
        manifest.unlink()
    else:
        manifest.write_text(contents)
    assert checkpoint.is_quantized_checkpoint(destination)
    with pytest.raises(ValueError, match="manifest|format/version"):
        runtime.load_model(destination)


@pytest.mark.parametrize("mutation", ["missing", "layer-version", "invalid-entry", "torch-version"])
def test_serialization_versions_rejected(source, tmp_path, mutation):
    destination = checkpoint.export_quantized_model(source, tmp_path / "export")
    if mutation == "torch-version":
        path = destination / checkpoint.MANIFEST_FILENAME
        manifest = json.loads(path.read_text())
        manifest["torch_version"] = "0.0.0"
        path.write_text(json.dumps(manifest))
    else:
        path = destination / checkpoint.STATE_FILENAME
        state = torch.load(path, weights_only=True)
        if mutation == "missing":
            del state._metadata
        elif mutation == "invalid-entry":
            state._metadata["proj"] = "invalid"
        else:
            state._metadata["proj"]["version"] = -1
        torch.save(state, path)
    with pytest.raises(ValueError, match="metadata|version"):
        runtime.load_model(destination)


@pytest.mark.parametrize("nonempty", [False, True])
def test_existing_destination_not_modified(source, tmp_path, nonempty):
    destination = tmp_path / "export"
    destination.mkdir()
    if nonempty:
        (destination / "keep.txt").write_text("keep")
    before = list(destination.iterdir())
    with pytest.raises(FileExistsError):
        checkpoint.export_quantized_model(source, destination)
    assert list(destination.iterdir()) == before


def test_generic_export_rejected(source, tmp_path, monkeypatch):
    monkeypatch.setitem(runtime.POLICY, "selected", "generic")
    with pytest.raises(ValueError, match="requires AVX2"):
        checkpoint.export_quantized_model(source, tmp_path / "export")
    assert not (tmp_path / "export").exists()


@pytest.mark.parametrize("missing", [True, False])
def test_missing_or_corrupt_state_rejected(source, tmp_path, missing):
    destination = checkpoint.export_quantized_model(source, tmp_path / "export")
    path = destination / checkpoint.STATE_FILENAME
    if missing:
        path.unlink()
    else:
        path.write_bytes(b"invalid checkpoint archive")
    with pytest.raises(FileNotFoundError if missing else RuntimeError):
        runtime.load_model(destination)


@pytest.mark.parametrize("mutation", ["missing-pair", "weight-shape", "weight-dtype", "bias-shape", "bias-dtype"])
def test_invalid_weight_layout_rejected(source, tmp_path, mutation):
    destination = checkpoint.export_quantized_model(source, tmp_path / "export")
    path = destination / checkpoint.STATE_FILENAME
    state = torch.load(path, weights_only=True)
    key = "proj._packed_params._packed_params"
    weight, bias = state[key]
    if mutation == "missing-pair":
        del state[key]
    else:
        if mutation == "weight-shape":
            weight = weight[:, :-1]
        elif mutation == "weight-dtype":
            weight = weight.dequantize()
        elif mutation == "bias-shape":
            bias = bias[:-1]
        else:
            bias = bias.double()
        state[key] = (weight, bias)
    torch.save(state, path)
    with pytest.raises(ValueError, match="INT8"):
        runtime.load_model(destination)


def test_failed_export_has_no_completion_manifest(source, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("simulated write error")

    monkeypatch.setattr(checkpoint.shutil, "copyfile", fail)
    destination = tmp_path / "export"
    with pytest.raises(OSError, match="simulated"):
        checkpoint.export_quantized_model(source, destination)
    assert (destination / checkpoint.STATE_FILENAME).exists()
    assert not (destination / checkpoint.MANIFEST_FILENAME).exists()
    with pytest.raises(ValueError, match="manifest"):
        runtime.load_model(destination)
