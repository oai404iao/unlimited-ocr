"""Opt-in wall-clock attribution; exclusive values exclude measured children.

No tensor copies, profiler trace, or hooks on every Linear. Used for finding
hotspots only: benchmark latency again with this instrumentation removed.
"""
from collections import defaultdict
from functools import wraps
from time import perf_counter


class StageProfiler:
    def __init__(self, model, detailed=False):
        self.model = model
        self.phase = "prefill"
        self.stack = []
        self.stats = defaultdict(lambda: {"calls": 0, "inclusive_seconds": 0., "exclusive_seconds": 0.})
        self.originals = []
        self.detailed = detailed

    def wrap(self, owner, attr, label, detect_phase=False):
        original = getattr(owner, attr)

        @wraps(original)
        def measured(*args, **kwargs):
            if detect_phase:
                ids = kwargs.get("input_ids")
                if ids is not None:
                    self.phase = "prefill" if ids.shape[1] > 1 else "decode"
            key = f"{self.phase}/{label}"
            frame = [perf_counter(), 0.]
            self.stack.append(frame)
            try:
                return original(*args, **kwargs)
            finally:
                elapsed = perf_counter() - frame[0]
                self.stack.pop()
                if self.stack:
                    self.stack[-1][1] += elapsed
                item = self.stats[key]
                item["calls"] += 1
                item["inclusive_seconds"] += elapsed
                item["exclusive_seconds"] += elapsed - frame[1]

        self.originals.append((owner, attr, original))
        setattr(owner, attr, measured)

    def __enter__(self):
        self.wrap(self.model.model, "forward", "model", detect_phase=True)
        self.wrap(self.model.lm_head, "forward", "lm_head")
        for name in ("sam_model", "vision_model", "projector"):
            self.wrap(getattr(self.model.model, name), "forward", f"vision.{name}")
        for layer in self.model.model.layers:
            self.wrap(layer.self_attn, "forward", "attention")
            mlp = layer.mlp
            if hasattr(mlp, "experts"):
                self.wrap(mlp, "forward", "moe")
                self.wrap(mlp.gate, "forward", "moe.gate")
                for expert in mlp.experts:
                    self.wrap(expert, "forward", "moe.experts")
                if hasattr(mlp, "shared_experts"):
                    self.wrap(mlp.shared_experts, "forward", "moe.shared")
            else:
                self.wrap(mlp, "forward", "dense_mlp")
        from upstream import modeling_deepseekv2 as module
        for function, label in (("_router_qk", "attention.router_projection"),
                                ("_block_scores_infer", "attention.router_scores"),
                                ("_infer_route_select", "attention.route_select"),
                                ("_decode_attend", "attention.attend")):
            self.wrap(module, function, label)
        self.wrap(module.DashAttention, "_forward_dense", "attention.attend")
        if self.detailed:
            sam = self.model.model.sam_model
            self.wrap(sam.patch_embed, "forward", "sam.patch_embed")
            for block in sam.blocks:
                kind = "local" if block.window_size else "global"
                self.wrap(block.attn, "forward", f"sam.{kind}.attention")
                self.wrap(block.attn.qkv, "forward", f"sam.{kind}.qkv")
                self.wrap(block.attn.proj, "forward", f"sam.{kind}.proj")
                self.wrap(block.mlp, "forward", "sam.mlp")
                self.wrap(block.norm1, "forward", "sam.norm")
                self.wrap(block.norm2, "forward", "sam.norm")
            self.wrap(sam.neck, "forward", "sam.neck")
            self.wrap(sam.net_2, "forward", "sam.net_2")
            self.wrap(sam.net_3, "forward", "sam.net_3")
            for layer in self.model.model.layers:
                for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    self.wrap(getattr(layer.self_attn, name), "forward", f"attention.{name}")
                self.wrap(layer.self_attn, "_main_rope_coords", "attention.rope_coords")
                if layer.self_attn.rotary_emb_2d_main is not None:
                    self.wrap(layer.self_attn.rotary_emb_2d_main, "forward", "attention.rope_embedding")
        return self

    def __exit__(self, *exc):
        for owner, attr, original in reversed(self.originals):
            setattr(owner, attr, original)
        self.originals.clear()

    def report(self):
        return dict(sorted(self.stats.items()))
