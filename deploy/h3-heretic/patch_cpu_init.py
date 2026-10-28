"""Keep the already-quantized DiT on CPU while the encoder quantizes on GPU."""

import hashlib
from pathlib import Path

SOURCE_SHA256 = "9ec33a19fc5f8405860702c295d15082cade5638538b7bd15a4cfdb4e478886e"
OLD = """        self.transformer = transformer_cls(
            od_config,
            quant_config=transformer_quant_config,
            diffusers_weights=modular,
        )"""
NEW = """        # A component FP8 override makes the loader initialize on CUDA.
        # The serialized NVFP4 DiT needs no online quantization: preserve its
        # CPU-offloaded initialization while the encoder streams through CUDA.
        dit_init_context = (
            torch.device("cpu")
            if packed_dit.is_file() and resolve_offload(od_config).offloads(DIT_COMPONENT)
            else nullcontext()
        )
        with dit_init_context:
            self.transformer = transformer_cls(
                od_config,
                quant_config=transformer_quant_config,
                diffusers_weights=modular,
            )"""


def patch(path):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256 or data.decode().count(OLD) != 1:
        raise ValueError("H3 pipeline differs from the pinned upstream source")
    path.write_text(data.decode().replace(OLD, NEW))


if __name__ == "__main__":
    patch(
        Path(
            "/usr/local/lib/python3.12/dist-packages/vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py"
        )
    )
