import os
import sys

sys.path.append('TRELLIS')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch

from trellis.pipelines import TrellisImageTo3DPipeline


def _load_sparse_structure_encoder(pipe: TrellisImageTo3DPipeline,
                                   pretrained: str) -> None:
    """Attach the SS VAE encoder to ``pipe.models`` under the key
    ``'sparse_structure_encoder'``.

    The default TRELLIS-image-large pipeline.json only lists models needed for
    image-to-3D inference (decoder + flow), omitting the encoder. Stage B v3
    needs it for the SDEdit guide encoding step. The encoder checkpoint lives
    alongside the decoder at
    ``<pretrained>/ckpts/ss_enc_conv3d_16l8_fp16.{json,safetensors}``.

    If the load fails (e.g. offline + no local cache), the encoder is not
    attached and SDEdit Pass 2 will skip gracefully with a warning.
    """
    if "sparse_structure_encoder" in pipe.models:
        return

    from trellis import models as trellis_models
    encoder_path = f"{pretrained}/ckpts/ss_enc_conv3d_16l8_fp16"
    try:
        encoder = trellis_models.from_pretrained(encoder_path)
    except Exception as e:
        print(
            f"[recon] WARNING: failed to load SS encoder from {encoder_path!r}: {e}. "
            f"Stage B SDEdit Pass 2 will be skipped."
        )
        return
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    pipe.models["sparse_structure_encoder"] = encoder


def build_trellis_pipeline(
    device: str = 'cuda',
    pretrained: str = 'JeffreyXiang/TRELLIS-image-large',
) -> TrellisImageTo3DPipeline:
    """Instantiate and move a TRELLIS image-to-3D pipeline to the target device.

    The v1 pipeline (stage_b_vgcf, stage_c_sajo, stage_d_placeholder, stage_f_assemble)
    consumes this pipeline object; callers are responsible for swapping
    `pipe.sparse_structure_sampler` with a VGCFSampler instance when they need
    cross-state velocity guidance.

    Stage B v3 also needs ``pipe.models['sparse_structure_encoder']`` for the
    SDEdit guide encoding step (not in the default pipeline.json bundle). We
    load it explicitly here; failure is non-fatal (Pass 2 degrades to Pass 1).
    """
    pipe = TrellisImageTo3DPipeline.from_pretrained(pretrained)
    _load_sparse_structure_encoder(pipe, pretrained)
    if device == 'cuda' and torch.cuda.is_available():
        pipe.cuda()
    return pipe
