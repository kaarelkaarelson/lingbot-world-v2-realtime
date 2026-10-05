from flashdreams.infra.config import derive_config
from flashdreams.recipes.wan.autoencoder.vae import WanVAEDecoderConfig
from lingbot.config import PIPELINE_LINGBOT_WORLD_FAST
from lingbot.impl.transformer.impl.network import LingbotWorldDiTNetwork1pt3BConfig

PIPELINE_LINGBOT_WORLD_V2_1P3B_CAUSAL_FAST = derive_config(
    PIPELINE_LINGBOT_WORLD_FAST,
    name="lingbot-world-v2-1p3b-causal-fast",
    decoder=WanVAEDecoderConfig(),
    diffusion_model=dict(
        transformer=dict(
            network=LingbotWorldDiTNetwork1pt3BConfig(
                patch_embedding_type="conv3d", control_type="cam",
                cp_method="ulysses", in_dim=16 + 4 + 16),
            checkpoint_path=("https://huggingface.co/robbyant/lingbot-world-v2-1.3b-causal-fast/"
                             "blob/main/model.safetensors.index.json"),
            checkpoint_min_free_gb=20.0,
            len_t=4, window_size_t=14, sink_size_t=6,
        ),
        scheduler=dict(shift=5.0),
    ),
)

def create_app():
    from lingbot.apps.cam2v.adapter import LingbotCam2VApplication
    return LingbotCam2VApplication(pipeline_config=PIPELINE_LINGBOT_WORLD_V2_1P3B_CAUSAL_FAST)
