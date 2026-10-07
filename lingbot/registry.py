"""Model id -> what to build. A new size or version of an existing architecture is a new entry here
(and a config in reference/wan/configs), not a new model file."""
MODELS = {
    "lingbot-world-2.0-1.3b": dict(
        task="i2v-1.3B",                                   # reference/wan/configs: shapes and checkpoints
        pipeline="lingbot.pipelines.lingbot_world:LingBotWorldPipeline",
        ckpt_dir="weights/lingbot-world-v2-1.3b-causal-fast",
        assets_dir="weights/lingbot-world-v2-14b-causal-fast",  # T5, VAE, tokenizer (not in the 1.3B upload)
    ),
}
DEFAULT_MODEL = "lingbot-world-2.0-1.3b"


def pipeline_class(model_id):
    import importlib
    module, cls = MODELS[model_id]["pipeline"].split(":")
    return getattr(importlib.import_module(module), cls)
