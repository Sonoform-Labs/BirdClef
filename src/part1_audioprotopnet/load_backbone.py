import os
os.environ.setdefault("HF_HOME", r"D:\hf_cache")
import torch
from transformers import AutoModel, AutoConfig

REPO = "DBD-research-group/ConvNeXT-Base-BirdSet-XCL"

print("== config ==")
cfg = AutoConfig.from_pretrained(REPO, trust_remote_code=True)
print("config class:", type(cfg).__name__)
for k in ["num_channels", "num_labels", "image_size", "hidden_sizes", "depths", "problem_type"]:
    print(f"  {k}: {getattr(cfg, k, '?')}")

print("\n== model ==")
model = AutoModel.from_pretrained(REPO, trust_remote_code=True)
model.eval()
npar = sum(p.numel() for p in model.parameters())
print(f"model class: {type(model).__name__}  params: {npar/1e6:.1f}M")

print("\n== image processor (expected preprocessing) ==")
try:
    from transformers import AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained(REPO, trust_remote_code=True)
    for k in ["size", "image_mean", "image_std", "do_normalize", "do_resize", "crop_size"]:
        print(f"  {k}: {getattr(proc, k, '?')}")
except Exception as e:
    print("  no/failed processor:", type(e).__name__, str(e)[:200])

print("\n== forward shape probe ==")
for (H, W) in [(128, 500), (256, 313), (224, 224), (128, 313)]:
    x = torch.randn(1, cfg.num_channels if hasattr(cfg, "num_channels") else 1, H, W)
    with torch.no_grad():
        try:
            out = model(pixel_values=x)
            lhs = out.last_hidden_state
            pooler = getattr(out, "pooler_output", None)
            msg = f"  in {tuple(x.shape)} -> last_hidden_state {tuple(lhs.shape)}"
            if pooler is not None:
                msg += f", pooler {tuple(pooler.shape)}"
            print(msg)
        except Exception as e:
            print(f"  in {tuple(x.shape)} -> ERR {type(e).__name__}: {str(e)[:160]}")
print("\nP2 done.")
