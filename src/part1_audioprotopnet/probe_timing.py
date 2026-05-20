import os, sys, time
os.environ.setdefault("HF_HOME", r"D:\hf_cache")
sys.path.insert(0, r"D:\Python\BirdClef")
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel
REPO = "DBD-research-group/ConvNeXT-Base-BirdSet-XCL"
dev = torch.device("cuda")

class Proto(nn.Module):
    def __init__(self, n=234, ppc=5):
        super().__init__()
        self.bb = AutoModel.from_pretrained(REPO, trust_remote_code=True)
        self.P = n*ppc; self.proto = nn.Parameter(torch.randn(self.P,1024,1,1)*0.02); self.fc = nn.Linear(self.P,n)
    def forward(self, x):
        o = self.bb(pixel_values=x); f = F.normalize(o.last_hidden_state,1); p = F.normalize(self.proto,1)
        return self.fc(F.relu(F.conv2d(f,p)).amax((2,3)))

m = Proto().to(dev)
opt = torch.optim.AdamW(m.parameters(), 1e-4); lossf = nn.BCEWithLogitsLoss(); scaler = torch.amp.GradScaler("cuda")
N_TRAIN = 29987
for B in [16, 32, 48, 64]:
    try:
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(B,1,128,500,device=dev); y = torch.randint(0,2,(B,234),device=dev).float()
        for _ in range(5):
            opt.zero_grad()
            with torch.amp.autocast("cuda"): l = lossf(m(x), y)
            scaler.scale(l).backward(); scaler.step(opt); scaler.update()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(20):
            opt.zero_grad()
            with torch.amp.autocast("cuda"): l = lossf(m(x), y)
            scaler.scale(l).backward(); scaler.step(opt); scaler.update()
        torch.cuda.synchronize(); dt = (time.time()-t0)/20
        peak = torch.cuda.max_memory_allocated()/1e9
        steps = (N_TRAIN + B - 1)//B
        ep_min = dt*steps/60
        print(f"B={B:3d}: {dt*1000:6.0f} ms/step  peak {peak:4.1f} GB  steps/ep {steps:5d}  "
              f"train-epoch ~{ep_min:4.1f} min  | 5fold x2heads x6ep ~{ep_min*5*2*6/60:.1f} h", flush=True)
    except RuntimeError as e:
        print(f"B={B:3d}: OOM/ERR {str(e)[:90]}", flush=True); torch.cuda.empty_cache()
