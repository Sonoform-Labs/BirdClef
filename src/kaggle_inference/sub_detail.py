import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
subs = api.competition_submissions("birdclef-2026")
for s in subs[:6]:
    ref = getattr(s, "ref", None)
    err = getattr(s, "errorDescription", None) or getattr(s, "error_description", None)
    desc = (getattr(s, "description", "") or "")[:34]
    print("ref=%s [%s] status=%s pub=%s priv=%s\n    err=%r" % (
        ref, desc, getattr(s, "status", None),
        getattr(s, "publicScore", None), getattr(s, "privateScore", None), err))
