EYE DEMO v2 (research prototype, not a medical device)

TABS
  Analyze         tick the models to run (3 CNNs, RETFound, Concat, Cross-attention, 2c), upload an image,
                  see a side-by-side table and per-model probabilities. Heatmaps (Grad-CAM) only for 2c.
  Compare models  graphs of the paper's numbers: ranking bars (AUROC, F1, AP, accuracy, latency, size),
                  by data source, by disease, accuracy-vs-cost, paired-bootstrap differences.
                  Chips choose which models are drawn. Values come from backend/results.json.
  History         saved analyses, with every model's result.
  Model & results summary tables.

CHECKPOINTS - put the .pth files in backend\checkpoints\ (a model with no file shows as unavailable):
  vgg16_full_best.pth  resnet50_full_best.pth  efficientnet_b3_full_best.pth  retfound_D123_best.pth
  fusion_D123_best.pth  attn_fusion_D123_best.pth  hybrid_xattn_query_best.pth
  (+ hybrid_xattn_query_thresholds.npy). Optional <name>_thresholds.npy per model, else 0.5 is used.

SETTINGS (environment variables)
  MAX_LOADED=2   models kept in memory at once      CNN_SIZE=300  CNN input size (use the trained size)
  DATABASE_URL   Neon/Postgres string (keep private); SQLite if empty
  FAKE_MODEL=1   UI test mode with random numbers

RUN
  cd backend
  pip install -r requirements.txt
  uvicorn app:app --port 8000        then open http://localhost:8000
  Front end is already built (backend\static). To change it: cd frontend, npm install, npm run build.
