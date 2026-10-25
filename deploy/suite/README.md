# RT research suite on one host

This directory starts the RTpipeline Web UI and the RTsegmentator portal together.
RTpipeline reaches the portal by its service name, `http://rtsegmentator:8080`.

Requirements: Docker with Compose v2, an NVIDIA GPU with the container toolkit, and the
licensed RTsegmentator model and runtime bundles described in the RTsegmentator
deployment guide. Model weights are not part of either image.

```bash
cp .env.example .env          # set RTSEG_MODEL_BUNDLE, RTSEG_RUNTIME_BUNDLE, enabled models
mkdir -p Input Output Logs Uploads
docker compose up -d
```

- RTpipeline Web UI: <http://127.0.0.1:8080>. The **RTsegmentator Models** option lists
  the portal's enabled CT models.
- RTsegmentator portal: <http://127.0.0.1:8081>.
- Command-line runs use `config.suite.yaml`:
  `docker compose exec rtpipeline snakemake --cores all --configfile /app/config.custom.yaml`.

Both services bind to 127.0.0.1 only. Neither application authenticates users; expose
them through an authenticating TLS proxy and keep patient data on controlled storage.
The portal and pipeline share one GPU in this layout. The portal runs one GPU job at a
time and queues the rest; with `segmentation.device: gpu`, RTpipeline also runs one
course's `segmentation_custom` stage at a time.
