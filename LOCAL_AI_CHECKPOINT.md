# AEGIS Local AI Checkpoint - 2026-11-04

## Gemma 4 E2B
- Runtime: LiteRT-LM 0.19.0
- Model downloaded and running locally on Raspberry Pi 5 8GB
- Model cache stored outside repository under ~/.litert-lm
- Local AI venv stored outside repository at ~/aegis-local-ai

### Benchmark
- Prefill: 96.03 tokens/s
- Decode: 5.43 tokens/s
- Init time: 0.7947 s
- Time to first token: 2.8500 s

### Navigation tests
- First test returned TURN_LEFT toward a front-left obstacle; result considered poor.
- Clean second test returned MOVE with confidence 0.85.
- Second result correctly used open center/right geometry and did not request an unnecessary scan.
- Whole second inference took about 19.1 seconds including process startup.

## Qwen3.5-4B
- mixed_int4 LiteRT model downloaded successfully.
- Benchmark currently crashes during warmup.
- Disk cache failed with XNNPACK: invalid size for buffer list descriptor.
- Memory-cache attempt also crashed.
- Resume debugging Qwen next session.

## Next steps
1. Diagnose Qwen3.5-4B LiteRT warmup crash.
2. Run identical AEGIS navigation scenarios on Gemma and Qwen.
3. Compare reasoning consistency, latency, RAM, and scan/no-op behavior.
4. Choose primary local model.
5. Build local_ai.py and common decision schema.
6. Add shadow mode: Luna controls rover while local AI decides in parallel.
7. Integrate camera and compact LiDAR representations later.
