# 2026-08-30

## What we tried
- Downloaded real 4K DJI drone flyover footage from YouTube.
- Trimmed to a 14.26s clip (requested -ss 5 -t 15, keyframe-snapped to 
  start 5.755s, actual duration 14.26s per ffprobe).
- Extracted frames at fps=2 → 29 frames, verified via `ls frames/ | wc -l`.
- Set up full CUT3R environment on RTX 4050 (6GB VRAM, 16GB RAM): CUDA, 
  PyTorch, RoPE CUDA kernels compiled, cut3r_224_linear_4.pth checkpoint 
  downloaded.
- Started GPU memory logging (`nvidia-smi --query-gpu=memory.used -l 1`) 
  to capture idle baseline (~100-150 MiB) ahead of inference.
- Launched CUT3R 224-checkpoint inference on the 29-frame set, timed 
  with `time python demo.py ...`.

## Result
- Runtime: TBD — inference in progress at time of writing
- VRAM peak: TBD — logger running, idle baseline confirmed ~100-150 MiB
- Scale check: TBD — pending point cloud output

## Reference (not our measurement)
Published CUT3R benchmark (Mem3R paper, arXiv:2604.07279): 26 fps, 
7930 MiB VRAM, 512 checkpoint, RTX PRO 6000, 512x384 res, 7-Scenes 
dataset. Not comparable to our 224-checkpoint / 6GB setup — cited 
for context only.

## Blockers
- Inference did not complete within submission window; run left 
  going in background.
- WSL terminal juggling (VRAM logger vs inference terminal) added 
  friction — worth scripting into one process next time.

## Next
- Capture actual runtime, peak VRAM, and scale-accuracy % once 
  current run finishes.
- Re-run with a smaller frame batch (5-10 frames) if OOM occurs, 
  to still get one real defensible number quickly.