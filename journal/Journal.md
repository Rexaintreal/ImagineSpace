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

# 2026-09-15

## What we tried
- Went back to the CUT3R build from the 2026-08-30 run, same RTX 4050 (6GB VRAM) and WSL setup.
- Looked into why the CUDA RoPE extension would not build. Conda's nvcc (12.1, the version that matches PyTorch) was being blocked by a system/WSL nvcc (13.3), even after adding `$CONDA_PREFIX/bin` to the front of PATH. `nvcc --version` kept showing 13.3 no matter what.
- Since the SIH presentation was today (2 to 4 PM), we stopped trying to fix the toolchain and focused on the actual crash instead.
- Found the real bug in `src/croco/models/pos_embed.py`. CUT3R gives its persistent-state tokens a placeholder position (likely -1). `torch.nn.functional.embedding` cannot look up a negative index on CUDA, and that was causing the `srcIndex < srcSelectDimSize` crash.
- Fixed it with one line: `positions = positions.clamp(min=0).long()` before the embedding lookup. Checked the file text matched before saving the change.
- Ran `demo.py` again on the same 29-frame drone clip. It finished cleanly using the PyTorch fallback (the CUDA RoPE kernel still was not being used).
- The point cloud viewer (viser) first showed nothing but a small camera icon. Fixed this by switching playback mode from 4D (single frame) to 3D (all frames) and increasing the point size.
- After that, the point cloud showed up properly, with clear shapes for the ground, trees, and camera path. See the screenshot below.

![CUT3R point cloud output](https://github.com/Rexaintreal/ImagineSpace/blob/main/images/cut3r.png)

## Result
- Runtime: 4.62 seconds total, or 0.16 seconds per frame. This was on the PyTorch fallback, not the compiled CUDA kernel.
- Point cloud output: clear and structured. You can see a green vegetation area, a gray road or ground strip with the camera path traced along it in blue, and a lighter, lower-confidence area for distant terrain.
- This confirms the full pipeline works, start to finish, on real 4K drone footage (a YouTube video, trimmed to a 14.26 second clip, 29 frames at 2 fps), even on a normal 6GB VRAM laptop GPU.
- The CUDA RoPE kernel still does not compile because of the nvcc version issue. We are running the slower fallback, not the fast version.

## Blockers
- The nvcc PATH issue from 08-30 is still not solved. We worked around it instead of fixing it.
- We do not have GPU VRAM peak numbers or scale accuracy numbers for this run yet.
- The viewer's default settings (single frame mode, tiny point size) made the point cloud look empty at first. This cost debugging time right before the presentation.

## Next
- Take clean screenshots from a few different angles (a wide view and a close up on the vegetation and road) for the SIH slides.
- Fill in the problem statement's "Desired Output" and "Evaluation Criteria" table using today's real numbers (0.16 seconds per frame, clear separation between terrain, vegetation, and road).
- After the submission, go back and fix the nvcc issue so the compiled CUDA RoPE kernel can run, to get a real-time speed number.
- Try a lower `vis_threshold` to check if the lighter, washed-out terrain area is just noise or if it holds usable data.
