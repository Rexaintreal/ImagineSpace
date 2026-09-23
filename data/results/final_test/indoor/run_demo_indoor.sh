#!/usr/bin/env bash
# ============================================================
# Imagine Space — INDOOR demo runner (presenter-paced)
# Video preview stage removed (shown manually beforehand).
# You control pacing with Enter after you're done showing the
# live point cloud viewer.
# ============================================================
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cut3r

FINAL_TEST_DIR="$HOME/final_test/indoor"
VIDEO_SRC="$FINAL_TEST_DIR/indoor_sample.mp4"
FRAMES_DIR="$FINAL_TEST_DIR/frames"
OUTPUT_DIR="$FINAL_TEST_DIR/output"
MESH_OUT_DIR="$FINAL_TEST_DIR/mesh_out"
CUT3R_DIR="$HOME/CUT3R"
FPS=5
VISER_URL="http://localhost:8080"

banner() {
  echo ""
  echo "============================================================"
  echo "  $1"
  echo "============================================================"
  sleep 1
}

banner "STAGE 1 / 4 — Frame extraction from source video (ffmpeg)"
mkdir -p "$FRAMES_DIR"
rm -f "$FRAMES_DIR"/*.jpg 2>/dev/null || true
ffmpeg -y -i "$VIDEO_SRC" -vf fps=$FPS "$FRAMES_DIR/frame_%04d.jpg"
echo ""
echo ">> Extracted $(ls "$FRAMES_DIR" | wc -l) frames at ${FPS} fps into $FRAMES_DIR"

banner "STAGE 2 / 4 — CUT3R inference + live point cloud viewer"
cd "$CUT3R_DIR"
python demo.py \
  --model_path src/cut3r_224_linear_4.pth \
  --size 224 \
  --seq_path "$FRAMES_DIR" \
  --vis_threshold 1.5 \
  --output_dir "$OUTPUT_DIR" &
DEMO_PID=$!

echo ">> Inference is running in the background."
echo ">> Waiting for the viser server to come up before opening the viewer..."
sleep 3
if command -v explorer.exe >/dev/null 2>&1; then
  explorer.exe "$VISER_URL" 2>/dev/null || true
elif command -v wslview >/dev/null 2>&1; then
  wslview "$VISER_URL" 2>/dev/null || true
else
  xdg-open "$VISER_URL" 2>/dev/null || true
fi
echo ">> Viewer should now be open in your browser at $VISER_URL"
echo ">> If nothing opened, open that URL manually in your Windows browser."
echo ">> IMPORTANT: wait until you see '(viser) Connection opened' below before"
echo ">>            pressing Enter — that confirms output has actually been written."
echo ">>            Ending it earlier will corrupt this run's output (as happened before)."
read -p ">> Press Enter ONLY after you've seen the viewer connect and shown it live..." _
kill "$DEMO_PID" 2>/dev/null || true
wait "$DEMO_PID" 2>/dev/null || true
echo ""
echo ">> CUT3R stage ended. Output should be in $OUTPUT_DIR"

banner "STAGE 3 / 4 — Merging point cloud + Poisson mesh reconstruction"
cd "$FINAL_TEST_DIR"
python generate_mesh.py \
  --output_dir "$OUTPUT_DIR" \
  --conf_thresh 3.0 \
  --voxel_size 0.5 \
  --poisson_depth 9 \
  --density_trim_quantile 0.15 \
  --out_pointcloud "$MESH_OUT_DIR/pointcloud.ply" \
  --out_mesh "$MESH_OUT_DIR/mesh.ply" \
  --screenshot_pointcloud "$MESH_OUT_DIR/pointcloud_view.png" \
  --screenshot_mesh "$MESH_OUT_DIR/mesh_view.png"
echo ""
echo ">> Point cloud + mesh written to $MESH_OUT_DIR"

banner "STAGE 4 / 5 — Result screenshots"
echo "Point cloud screenshot: $MESH_OUT_DIR/pointcloud_view.png"
echo "Mesh screenshot:        $MESH_OUT_DIR/mesh_view.png"

banner "STAGE 5 / 5 — Opening interactive 3D viewer (browser — mesh + point cloud)"
echo ">> Native Open3D windows don't render reliably under WSL (WSLg's OpenGL passthrough"
echo ">> can't satisfy Open3D's EGL/Zink context request, confirmed by the 09-22 log)."
echo ">> Using a browser-based WebGL viewer instead — it runs through your Windows GPU,"
echo ">> the same reason the viser point cloud viewer worked earlier."
VIEWER_PORT=8091

cat > "$MESH_OUT_DIR/viewer.html" <<'HTMLEOF'
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>CUT3R 3D Viewer</title>
<style>
  html, body { margin: 0; height: 100%; background: #111318; overflow: hidden; font-family: sans-serif; }
  canvas { display: block; }
  #panel {
    position: fixed; top: 10px; left: 10px; color: #cdd; font-size: 13px;
    background: rgba(20,22,28,0.7); padding: 10px 14px; border-radius: 6px; line-height: 1.6;
  }
  #panel label { display: block; cursor: pointer; }
  #hint { opacity: 0.7; margin-top: 6px; }
</style>
</head>
<body>
<div id="panel">
  <label><input type="checkbox" id="toggleMesh" checked> Mesh</label>
  <label><input type="checkbox" id="togglePoints"> Point cloud</label>
  <div id="hint">Left-drag: rotate | Scroll: zoom | Right-drag: pan</div>
</div>
<script type="importmap">
{ "imports": {
  "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
  "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
} }
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111318);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.01, 5000);
camera.position.set(0, 0, 3);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const dl = new THREE.DirectionalLight(0xffffff, 0.9);
dl.position.set(1, 1, 1);
scene.add(dl);

let meshObj = null;
let pointsObj = null;
let boundsSet = false;

function fitCameraToObject(object) {
  if (boundsSet) return;
  const box = new THREE.Box3().setFromObject(object);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const radius = Math.max(size.x, size.y, size.z) * 0.6 || 1;
  object.position.sub(center);
  camera.position.set(0, 0, radius * 2.5);
  camera.far = radius * 20;
  camera.updateProjectionMatrix();
  controls.target.set(0, 0, 0);
  boundsSet = true;
}

const loader = new PLYLoader();

loader.load('mesh.ply', (geometry) => {
  geometry.computeVertexNormals();
  const hasColor = !!geometry.attributes.color;
  const material = new THREE.MeshStandardMaterial({
    vertexColors: hasColor,
    color: hasColor ? 0xffffff : 0x8ca0c8,
    side: THREE.DoubleSide,
  });
  meshObj = new THREE.Mesh(geometry, material);
  fitCameraToObject(meshObj);
  scene.add(meshObj);
}, undefined, (err) => console.error('mesh load failed', err));

loader.load('pointcloud.ply', (geometry) => {
  const hasColor = !!geometry.attributes.color;
  const material = new THREE.PointsMaterial({
    size: 0.02,
    vertexColors: hasColor,
    color: hasColor ? 0xffffff : 0xff6b6b,
  });
  pointsObj = new THREE.Points(geometry, material);
  fitCameraToObject(pointsObj);
  pointsObj.visible = document.getElementById('togglePoints').checked;
  scene.add(pointsObj);
}, undefined, (err) => console.error('pointcloud load failed', err));

document.getElementById('toggleMesh').addEventListener('change', (e) => {
  if (meshObj) meshObj.visible = e.target.checked;
});
document.getElementById('togglePoints').addEventListener('change', (e) => {
  if (pointsObj) pointsObj.visible = e.target.checked;
});

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

(function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
})();
</script>
</body>
</html>
HTMLEOF

(cd "$MESH_OUT_DIR" && python3 -m http.server "$VIEWER_PORT" >/dev/null 2>&1) &
VIEWER_SERVER_PID=$!
sleep 1
VIEWER_URL="http://localhost:$VIEWER_PORT/viewer.html"
if command -v explorer.exe >/dev/null 2>&1; then
  # WSL: launches the real Windows browser directly, bypassing xdg-open (no desktop portal in WSL)
  explorer.exe "$VIEWER_URL" 2>/dev/null || true
elif command -v wslview >/dev/null 2>&1; then
  wslview "$VIEWER_URL" 2>/dev/null || true
else
  xdg-open "$VIEWER_URL" 2>/dev/null || true
fi
echo ">> If nothing opened automatically, open this URL manually in your Windows browser: $VIEWER_URL"

echo ">> Viewer running — use the checkboxes top-left to toggle mesh / point cloud."
read -p ">> Press Enter here once you're done viewing to shut down the local server..." _
kill "$VIEWER_SERVER_PID" 2>/dev/null || true
wait "$VIEWER_SERVER_PID" 2>/dev/null || true

banner "INDOOR PIPELINE COMPLETE"
