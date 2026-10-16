"""DRAFT, untested: run Microsoft TRELLIS (image -> 3D .glb) on Modal.

Usage (after `pip install modal` and `modal setup`):
    modal run modal_trellis.py --image-path ./input.png --out ./out.glb

Not verified against a live Modal account or GPU. The image-build section is the part
most likely to need fixing: TRELLIS has heavy CUDA deps (spconv, kaolin, nvdiffrast,
diffoctreerast). Check https://github.com/microsoft/TRELLIS (setup.sh / README) and
align the versions below. TRELLIS.2 is a newer model with a different API.
"""

import modal

app = modal.App("trellis-image-to-3d")

CUDA_TAG = "12.1.1-devel-ubuntu22.04"  # devel image: needed to compile CUDA extensions

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "build-essential", "ninja-build")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "numpy<2", "pillow", "imageio", "imageio-ffmpeg", "tqdm", "easydict",
        "opencv-python-headless", "scipy", "ninja", "rembg", "onnxruntime",
        "trimesh", "open3d", "xatlas", "pyvista", "pymeshfix", "igraph",
        "transformers", "huggingface_hub", "xformers==0.0.27.post2",
        "spconv-cu120",
        "utils3d @ git+https://github.com/EasternJournalist/utils3d.git",
    )
    # TODO: flash-attn, kaolin, nvdiffrast, diffoctreerast per TRELLIS setup.sh
    # (all need a CUDA build step; add with .run_commands(...) once versions are confirmed)
    .run_commands("git clone --depth 1 https://github.com/microsoft/TRELLIS.git /opt/TRELLIS")
    .env(
        {
            "PYTHONPATH": "/opt/TRELLIS",
            "SPCONV_ALGO": "native",
            "ATTN_BACKEND": "xformers",  # avoids requiring flash-attn
            "HF_HOME": "/cache/hf",
        }
    )
)

# Persist downloaded weights so cold starts don't re-download them.
cache = modal.Volume.from_name("trellis-hf-cache", create_if_missing=True)


@app.cls(
    image=image,
    gpu="L40S",  # 48GB; an A10G/24GB card may also be enough (cheaper)
    volumes={"/cache": cache},
    timeout=15 * 60,
    scaledown_window=120,  # keep warm 2 min; older Modal versions call this container_idle_timeout
)
class Trellis:
    @modal.enter()
    def load(self):
        from trellis.pipelines import TrellisImageTo3DPipeline

        self.pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
        self.pipeline.cuda()

    @modal.method()
    def generate(self, image_bytes: bytes, seed: int = 1) -> bytes:
        import io

        from PIL import Image
        from trellis.utils import postprocessing_utils

        img = Image.open(io.BytesIO(image_bytes))
        outputs = self.pipeline.run(img, seed=seed)
        glb = postprocessing_utils.to_glb(
            outputs["gaussian"][0],
            outputs["mesh"][0],
            simplify=0.95,
            texture_size=1024,
        )
        buf = io.BytesIO()
        glb.export(buf, file_type="glb")
        return buf.getvalue()


@app.local_entrypoint()
def main(image_path: str, out: str = "out.glb", seed: int = 1):
    with open(image_path, "rb") as f:
        data = f.read()
    glb_bytes = Trellis().generate.remote(data, seed)
    with open(out, "wb") as f:
        f.write(glb_bytes)
    print(f"saved {out} ({len(glb_bytes) / 1e6:.1f} MB)")
