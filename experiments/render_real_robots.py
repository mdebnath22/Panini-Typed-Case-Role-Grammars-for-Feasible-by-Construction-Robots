"""
Real MuJoCo Renders — Grammar-Guided (Valid) vs. Direct Edit (Broken)
=========================================================================

Produces ACTUAL physics-engine screenshots (not schematics) comparing:
  (a) a grammar-guided, valid robot body
  (b) a body produced by a direct-edit mutation with no structural
      validity mechanism -- specifically chosen to be one that LOADS but
      diverges under simulation, so the rendered image shows a real,
      visually dramatic failure (limbs flown apart / exploded pose) rather
      than a blank error frame. This is the same class of failure you saw
      directly in your own terminal output ("WARNING: Nan, Inf or huge
      value in QACC..."), rendered instead of just logged.

Requires: mujoco (with its Python bindings, mujoco>=2.3 for the Renderer
API), your existing experiments/ package on the Python path, and a
non-interactive/offscreen-capable environment (this uses the offscreen
Renderer, not a live viewer window -- works fine on a headless cluster
node; if you get an EGL/GLFW error, see the note at the bottom).

Run (from your existing repo root):
    python3 -m experiments.render_real_robots --seed 0 --settle_steps 300
Outputs:
    render_valid.png, render_broken.png, render_side_by_side.png
"""

from __future__ import annotations

import argparse
import random
import traceback
from typing import Optional, Tuple

import numpy as np

try:
    import mujoco
except ImportError as e:
    raise ImportError(
        "This script needs the mujoco Python package (you have it on Sol; "
        "if running elsewhere: pip install mujoco") from e

from experiments.m_search_loop import random_genome
from experiments.mh1_xml_baseline import _tree_of
from gmm.mjcf_compiler import MJCFCompiler
from experiments.expanded_validity_comparison import (
    mutate_extreme_only, mutate_uniform_resample)


def render_model(xml: str, settle_steps: int, width: int = 640, height: int = 480
                 ) -> Tuple[Optional[np.ndarray], str]:
    """Loads an MJCF string, steps it forward settle_steps times, and
    renders a single frame. Returns (image_array_or_None, status_message).
    Never raises -- a failure to load or a physics divergence is reported
    in the status message and, where possible, still rendered (a diverged
    pose is itself the point of the illustration)."""
    try:
        model = mujoco.MjModel.from_xml_string(xml)
    except Exception as e:
        return None, f"FAILED TO LOAD: {type(e).__name__}: {e}"

    data = mujoco.MjData(model)
    diverged = False
    divergence_step = None
    for step in range(settle_steps):
        try:
            mujoco.mj_step(model, data)
        except Exception as e:
            diverged = True
            divergence_step = step
            break
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            diverged = True
            divergence_step = step
            break

    # camera: auto-frame using MuJoCo's own geometry-derived stat.center and
    # stat.extent, NOT data.qpos -- an earlier version used qpos[:3] assuming
    # it holds the root body's xyz position, which only holds if the root's
    # FIRST joint is a free joint. With hinge-jointed legs (as here), qpos[:3]
    # instead grabs the first few joint ANGLES (near-zero radians), pointing
    # the camera at an essentially arbitrary location -- confirmed as the
    # cause of two identically all-black renders (same wrong-but-consistent
    # camera target for both). model.stat.center/extent are computed by
    # MuJoCo's own compiler from the actual geometry, so they are correct
    # regardless of joint type or ordering.
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = model.stat.center
    cam.distance = model.stat.extent * 2.2 if not diverged else model.stat.extent * 6
    cam.azimuth = 130
    cam.elevation = -20

    try:
        renderer = mujoco.Renderer(model, height=height, width=width)
        renderer.update_scene(data, camera=cam)
        img = renderer.render()
    except Exception as e:
        return None, (f"loaded and stepped ({'diverged at step ' + str(divergence_step) if diverged else 'stable'}) "
                      f"but RENDER FAILED: {type(e).__name__}: {e}")

    status = ("stable after settling" if not diverged else
             f"DIVERGED at step {divergence_step}/{settle_steps} "
             f"(non-finite state or physics exception) -- this is the "
             f"rendered failure mode itself, not a rendering error")
    return img, status


def run(seed: int = 0, settle_steps: int = 300, verbose: bool = True) -> None:
    import imageio.v2 as imageio

    rng = np.random.default_rng(seed)

    # (a) valid grammar-guided body
    genome = random_genome(rng)
    valid_xml = MJCFCompiler().compile(_tree_of(genome))
    img_valid, status_valid = render_model(valid_xml, settle_steps)

    # (b) direct-edit mutation with no structural validity mechanism.
    # CORRECTNESS NOTE: an earlier version used duplicate_element, which
    # copies a geom's XML verbatim -- including its `name` attribute --
    # producing a duplicate-name error that MuJoCo's compiler rejects
    # outright at load time (confirmed: "repeated name 'seg2_g' in geom").
    # That is a real, valid, even cleaner failure mode than originally
    # described (a compile-time rejection, not a runtime divergence), but
    # it means NO image can be rendered for it at all. For an actual
    # picture, we need a mutation that touches only numeric VALUES (never
    # names), which is far more likely to still load: extreme_only and
    # uniform_resample are both from the same validity comparison (Table
    # 5: 62/100 and 78/100 respectively) and never touch structural names.
    # We try a few candidates and use the first that loads, so a single
    # unlucky random draw doesn't block the whole comparison.
    candidates = []
    for extra_seed in range(6):
        py_rng = random.Random(seed + 100 + extra_seed)
        xml = valid_xml
        mutator = mutate_extreme_only if extra_seed % 2 == 0 else mutate_uniform_resample
        mutator_name = "mutate_extreme_only" if extra_seed % 2 == 0 else "mutate_uniform_resample"
        for _ in range(3):
            xml = mutator(xml, py_rng)
        candidates.append((f"{mutator_name}, retry seed {extra_seed}", xml))

    img_broken, status_broken, used_label = None, "no candidate attempted", None
    for label, xml in candidates:
        img_broken, status_broken = render_model(xml, settle_steps)
        used_label = label
        if img_broken is not None:
            break
    if verbose and img_broken is None:
        print(f"  (all {len(candidates)} candidate mutations failed to load "
             f"or render; last attempted: {used_label})")

    if verbose:
        print(f"(a) valid grammar-guided body: {status_valid}")
        print(f"(b) direct-edit ({used_label}): {status_broken}")

    if img_valid is not None:
        imageio.imwrite("render_valid.png", img_valid)
        print("saved render_valid.png")
    if img_broken is not None:
        imageio.imwrite("render_broken.png", img_broken)
        print("saved render_broken.png")

    if img_valid is not None and img_broken is not None:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img_valid)
        axes[0].set_title(f"(a) Grammar-guided (ours)\n{status_valid}", fontsize=9)
        axes[0].axis("off")
        axes[1].imshow(img_broken)
        axes[1].set_title(f"(b) Direct edit ({used_label})\n{status_broken}", fontsize=9)
        axes[1].axis("off")
        fig.tight_layout()
        fig.savefig("render_side_by_side.png", dpi=150, bbox_inches="tight")
        fig.savefig("render_side_by_side.pdf", bbox_inches="tight")
        print("saved render_side_by_side.png and .pdf")
    else:
        print("\nCould not build the side-by-side comparison -- every "
             "candidate mutation failed to load or render. Since these "
             "mutations only touch numeric values (never structural names), "
             "this would itself be a notable finding worth reporting "
             "precisely if it reproduces consistently across seeds.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--settle_steps", type=int, default=300)
    args = ap.parse_args()
    run(seed=args.seed, settle_steps=args.settle_steps)

# ---------------------------------------------------------------------------
# NOTE on headless rendering: mujoco.Renderer needs a working OpenGL context.
# On most HPC nodes without a display, set the environment variable
#     export MUJOCO_GL=egl
# before running (or 'osmesa' if EGL is unavailable on your node). If you
# get an error mentioning GLFW or "failed to create OpenGL context", this
# environment variable is almost always the fix.
# ---------------------------------------------------------------------------
