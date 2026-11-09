"""
Schematic Robot Comparison — Valid (Grammar-Guided) vs. Invalid (Direct Edit)
=================================================================================

IMPORTANT HONESTY NOTE: this is a schematic top-down technical illustration
drawn from the REAL default parameters in Appendix A of the paper (torso
0.30x0.15m, 4 legs, 2 segments/leg at 0.12m each, segment radius 0.02m,
foot radius 0.015m) and a REAL, documented failure mode from the validity
comparison (duplicate_element, which scored 0/100 in Table 5 -- it
duplicates a compiled geometry element immediately after itself, producing
two perfectly overlapping, colliding geoms). This is NOT a MuJoCo physics
render -- see render_real_robots.py for a script that produces an actual
physics-engine screenshot on a machine with MuJoCo installed. This script
only needs matplotlib + numpy.

Run:
    python3 schematic_robots.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Real default parameters, Appendix A
TORSO_L, TORSO_W = 0.30, 0.15
N_LEGS = 4
SEG_LEN = 0.12
SEG_RAD = 0.02
FOOT_RAD = 0.015
N_SEGMENTS = 2


def draw_leg(ax, mount_angle, color="#1f77b4", offset=(0, 0), duplicate_seg=False):
    """Draws one leg (N_SEGMENTS capsule segments radiating outward from the
    torso edge at mount_angle) plus a foot. If duplicate_seg, draws an
    identical extra copy of the LAST segment exactly on top of itself,
    illustrating the real duplicate_element failure mode."""
    ox, oy = offset
    x, y = ox + (TORSO_L / 2) * np.cos(mount_angle), oy + (TORSO_L / 2) * np.sin(mount_angle)
    for seg in range(N_SEGMENTS):
        x2 = x + SEG_LEN * np.cos(mount_angle)
        y2 = y + SEG_LEN * np.sin(mount_angle)
        ax.plot([x, x2], [y, y2], color=color, linewidth=SEG_RAD * 400, solid_capstyle="round",
                zorder=2, alpha=0.9)
        if duplicate_seg and seg == N_SEGMENTS - 1:
            # the real failure: an IDENTICAL geom duplicated at the exact
            # same position, drawn here in red with a dashed outline to
            # make the overlap visible rather than literally invisible
            ax.plot([x, x2], [y, y2], color="red", linewidth=SEG_RAD * 400 + 3,
                    solid_capstyle="round", zorder=1, alpha=0.5, linestyle=(0, (2, 2)))
        x, y = x2, y2
    foot = patches.Circle((x, y), FOOT_RAD, color="#333333", zorder=3)
    ax.add_patch(foot)


def draw_robot(ax, title, broken_leg_idx=None):
    torso = patches.FancyBboxPatch((-TORSO_L / 2, -TORSO_W / 2), TORSO_L, TORSO_W,
                                   boxstyle="round,pad=0,rounding_size=0.02",
                                   facecolor="#888888", edgecolor="black", zorder=4)
    ax.add_patch(torso)
    angles = np.linspace(0, 2 * np.pi, N_LEGS, endpoint=False) + np.pi / 4
    for i, angle in enumerate(angles):
        is_broken = (broken_leg_idx is not None and i == broken_leg_idx)
        draw_leg(ax, angle, duplicate_seg=is_broken,
                color="#d62728" if is_broken else "#1f77b4")
    ax.set_xlim(-0.45, 0.45)
    ax.set_ylim(-0.45, 0.45)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.axis("off")


fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

draw_robot(axes[0], "(a) Grammar-guided (ours)\nvalid: 100/100 trials")
draw_robot(axes[1], "(b) Direct edit: duplicate\\_element\ninvalid: 0/100 trials",
          broken_leg_idx=1)

fig.suptitle("Schematic illustration, not a physics render \u2014 see render_real_robots.py "
            "for an actual MuJoCo screenshot", fontsize=8.5, y=0.02, color="#555555")
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig("schematic_valid_vs_invalid.pdf", bbox_inches="tight")
fig.savefig("schematic_valid_vs_invalid.png", bbox_inches="tight", dpi=150)
print("saved schematic_valid_vs_invalid.pdf and .png")
