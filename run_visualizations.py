"""
One-click: generate all 3D visualizations and open in browser.

Usage:
    python run_visualizations.py
"""

import sys
import webbrowser
from pathlib import Path

# Setup path
_project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_project_root))

from src.visualization.interactive_3d import generate_all_visualizations

print("Generating 3D interactive visualizations...")
print("This takes ~30 seconds.\n")

paths = generate_all_visualizations(
    output_dir=str(_project_root / "data" / "visualizations")
)

print("\n" + "=" * 50)
print("Opening visualizations in your browser...")
print("=" * 50)

# Open the most interesting ones
for p in paths:
    abs_path = Path(p).resolve()
    url = abs_path.as_uri()
    print(f"  Opening: {abs_path.name}")
    webbrowser.open(url)

print(f"""
All {len(paths)} visualizations saved to:
  {_project_root / 'data' / 'visualizations'}

What you're looking at:

  01_cell_landscape_responder.html
     Left panel: 3000+ CAR-T cells in 3D, colored by state
       (green=naive, red=effector, blue=memory, brown=exhausted)
     Right panel: same cells colored by exhaustion score
       (blue=healthy, red=exhausted)
     >> Drag to rotate, scroll to zoom, hover for cell info

  02_responder_comparison.html
     Side-by-side: Responder (left) vs Non-Responder (right)
     >> Notice: non-responder has a dense red/orange exhaustion
        cluster that's SEPARATED from the rest - this is the
        topological signature TDA detects

  03_persistence_diagram_3d.html
     Birth vs Death vs Dimension
     >> Blue dots = H0 (components), Orange = H1 (loops)
     >> Bigger dots = more persistent (longer-lived) features
     >> Points far from diagonal = significant topology

  04_mapper_graph_responder.html
     Mapper graph: each sphere = cluster of cells, size = cell count
     >> Color = exhaustion (blue=low, red=high)
     >> Notice: BRANCHING structure with connected arms

  05_mapper_graph_non_responder.html
     Same but for non-responder
     >> Notice: MORE FRAGMENTED, fewer connections, red nodes isolated

  06_timecourse_animation.html
     Press PLAY or drag slider to watch CAR-T cells evolve
     from pre-infusion through 6 months of treatment
     >> Watch how the landscape shifts from naive (clustered)
        to diverse (spread) to memory-dominated (concentrated)
""")
