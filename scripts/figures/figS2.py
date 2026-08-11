"""Figure S2: the same convergence plot as Figure S1, for HdLoss.

Panel b shows the decomposition-enthalpy penalty instead of the interference
term, so the two figures are comparable in shape but not on that axis.

Usage
-----
    python scripts/figures/figS2.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import figS1

if __name__ == "__main__":
    sys.argv[1:] = ["--prefix", "HdLoss_finetune_",
                    "--out", os.path.join(figS1.REPO_ROOT, "figures",
                                          "figureS2.png")] + sys.argv[1:]
    figS1.main()
