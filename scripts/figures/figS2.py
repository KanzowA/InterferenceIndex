"""Figure S2: the convergence plot of Figure S1 for HdLoss (lambda = 0.1).

Panel b shows the decomposition-enthalpy term instead of the interference
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
    sys.argv[1:] = ["--objective", "HdLoss"] + sys.argv[1:]
    figS1.main()
