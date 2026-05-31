# Zenodo Deposit — Step-by-Step

This file walks you through depositing the trained model predictions to Zenodo
so you can reference a stable DOI in the paper.

---

## What to upload

**One zip file** containing the gitignored prediction data:

```
data/2026/ml/
├── README_deposit.md          ← already written, include this
└── Ef/
    ├── iiLoss_0.0/ml_input.json
    ├── iiLoss_0.1/ml_input.json
    ...  (40 model directories, ~132 MB total, ~30 MB zipped)
```

**Create the zip on your machine** (from the repo root):
```bash
# Windows PowerShell
Compress-Archive -Path data\2026\ml -DestinationPath ml_predictions_2026.zip

# macOS / Linux
zip -r ml_predictions_2026.zip data/2026/ml/
```

---

## Step 1 — Create a Zenodo account

Go to [https://zenodo.org](https://zenodo.org) and sign in with ORCID, GitHub,
or email. Using ORCID links the deposit to your researcher profile automatically.

---

## Step 2 — Start a new upload

Click **"+ New Upload"** (top-right menu).

---

## Step 3 — Upload the file

Drag `ml_predictions_2026.zip` into the upload box, or use "Choose files".
Wait for the upload to complete (progress bar).

---

## Step 4 — Fill in the metadata

| Field | Value |
|---|---|
| **Upload type** | Dataset |
| **Title** | ML Formation Energy Predictions for the Interference Index (2026 MP Snapshot) |
| **Authors** | [your name(s) and affiliations] |
| **Description** | Trained model predictions (Hf in eV/atom) for 40 iiLoss and HdLoss model variants on the 2026 Materials Project snapshot, used to compute interference scores in [paper title]. See README_deposit.md inside the zip for full details. |
| **License** | Creative Commons Attribution 4.0 International (CC-BY-4.0) |
| **Keywords** | machine learning, materials science, formation energy, interference index, Materials Project |
| **Related identifiers** | Add the GitHub repo URL under "is supplemented by": `https://github.com/KanzowA/InterferenceIndex` |
| **Version** | 1.0.0 |

Leave "Access" as **Open** unless your institution requires an embargo.

---

## Step 5 — Reserve a DOI (optional but recommended)

Click **"Reserve DOI"** before publishing. This gives you the DOI immediately
so you can put it in the paper before the deposit is finalised.

Copy the DOI (format: `10.5281/zenodo.XXXXXXX`).

---

## Step 6 — Publish

Click **"Publish"**. The record is now permanently archived and the DOI is active.

---

## Step 7 — Update the repository

Once you have the DOI, update two places:

**`README.md`** — replace the placeholder in the data section:
```markdown
├── data/                        # Materials Project hull data
│                                # 2026 ML predictions: https://doi.org/10.5281/zenodo.XXXXXXX
```

And in the **Data availability** sentence:
```
The trained model predictions are available at https://doi.org/10.5281/zenodo.XXXXXXX.
```

**`.gitignore`** — the DOI comment at the top of the data/2026/ml/ block:
```
# data/2026/ml/ — gitignored; available at https://doi.org/10.5281/zenodo.XXXXXXX
data/2026/ml/
```

---

## Checklist

- [ ] Zip created (`ml_predictions_2026.zip`)
- [ ] File uploaded to Zenodo
- [ ] Metadata filled in
- [ ] DOI reserved / noted
- [ ] Deposit published
- [ ] README.md updated with DOI
- [ ] .gitignore comment updated with DOI
- [ ] DOI added to paper's Data Availability section
