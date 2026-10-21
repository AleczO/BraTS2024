# Data

Notes on the BraTS 2024 data used in this project. Raw data and caches live in
`data/`, which is ignored by git. See the main README for download instructions.

## Source

- BraTS 2024 Adult Glioma Post-Treatment (BraTS-GLI), downloaded from Synapse
  folder `syn59059776` with `scripts/download.py`.
- The official citation list is shipped with the data as `raw/CITATIONS.bib`.

## Expected layout

```
data/
├── raw/
│   ├── training_data1_v2/          # main training set, with labels
│   ├── training_data_additional/   # additional training set, with labels
│   └── validation_data/            # challenge validation set, NO labels
└── cache/                          # preprocessing cache (created by the code)
```

Case folders are named `BraTS-GLI-XXXXX-YYY`, where `XXXXX` is the patient ID and
`YYY` identifies the scan (timepoint). Each folder contains `*-t1n`, `*-t1c`, `*-t2w`,
`*-t2f` and, for labelled cases, `*-seg` NIfTI files.

## Dataset summary (checked on 2026-10-21)

| Folder                     | Cases | With labels |
|----------------------------|------:|------------:|
| `training_data1_v2`        |  1350 |        1350 |
| `training_data_additional` |   271 |         271 |
| `validation_data`          |   188 |           0 |

- Labelled cases in total: **1621**, from **731 unique patients**.
- Patients shared between `training_data1_v2` and `training_data_additional`: **0**.
- `validation_data` has no ground truth and is not used for evaluation in this project.

## Implications for the pipeline

- Many patients have more than one scan, so **train/val/test splits must be made by
  patient ID**, never by scan. Splits are built once from both labelled folders together
  and saved to `splits/`.
- Label convention (BraTS 2024): 1 = NETC, 2 = SNFH, 3 = ET, 4 = RC (resection cavity).
  Evaluation regions: ET = 3, TC = 1 + 3, WT = 1 + 2 + 3, plus RC = 4.
  MONAI's `ConvertToMultiChannelBasedOnBratsClasses` defaults to `et_label=4`
  (BraTS 2018–2022 convention) and must be called with `et_label=3`.


The `data/` directory is not part of the repository; it is created by
`scripts/download.py` on first run.

## Reproducing the counts

```bash
cd data/raw
for d in */; do
  echo "$d  seg: $(find "$d" -name '*seg.nii.gz' | wc -l)  cases: $(find "$d" -name '*t1c.nii.gz' | wc -l)"
done
find . -name '*seg.nii.gz' | grep -oE 'BraTS-GLI-[0-9]{5}' | sort -u | wc -l
```



