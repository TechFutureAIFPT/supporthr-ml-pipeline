# ML Pipeline Workspace

Thu muc `ml_pipeline/` da duoc tach rieng khoi backend de ban co the:

- day len GitHub nhu mot repo doc lap
- clone thang sang Colab
- train model va lay file `.pkl` ma khong phu thuoc vao `BE`

## Cau truc sau khi tach

```text
website/
|- BE/
|- FE/
`- ml_pipeline/
   |- artifacts/
   |- data/
   |  `- raw/
   |- scripts/
   |  |- seed_exemplars.py
   |  `- train_classifier.py
   |- README.md
   |- requirements.txt
   `- requirements-colab.txt
```

## GitHub repo cho ml_pipeline

Neu ban muon day thu muc nay len mot repo rieng:

```bash
cd /path/to/ml_pipeline
git init -b main
git add .
git commit -m "Initial ml_pipeline workspace"
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

Luu y:

- `data/raw/` dang duoc phep track de ban clone repo tren Colab la train duoc ngay.
- `artifacts/` dang bi ignore vi day la output sinh ra sau khi train.
- Repo co kha nhieu file data, vi vay day bang Git local hoac GitHub Desktop se on dinh hon upload tay qua web.

## Colab nhanh nhat tu GitHub

Sau khi repo da len GitHub, trong Colab chay:

```python
!git clone https://github.com/<your-username>/<your-repo>.git
%cd /content/<your-repo>
!pip install -r requirements-colab.txt
!python scripts/train_classifier.py --download-artifacts
```

Lenh tren se:

- tu tim dataset trong `data/raw`
- train model
- xuat artifact vao `artifacts/`
- download file ve may neu dang chay trong Colab

## Luu artifact vao Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

```python
%cd /content/<your-repo>
!pip install -r requirements-colab.txt
!python scripts/train_classifier.py \
  --copy-output-dir "/content/drive/MyDrive/supporthr-artifacts"
```

## Neu ban muon train bang zip hoac CSV rieng

Bang file zip:

```python
!python scripts/train_classifier.py \
  --dataset-zip "/content/resume_dataset.zip" \
  --force-extract \
  --download-artifacts
```

Bang CSV rieng:

```python
!python scripts/train_classifier.py \
  --dataset-csv "/content/Resume.csv" \
  --text-column "Resume_str" \
  --label-column "Category" \
  --download-artifacts
```

Neu CSV cua ban dung ten cot khac, chi can doi `--text-column` va `--label-column`.

## Tuy chon huu ich

- `--classifier-type linear_svm`: doi loai classifier
- `--sample-per-label 200`: train nhanh de smoke test
- `--max-features 12000`: tang so feature TF-IDF
- `--min-df 2`: nguong xuat hien toi thieu cua term
- `--model-output`: doi duong dan file `.pkl`
- `--report-output`: doi duong dan report text
- `--metadata-output`: doi duong dan metadata json
- `--copy-output-dir`: copy artifact sang folder khac

## Output sau khi train

- `artifacts/text_classifier_model.pkl`
- `artifacts/classification_report.txt`
- `artifacts/training_metadata.json`

## Cai thu vien

Cho Colab:

```bash
pip install -r requirements-colab.txt
```

Cho local day du:

```bash
pip install -r requirements.txt
```
