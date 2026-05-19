from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Iterable

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_FEATURES = 8000
MIN_DF = 2
MIN_SAMPLES_REQUIRED = 2
DEFAULT_TEXT_COLUMN = "Resume_str"
DEFAULT_LABEL_COLUMN = "Category"
DEFAULT_MODEL_NAME = "text_classifier_model.pkl"
DEFAULT_REPORT_NAME = "classification_report.txt"
DEFAULT_METADATA_NAME = "training_metadata.json"


def configure_console_output() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def clean_text(text: str) -> str:
    normalized = str(text).lower()
    normalized = re.sub(r"[^\w\s]+", " ", normalized, flags=re.UNICODE)
    normalized = normalized.replace("_", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a CV industry classifier and export a .pkl artifact."
    )
    parser.add_argument(
        "--dataset-csv",
        default=None,
        help="Optional absolute or relative path to a CSV dataset.",
    )
    parser.add_argument(
        "--text-dir",
        default=None,
        help="Optional absolute or relative path to a folder of label/*.txt samples.",
    )
    parser.add_argument(
        "--dataset-zip",
        default=None,
        help="Optional absolute or relative path to a zip file that contains Resume.csv or String_Folder.",
    )
    parser.add_argument(
        "--extract-dir",
        default=None,
        help="Optional folder used when extracting --dataset-zip.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Delete the previous extract directory before extracting the zip again.",
    )
    parser.add_argument(
        "--text-column",
        default=DEFAULT_TEXT_COLUMN,
        help=f"CSV column that contains resume text. Default: {DEFAULT_TEXT_COLUMN}",
    )
    parser.add_argument(
        "--label-column",
        default=DEFAULT_LABEL_COLUMN,
        help=f"CSV column that contains the class label. Default: {DEFAULT_LABEL_COLUMN}",
    )
    parser.add_argument(
        "--classifier-type",
        choices=("logreg", "linear_svm"),
        default="logreg",
        help="Classifier family to train. Default: logreg",
    )
    parser.add_argument(
        "--model-output",
        default=None,
        help="Optional absolute or relative path for the exported .pkl model.",
    )
    parser.add_argument(
        "--report-output",
        default=None,
        help="Optional absolute or relative path for the text classification report.",
    )
    parser.add_argument(
        "--metadata-output",
        default=None,
        help="Optional absolute or relative path for the JSON training metadata.",
    )
    parser.add_argument(
        "--copy-output-dir",
        default=None,
        help="Optional directory to copy the exported files into, for example a Google Drive folder.",
    )
    parser.add_argument(
        "--download-artifacts",
        action="store_true",
        help="If running on Colab, try to download the exported files to your local machine.",
    )
    parser.add_argument(
        "--sample-per-label",
        type=int,
        default=None,
        help="Optional cap for samples per label, useful for quick smoke tests on Colab.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help=f"Test split ratio. Default: {TEST_SIZE}",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=MAX_FEATURES,
        help=f"Maximum TF-IDF features. Default: {MAX_FEATURES}",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=MIN_DF,
        help=f"Minimum document frequency for TF-IDF terms. Default: {MIN_DF}",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help=f"Random seed. Default: {RANDOM_STATE}",
    )
    return parser.parse_args()


def iter_text_files(dataset_dir: str) -> Iterable[tuple[str, str]]:
    for label in sorted(os.listdir(dataset_dir)):
        label_dir = os.path.join(dataset_dir, label)
        if not os.path.isdir(label_dir):
            continue

        for file_name in sorted(os.listdir(label_dir)):
            if not file_name.lower().endswith(".txt"):
                continue

            file_path = os.path.join(label_dir, file_name)
            if not os.path.isfile(file_path):
                continue

            yield label, file_path


def validate_dataset(dataset: pd.DataFrame, skipped_items: int = 0) -> pd.DataFrame:
    if dataset.empty:
        raise ValueError("Dataset is empty after cleaning.")

    label_counts = dataset["label"].value_counts()
    invalid_labels = label_counts[label_counts < MIN_SAMPLES_REQUIRED]
    if not invalid_labels.empty:
        details = ", ".join(f"{label}={count}" for label, count in invalid_labels.items())
        raise ValueError(
            "Each class needs at least 2 samples for stratified train/test split. "
            f"Classes with too few samples: {details}"
        )

    print(f"Loaded {len(dataset)} samples across {dataset['label'].nunique()} labels.")
    if skipped_items:
        print(f"Skipped {skipped_items} unreadable or empty samples.")
    print("Label distribution:")
    print(label_counts.sort_index().to_string())
    return dataset


def load_dataset_from_text_folders(dataset_dir: str) -> pd.DataFrame:
    import pandas as pd

    records: list[dict[str, str]] = []
    skipped_files = 0

    for label, file_path in iter_text_files(dataset_dir):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                raw_text = file.read()
        except OSError as error:
            print(f"[WARN] Cannot read file: {file_path} -> {error}")
            skipped_files += 1
            continue

        cleaned = clean_text(raw_text)
        if not cleaned:
            skipped_files += 1
            continue

        records.append(
            {
                "label": label,
                "text": cleaned,
                "source_file": file_path,
            }
        )

    if not records:
        raise ValueError(
            "No valid .txt samples were loaded. "
            "Please verify that the dataset contains label folders with non-empty .txt files."
        )

    return validate_dataset(pd.DataFrame(records), skipped_items=skipped_files)


def load_dataset_from_csv(csv_path: str, *, text_column: str, label_column: str) -> pd.DataFrame:
    import pandas as pd

    try:
        dataset = pd.read_csv(
            csv_path,
            usecols=[text_column, label_column],
            encoding="utf-8",
            encoding_errors="ignore",
            on_bad_lines="skip",
        )
    except ValueError as error:
        raise ValueError(
            f"CSV file is missing required columns '{text_column}' and '{label_column}': {csv_path}"
        ) from error

    dataset = dataset.rename(columns={text_column: "text", label_column: "label"})
    dataset["text"] = dataset["text"].fillna("").map(clean_text)
    dataset["label"] = dataset["label"].fillna("").astype(str).str.strip()
    dataset = dataset[(dataset["text"] != "") & (dataset["label"] != "")]
    dataset = dataset.reset_index(drop=True)
    return validate_dataset(dataset)


def maybe_sample_dataset(dataset: pd.DataFrame, sample_per_label: int | None, *, random_state: int) -> pd.DataFrame:
    import pandas as pd

    if sample_per_label is None:
        return dataset
    if sample_per_label < MIN_SAMPLES_REQUIRED:
        raise ValueError("sample_per_label must be at least 2 when provided.")

    sampled_frames = [
        frame.sample(n=min(len(frame), sample_per_label), random_state=random_state)
        for _, frame in dataset.groupby("label", sort=True)
    ]
    sampled = pd.concat(sampled_frames, ignore_index=True)
    print(f"Applied sample_per_label={sample_per_label}. Remaining rows: {len(sampled)}")
    return validate_dataset(sampled)


def resolve_workspace_dir(script_dir: str) -> str:
    return os.path.abspath(os.path.join(script_dir, ".."))


def ensure_parent_dir(file_path: str) -> str:
    parent_dir = os.path.dirname(file_path)
    os.makedirs(parent_dir, exist_ok=True)
    return file_path


def ensure_dir(dir_path: str) -> str:
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def resolve_cli_path(base_dir: str, path_value: str | None) -> str | None:
    if not path_value:
        return None
    if os.path.isabs(path_value):
        return os.path.abspath(path_value)
    return os.path.abspath(os.path.join(base_dir, path_value))


def resolve_default_paths(workspace_dir: str) -> dict[str, str]:
    return {
        "csv_path": os.path.abspath(
            os.path.join(
                workspace_dir,
                "data",
                "raw",
                "kaggle-nlp-classification",
                "Resume",
                "Resume.csv",
            )
        ),
        "text_dir": os.path.abspath(
            os.path.join(workspace_dir, "data", "raw", "kaggle-nlp-classification", "String_Folder")
        ),
        "extract_dir": os.path.abspath(
            os.path.join(workspace_dir, "data", "raw", "extracted")
        ),
        "model_path": os.path.abspath(
            os.path.join(workspace_dir, "artifacts", DEFAULT_MODEL_NAME)
        ),
        "report_path": os.path.abspath(
            os.path.join(workspace_dir, "artifacts", DEFAULT_REPORT_NAME)
        ),
        "metadata_path": os.path.abspath(
            os.path.join(workspace_dir, "artifacts", DEFAULT_METADATA_NAME)
        ),
    }


def extract_zip_if_needed(zip_path: str, extract_dir: str, *, force_extract: bool) -> str:
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"Dataset zip was not found: {zip_path}")

    if force_extract and os.path.isdir(extract_dir):
        shutil.rmtree(extract_dir)

    ensure_dir(extract_dir)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir)

    return extract_dir


def _ranked_csv_candidates(root_dir: str) -> list[str]:
    root = Path(root_dir)
    candidates = [path for path in root.rglob("*.csv") if path.is_file()]
    ranked = sorted(
        candidates,
        key=lambda path: (
            0 if path.name.lower() == "resume.csv" else 1,
            0 if "resume" in path.name.lower() else 1,
            len(path.parts),
            str(path).lower(),
        ),
    )
    return [str(path.resolve()) for path in ranked]


def _ranked_text_dir_candidates(root_dir: str) -> list[str]:
    root = Path(root_dir)
    candidates = [path for path in root.rglob("*") if path.is_dir() and path.name.lower() == "string_folder"]
    ranked = sorted(candidates, key=lambda path: (len(path.parts), str(path).lower()))
    return [str(path.resolve()) for path in ranked]


def resolve_dataset_source(
    workspace_dir: str,
    *,
    dataset_csv: str | None = None,
    text_dir: str | None = None,
    dataset_zip: str | None = None,
    extract_dir: str | None = None,
    force_extract: bool = False,
) -> tuple[str, str]:
    defaults = resolve_default_paths(workspace_dir)
    explicit_csv = resolve_cli_path(workspace_dir, dataset_csv)
    if explicit_csv and os.path.isfile(explicit_csv):
        return "csv", explicit_csv

    explicit_text_dir = resolve_cli_path(workspace_dir, text_dir)
    if explicit_text_dir and os.path.isdir(explicit_text_dir):
        return "txt", explicit_text_dir

    search_roots: list[str] = []
    dataset_zip_path = resolve_cli_path(workspace_dir, dataset_zip)
    if dataset_zip_path:
        resolved_extract_dir = resolve_cli_path(workspace_dir, extract_dir) or defaults["extract_dir"]
        extracted_root = extract_zip_if_needed(
            dataset_zip_path,
            resolved_extract_dir,
            force_extract=force_extract,
        )
        search_roots.append(extracted_root)

    search_roots.extend(
        [
            workspace_dir,
            os.path.join(workspace_dir, "data"),
            os.path.join(workspace_dir, "data", "raw"),
        ]
    )

    seen_roots: set[str] = set()
    unique_search_roots: list[str] = []
    for root in search_roots:
        normalized = os.path.abspath(root)
        if normalized not in seen_roots and os.path.isdir(normalized):
            seen_roots.add(normalized)
            unique_search_roots.append(normalized)

    if os.path.isfile(defaults["csv_path"]):
        return "csv", defaults["csv_path"]
    if os.path.isdir(defaults["text_dir"]):
        return "txt", defaults["text_dir"]

    for root in unique_search_roots:
        csv_candidates = _ranked_csv_candidates(root)
        if csv_candidates:
            return "csv", csv_candidates[0]

        text_dir_candidates = _ranked_text_dir_candidates(root)
        if text_dir_candidates:
            return "txt", text_dir_candidates[0]

    raise FileNotFoundError(
        "No supported dataset source was found. "
        "Provide --dataset-csv, --text-dir, or --dataset-zip."
    )


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    classifier_step: object
    if args.classifier_type == "linear_svm":
        classifier_step = LinearSVC(
            class_weight="balanced",
            random_state=args.random_state,
        )
    else:
        classifier_step = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=args.random_state,
        )

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max(100, int(args.max_features)),
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=max(1, int(args.min_df)),
                    sublinear_tf=True,
                ),
            ),
            ("classifier", classifier_step),
        ]
    )


def resolve_model_output_path(workspace_dir: str, model_output: str | None) -> str:
    defaults = resolve_default_paths(workspace_dir)
    if model_output:
        return ensure_parent_dir(resolve_cli_path(workspace_dir, model_output))
    return ensure_parent_dir(defaults["model_path"])


def resolve_report_output_path(workspace_dir: str, report_output: str | None) -> str:
    defaults = resolve_default_paths(workspace_dir)
    if report_output:
        return ensure_parent_dir(resolve_cli_path(workspace_dir, report_output))
    return ensure_parent_dir(defaults["report_path"])


def resolve_metadata_output_path(workspace_dir: str, metadata_output: str | None) -> str:
    defaults = resolve_default_paths(workspace_dir)
    if metadata_output:
        return ensure_parent_dir(resolve_cli_path(workspace_dir, metadata_output))
    return ensure_parent_dir(defaults["metadata_path"])


def save_report(report_output_path: str, report_text: str) -> None:
    with open(report_output_path, "w", encoding="utf-8") as report_file:
        report_file.write(report_text.strip())
        report_file.write("\n")


def save_metadata(metadata_output_path: str, metadata: dict[str, object]) -> None:
    with open(metadata_output_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)
        metadata_file.write("\n")


def copy_outputs(copy_output_dir: str | None, output_paths: list[str]) -> list[str]:
    if not copy_output_dir:
        return []

    target_dir = ensure_dir(copy_output_dir)
    copied_paths: list[str] = []
    for output_path in output_paths:
        target_path = os.path.join(target_dir, os.path.basename(output_path))
        shutil.copy2(output_path, target_path)
        copied_paths.append(os.path.abspath(target_path))
    return copied_paths


def maybe_download_artifacts(output_paths: list[str], *, enabled: bool) -> None:
    if not enabled:
        return

    try:
        from google.colab import files  # type: ignore[import-not-found]
    except Exception:
        print("[INFO] --download-artifacts was requested, but google.colab.files is unavailable.")
        return

    for output_path in output_paths:
        print(f"[INFO] Downloading artifact: {output_path}")
        files.download(output_path)


def train_and_export(args: argparse.Namespace) -> tuple[str, str, str]:
    import joblib
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split

    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = resolve_workspace_dir(script_dir)
    source_type, source_path = resolve_dataset_source(
        workspace_dir,
        dataset_csv=args.dataset_csv,
        text_dir=args.text_dir,
        dataset_zip=args.dataset_zip,
        extract_dir=args.extract_dir,
        force_extract=args.force_extract,
    )
    print(f"Using dataset source [{source_type}]: {source_path}")

    if source_type == "csv":
        dataset = load_dataset_from_csv(
            source_path,
            text_column=args.text_column,
            label_column=args.label_column,
        )
    else:
        dataset = load_dataset_from_text_folders(source_path)

    dataset = maybe_sample_dataset(
        dataset,
        args.sample_per_label,
        random_state=args.random_state,
    )

    x = dataset["text"]
    y = dataset["label"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=float(args.test_size),
        random_state=int(args.random_state),
        stratify=y,
    )

    model = build_pipeline(args)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    report_text = classification_report(y_test, predictions, zero_division=0)
    report_dict = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    print("\nClassification report:")
    print(report_text)

    report_output_path = resolve_report_output_path(workspace_dir, args.report_output)
    save_report(report_output_path, report_text)

    model_output_path = resolve_model_output_path(workspace_dir, args.model_output)
    joblib.dump(model, model_output_path)

    metadata_output_path = resolve_metadata_output_path(workspace_dir, args.metadata_output)
    metadata = {
        "dataset_source_type": source_type,
        "dataset_source_path": source_path,
        "row_count": int(len(dataset)),
        "label_count": int(dataset["label"].nunique()),
        "labels": sorted(str(label) for label in dataset["label"].unique()),
        "text_column": args.text_column if source_type == "csv" else "text",
        "label_column": args.label_column if source_type == "csv" else "label",
        "classifier_type": args.classifier_type,
        "test_size": float(args.test_size),
        "max_features": int(args.max_features),
        "min_df": int(args.min_df),
        "sample_per_label": int(args.sample_per_label) if args.sample_per_label is not None else None,
        "random_state": int(args.random_state),
        "report": report_dict,
    }
    save_metadata(metadata_output_path, metadata)

    copied_paths = copy_outputs(
        resolve_cli_path(workspace_dir, args.copy_output_dir) if args.copy_output_dir else None,
        [model_output_path, report_output_path, metadata_output_path],
    )
    if copied_paths:
        print("\nCopied artifacts:")
        for path in copied_paths:
            print(f"- {path}")

    maybe_download_artifacts(
        [model_output_path, report_output_path, metadata_output_path],
        enabled=bool(args.download_artifacts),
    )

    return model_output_path, report_output_path, metadata_output_path


def main() -> int:
    try:
        configure_console_output()
        args = parse_args()
        model_path, report_path, metadata_path = train_and_export(args)
        print(f"\nModel exported successfully to: {model_path}")
        print(f"Classification report saved to: {report_path}")
        print(f"Training metadata saved to: {metadata_path}")
        return 0
    except FileNotFoundError as error:
        print(f"[ERROR] {error}")
        return 1
    except ValueError as error:
        print(f"[ERROR] {error}")
        return 1
    except Exception as error:
        print(f"[ERROR] Unexpected training failure: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
