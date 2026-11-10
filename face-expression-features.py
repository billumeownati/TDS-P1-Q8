#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["py-feat==2.1.5", "Pillow"]
# ///
"""Extract the course's 78 Py-Feat features from consenting photos on your computer.

Example:
    uv run face-expression-features.py photos/ --labels labels.csv --output my-features.csv

Photo names must be PERSON-NUMBER.jpg, e.g. sara-01.jpg or 01-01.jpg.
Optional labels.csv has photo_id,intended_expression columns; unlabeled photos get blank labels.
Images stay local; only a CSV of numerical features is written.
"""
import argparse
import csv
import math
import sys
from pathlib import Path

from feat import Detectorv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photos", type=Path, help="Directory of JPEG/PNG photos, named PERSON-NUMBER.jpg")
    parser.add_argument("--labels", type=Path, help="Optional CSV: photo_id,intended_expression")
    parser.add_argument("--output", type=Path, default=Path("my-features.csv"))
    args = parser.parse_args()
    photos = sorted(p.resolve() for p in args.photos.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not photos or any("-" not in p.stem or not p.stem.rsplit("-", 1)[1].isdigit() for p in photos):
        parser.error("Add JPEG/PNG files named PERSON-NUMBER.jpg, e.g. sara-01.jpg")
    if len({p.stem for p in photos}) != len(photos):
        parser.error("Each photograph needs a unique name, even across file extensions")
    labels = {}
    if args.labels:
        with args.labels.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["photo_id"] in labels:
                    parser.error(f"Duplicate label for {row['photo_id']}")
                labels[row["photo_id"]] = row["intended_expression"]
        if missing := {p.stem for p in photos} - labels.keys():
            parser.error(f"Missing labels for: {', '.join(sorted(missing)[:5])}")

    # Match the course pipeline: Py-Feat 2.1.5, RetinaFace + multitask v2, 320px input.
    # Skip identity embeddings; do not infer or publish people's identities.
    detector = Detectorv2(device="cpu", identity_model=None)
    rows = []
    columns = None
    for start in range(0, len(photos), 10):
        batch = photos[start:start + 10]
        output = detector.detect([str(p) for p in batch], data_type="image",
                                 output_size=320, batch_size=len(batch), progress_bar=False)
        blend = [name for name in output.blendshapes.columns if name != "_neutral"]
        au = list(output.aus.columns)
        emotion = list(output.emotions.columns)
        if (len(blend), len(au), len(emotion)) != (51, 20, 7):
            raise RuntimeError("Py-Feat feature schema changed; use py-feat==2.1.5")
        names = [f"bs_{k}" for k in blend] + [f"au_{k}" for k in au] + [f"emotion_{k}" for k in emotion]
        if columns is not None and names != columns:
            raise RuntimeError("Py-Feat returned inconsistent feature columns")
        columns = names
        for path in batch:
            found = output.loc[output["input"] == str(path)]
            if len(found) != 1:
                print(f"SKIP {path.name}: expected one detected face, found {len(found)}", file=sys.stderr)
                continue
            f = found.iloc[0]
            values = [float(f[k]) for k in blend + au + emotion]
            if not all(math.isfinite(x) for x in values):
                print(f"SKIP {path.name}: missing features", file=sys.stderr)
                continue
            rows.append([path.stem.rsplit("-", 1)[0], path.stem, labels.get(path.stem, ""), *values])
        print(f"Processed {min(start + 10, len(photos))}/{len(photos)} photographs", file=sys.stderr)

    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["person_id", "photo_id", "intended_expression", *columns])
        writer.writerows(rows)
    print(f"Saved {len(rows)} photographs and 78 features to {args.output}")


if __name__ == "__main__":
    main()
