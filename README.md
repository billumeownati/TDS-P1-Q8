# Q8: Train a face-expression classifier — explained for a 15-year-old

**In a hurry?** The answer file is [`model.json`](model.json). Upload it on the question page, click **Check**, then **Save**. Read on to understand how it was made and how to make it better.

| File | What it is |
| --- | --- |
| `model.json` | The answer you submit: 10 rows × 156 numbers |
| `train.py` | The program that produces `model.json` |
| `data/features.csv`, `data/features-2.csv` | The numbers from the two course ZIPs, 300 rows each (no photos are stored here) |
| `face-expression-features.py` | The course's tool for turning *your own* photos into the same 78 numbers |

---

## Step 0. What is the game?

Imagine a stranger hands you **10 photos of themselves**. In each one they were asked to make a different face: neutral, happy, sad, angry, surprised, contempt, restrained amusement, scepticism, uncertainty, hopelessness. Your job is to put the right label on each photo.

Two things make this easier than it sounds:

1. **You see all 10 photos together.** You don't have to decide "is this person happy?" in isolation. You only have to decide "which of these ten is the *happiest*?"
2. **Each label is used exactly once.** If you are sure which photo is "happy", no other photo can be "happy". Good guesses help the bad ones.

Guessing randomly gets 1 in 10 right (10%). Bonus marks start above 60% and max out at 80%.

## Step 1. What does the data look like?

Computers can't "look" at a face the way you do, so the course already ran each photo through a tool called Py-Feat. It measured **78 things**, each a number, for example:

- `bs_mouthSmileLeft` — how much the left mouth corner is pulled into a smile
- `au_AU04` — how much the eyebrows are pulled down
- `emotion_Happy` — Py-Feat's own guess at "how happy does this look"

So each photo is one row of 78 numbers, plus the label of what the person was *asked* to show.

We have two downloads of 30 people each. **15 people appear in both** (identical numbers), so together they give **45 different people** × 10 photos = 450 rows. `train.py` spots the repeats and counts each person once. This matters: if the same person sat in both the training group and the test group, the test would be too easy and the accuracy would look better than it really is.

## Step 2. What must we hand in?

A grid of numbers called **weights**: 10 rows (one per expression) × 156 columns.

Think of each row as a **recipe** for spotting one expression. The "happy" recipe might say: *smile matters a lot (+), frown counts against (−), eyebrows don't matter (0)*. To score a photo for "happy", the grader multiplies each of the photo's numbers by the matching weight and adds everything up. Big total = "this looks happy".

The grader does that for all 10 expressions on all 10 photos (a 10 × 10 table of scores), then picks the one-label-per-photo arrangement with the highest total.

## Step 3. Why 156 columns when there are only 78 features?

People's faces differ. Someone's relaxed face may already look a bit like a smile, so "smile = 0.4" means different things for different people.

So the grader adds 78 **rank** numbers. For each feature it asks: *among this person's ten photos, where does this photo stand?* Lowest gets 0, highest gets 1, the rest are evenly spread in between. Now "this is the smiliest of this person's ten photos" is a number the model can use, whatever that person's face normally looks like.

78 raw numbers + 78 ranks = **156 inputs**, hence 156 weights per row.

## Step 4. How do we find good weights? (`train.py`)

We don't write the recipes by hand. We let the computer **learn** them from the 300 labelled examples: start with all weights at zero, check how wrong the answers are, nudge every weight slightly in the direction that makes them less wrong, and repeat a few hundred times. That is all "training" means.

`train.py` adds four ideas on top:

1. **Compare each person with themselves.** Before training, it subtracts each person's own average from their numbers. This is free: adding the same amount to all ten of a person's photos can never change which photo wins, so the weights still work on the grader's raw numbers.
2. **Practise the real game.** The model is trained on two questions at once: "which expression is this photo?" and "out of this person's ten photos, which one shows this expression?". The second is exactly what the grader's matching rewards.
3. **Mirror the faces.** A face flipped left-to-right shows the same expression. Swapping every `...Left` feature with its `...Right` partner gives a second copy of each person for free, doubling the training data.
4. **Don't let the model memorise.** With 1,560 weights and only 450 photos, the model could simply memorise these 45 people. A penalty on large weights (called *regularisation*, the `alpha` setting) forces it to keep the recipes simple, which works better on strangers.

At the end, the weights are divided by each feature's spread (the question's "divide each coefficient by `scale_`"), so they apply directly to the raw numbers the grader uses.

## Step 5. How do we know it works on *new* people?

The final test uses people the model has never seen, so we must test the same way. Scoring it on the people it trained on would be like marking an exam the student has already seen the answers to.

`train.py` hides about a sixth of the 45 people, trains on the rest, and tests on the hidden ones. It repeats this until everyone has been hidden once, then does the whole thing again with 5 different shuffles and averages. A person's ten photos always stay together, never split between training and testing.

Results from this data (45 people):

| Setting | Accuracy on unseen people |
| --- | --- |
| **alpha = 1 (chosen)** | **68.2%** |
| alpha = 3 | 67.3% |
| alpha = 10 | 65.0% |
| alpha = 30 | 64.1% |

For comparison, the same model scores 87.3% on the people it trained on. That gap is why you never trust the training score.

**Honest expectation:** about 68%, which on the provisional scale (0 marks at 60%, 5 at 70%, 10 at 80%) is roughly 4 of 10 bonus marks. With only 45 people the estimate can easily be a few points off either way.

## Step 6. The best way to improve: more people

How accuracy grows as the model sees more people:

| People trained on | Accuracy on unseen people |
| --- | --- |
| 5 | 50.0% |
| 10 | 57.6% |
| 15 | 62.6% |
| 20 | 64.8% |
| 37 | 68.1% |

Adding the second download proved the point: going from 30 to 45 people lifted the score from about 64% to about 68%. It is still climbing, so more data is the most reliable route to 70%+. Two ways to get it:

- **Combine with classmates.** IITM accounts get different subsets of people, and sharing is allowed. Just pass several CSVs to `train.py`; people you already have are dropped automatically.
- **Photograph yourself and willing friends** making all ten faces, then:

  1. Name the files `sara-01.jpg` … `sara-10.jpg` and put them in a folder `photos/`.
  2. Write `labels.csv`:

     ```csv
     photo_id,intended_expression
     sara-01,neutral
     sara-02,happy
     ```

  3. Turn the photos into numbers (this runs on your computer; photos are not uploaded):

     ```bash
     uv run face-expression-features.py photos/ --labels labels.csv --output more-features.csv
     ```

Each person needs exactly one photo for each of the ten expressions, spelled exactly as in the course CSV (for example `restrained-amusement`); otherwise `train.py` skips that person. Only use photos you have permission to use.

## Step 7. Run it yourself

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) once, then:

```bash
uv run train.py data/features.csv data/features-2.csv
```

With extra data and the accuracy-versus-people table:

```bash
uv run train.py data/features.csv data/features-2.csv more-features.csv --learning-curve
```

It takes a few seconds, prints the accuracy on unseen people, and rewrites `model.json`.

## Step 8. Submit

1. On the question page, choose `model.json` (or paste its contents).
2. Click **Check** — this only confirms the file's shape, not its accuracy.
3. Click **Save**. You can retrain and save again as often as you like until Project 1 closes; only the last save counts.

`train.py` already confirms the file has exactly the two keys `version` and `weights`, is 10 × 156 finite numbers, and is under 64 KiB (it is about 15 KiB).
