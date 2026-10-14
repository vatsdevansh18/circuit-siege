# Circuit Siege — Design Document

My planning notes from before I started building, updated to match the final version.

## 1. Concept

**Circuit Siege** is a two-player, perfect-information territory game played on a 7×7
"circuit board". The human (blue) and the AI (red) take turns placing *nodes* on
empty cells. A placed node captures enemy nodes by **flanking** them, Othello-style,
in any of the 8 directions. Unlike Othello, you may place on **any** empty cell, so you
always have a legal move, and the game always lasts exactly 49 moves.

Five cells are **cores** worth **3 points**. Every other cell is worth 1. The total
is 44 + 15 = **59 points**. Because that is odd, a game can never end in a draw.

### Rules
1. The board starts empty. The starting player alternates between matches (or is
   chosen on the start screen).
2. On your turn, place one node on any empty cell.
3. **Flank capture:** in each of the 8 directions, if the new node is followed by an
   unbroken line of one or more enemy nodes that ends in one of *your* nodes, the
   whole line flips to your colour. Several directions can capture on the same turn.
4. The game ends when the board is full. Each player scores 1 point per normal cell
   and 3 points per core they control. The higher score wins.

### Why it is strategic
- Corners can never be flanked, so they are permanently safe. Edges are only
  partly safe.
- Cores are worth 3×, so they are hot targets, but they sit in exposed positions.
- Every placement can also open up a line for your opponent to recapture. Good play
  means weighing immediate gains against what you expose.
- Captures cascade, so a single move late in the game can swing the score a lot.

## 2. Why machine learning?

Reading a position ("who is really ahead here?") is hard. Raw score is misleading
because stones flip constantly. A *learned evaluation function*, which estimates
the probability of winning from a given position, is the classical ML approach to
board games (see Samuel's checkers program and TD-Gammon). It is explainable,
measurable (it is a binary classifier), and fast to run during play.

## 3. What the AI learns

A **value model**: `P(win | position, perspective)`.

The AI decides by **1-ply lookahead plus the learned evaluation**. For each legal
move, the engine simulates the resulting position. That position is converted into
features, and the model scores the AI's win probability. The AI plays the move with
the highest probability (Hard), or samples among strong moves (Normal and Easy).
The game rules only *generate* candidate moves. The *choice* between them comes
entirely from the trained model.

## 4. Dataset construction

- **Source:** synthetic self-play. Four scripted *data-generator* bots (Random,
  Greedy, Cautious, Positional), each with ε-randomness, play thousands of games in
  random pairings with seeded RNGs. These bots exist only to produce varied,
  realistic positions. They are **not** the AI opponent.
- **Raw data:** `data/raw/selfplay_games.jsonl`, one record per game (move list,
  bots used, final score, winner).
- **Processed data:** `data/processed/positions.csv.gz`. Every non-terminal position
  of every game becomes one row, seen from a randomly chosen player's perspective.
  It has 49 raw cell features (+1 mine / −1 theirs / 0 empty) plus engineered
  features (score, cores, corners, edges, frontier, vulnerability, best immediate
  gain, turn and parity). The label is `1` if the perspective player won the game.
- **Leakage control:** the train, validation and test split is done **by game**,
  never by row, so no game contributes positions to more than one split.
- **Limitations:** the labels reflect outcomes between imperfect scripted bots, not
  optimal play or human play. See README → Limitations.

## 5. Algorithm choice

Four candidate classifiers are trained and compared on the validation split by
log-loss: Logistic Regression, Random Forest, Histogram Gradient Boosting, and an
MLP. The one with the best log-loss is refit on train + validation and evaluated
once on the held-out test games. Log-loss is the selection metric because the AI
uses the *probabilities* to rank moves, not just the predicted class.

Baselines: majority class, and the rule "whoever leads on points will win".

Gameplay evaluation (reward-based): the trained AI plays an arena against each
scripted bot, and win rates are reported with 95% confidence intervals.

## 6. Stack

- **Python** (engine, ML pipeline, Flask server). scikit-learn, pandas, NumPy,
  matplotlib, joblib.
- **HTML/CSS/JS frontend** served locally by Flask. This allows a polished,
  animated UI without a game-engine dependency.
- Local-first: no API keys, no cloud, no paid services.

## 7. Adaptive mode (offline retraining)

Every finished human-vs-AI match is logged to `data/player_games/matches.jsonl`.
`scripts/retrain_with_player_data.py` turns those matches into extra training rows,
trains a *candidate* model, and evaluates it against the current model, both on
held-out data and head-to-head. It promotes the candidate only if it is not worse,
and it archives the old model first. This is **deliberate offline retraining**, not
online learning. The AI does not change during a match.
