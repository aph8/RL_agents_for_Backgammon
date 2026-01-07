#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Submission-only SARSA agent (inference only).

- No training, only greedy move selection.
- Lazy-loads value network weights from a checkpoint file.
- Compatible API:
    action(board_copy, dice, player, i, train=False, train_config=None)
    episode_start(), end_episode(...), game_over_update(...), set_eval_mode(...)
"""

from pathlib import Path
import numpy as np
import torch
from torch import Tensor

try:
    import backgammon as Backgammon
except Exception:
    import Backgammon  # fallback

import flipped_agent

device = torch.device("cpu")
print(f"[agent_sarsa_submit] Using device: {device}")

# -------------------- Features --------------------
nx = 24 * 2 * 6 + 4 + 1
H = nx // 2

def one_hot_encoding(board, nSecondRoll: bool) -> np.ndarray:
    oneHot = np.zeros(nx, dtype=np.float32)

    # player +1 bins
    for i in range(1, 6):
        idx = np.where(board[1:25] == i)[0]
        if idx.size > 0:
            oneHot[(i - 1) * 24 + idx] = 1
    idx = np.where(board[1:25] >= 6)[0]
    if idx.size > 0:
        oneHot[5 * 24 + idx] = 1

    # player -1 bins
    for i in range(0, 5):
        idx = np.where(board[1:25] == -i)[0] - 1
        if idx.size > 0:
            oneHot[6 * 24 + i * 24 + idx] = 1
    idx = np.where(board[1:25] <= -5)[0]
    if idx.size > 0:
        oneHot[11 * 24 + idx] = 1

    # bars/offs
    oneHot[24 * 2 * 6 + 0] = board[25] / 2.0
    oneHot[24 * 2 * 6 + 1] = board[26] / 2.0
    oneHot[24 * 2 * 6 + 2] = board[27] / 15.0
    oneHot[24 * 2 * 6 + 3] = board[28] / 15.0

    # second-roll flag can be used, but must match training. Keep it here.
    oneHot[24 * 2 * 6 + 4] = 1.0 if nSecondRoll else 0.0
    return oneHot

# -------------------- Parameters (inference only) --------------------
w1: Tensor = torch.zeros((H, nx), dtype=torch.float32, device=device)
b1: Tensor = torch.zeros((H, 1),  dtype=torch.float32, device=device)
w2: Tensor = torch.zeros((1, H),  dtype=torch.float32, device=device)
b2: Tensor = torch.zeros((1, 1),  dtype=torch.float32, device=device)

# Model path, prefer local file next to this script, else checkpoints/best.pt
_LOCAL_MODEL = Path(__file__).with_name("andri_best.pt")
_FALLBACK_MODEL = Path("checkpoints") / "best.pt"
_loaded_once = False

def _load_weights_if_available():
    global _loaded_once
    if _loaded_once:
        return

    path = _LOCAL_MODEL if _LOCAL_MODEL.exists() else _FALLBACK_MODEL
    if not path.exists():
        _loaded_once = True
        print("[agent_sarsa_submit] Warning: no model file found, playing untrained.")
        return

    try:
        state = torch.load(path, map_location=device)

        # state might store raw tensors or parameters, accept both
        def _t(k):
            v = state[k]
            return v.data if hasattr(v, "data") else v

        if all(k in state for k in ("w1", "b1", "w2", "b2")):
            w1.copy_(_t("w1"))
            b1.copy_(_t("b1"))
            w2.copy_(_t("w2"))
            b2.copy_(_t("b2"))
        else:
            raise KeyError("Checkpoint missing keys w1, b1, w2, b2")

        _loaded_once = True
        print(f"[agent_sarsa_submit] Loaded model from {path}")
    except Exception as e:
        _loaded_once = True
        print(f"[agent_sarsa_submit] Warning: failed to load checkpoint: {e}")

# -------------------- Eval switch and hooks (no-ops) --------------------
_eval_mode = True
def set_eval_mode(is_eval: bool):
    global _eval_mode
    _eval_mode = bool(is_eval)

def episode_start():
    pass

def end_episode(outcome, final_board, perspective):
    pass

def game_over_update(board, reward):
    pass

# -------------------- Forward and greedy policy --------------------
@torch.no_grad()
def _V_from_features(x: torch.Tensor) -> torch.Tensor:
    # x shape (nx, na)
    h = torch.tanh(w1 @ x + b1)     # (H, na)
    y = w2 @ h + b2                 # (1, na)
    return torch.sigmoid(y)         # (1, na)

@torch.no_grad()
def _greedy_action(board_np: np.ndarray, dice, player: int, nSecondRoll: bool):
    # View as +1 POV for consistency
    flipped = (player == -1)
    board_eff = flipped_agent.flip_board(np.copy(board_np)) if flipped else np.copy(board_np)

    # In this codebase legal_moves is typically legal_moves(board, dice, player)
    possible_moves, possible_boards = Backgammon.legal_moves(board_eff, dice, 1)
    if len(possible_moves) == 0:
        return []

    na = len(possible_boards)
    xa = np.zeros((na, nx), dtype=np.float32)
    for k in range(na):
        xa[k, :] = one_hot_encoding(possible_boards[k], nSecondRoll)

    x = torch.tensor(xa.T, dtype=torch.float32, device=device)  # (nx, na)
    v = _V_from_features(x).cpu().numpy().reshape(-1)          # (na,)

    best_idx = int(np.argmax(v))
    move = possible_moves[best_idx]
    if flipped:
        move = flipped_agent.flip_move(move)
    return move

# -------------------- Main API --------------------
def action(board_copy, dice, player, i, train=False, train_config=None):
    _load_weights_if_available()
    nSecondRoll_flag = bool((dice[0] == dice[1]) and (i == 0))
    return _greedy_action(np.copy(board_copy), dice, int(player), nSecondRoll_flag)
