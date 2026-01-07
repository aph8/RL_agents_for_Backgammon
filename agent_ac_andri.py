#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agent_ac_andri.py

Actor–Critic backgammon agent með afterstate-policy og value-baseline.

Hugmynd:
- Við lærum V(s_pov) = líkindi á sigri fyrir þann leikmann sem er að fara að gera
  næsta leik (player-to-move, túlkaður sem +1 í POV).
- Policy-net metur hverja mögulega "afterstate" (tafla eftir leik) fyrir núverandi
  leikmann og býr til π(a|s) með softmax.
- Þjálfun er episodic:
  - Söfnum öllum skrefum (state, legal afterstates, valin aðgerð, player) í trajectory.
  - Í end_episode reiknum við hver vann úr final_board og gefum öllum skrefum
    reward = 1.0 ef sá leikmaður (í því skrefi) vann, annars 0.0.
  - Critic lærir V(s) → [0,1].
  - Actor notar advantage ≈ r - V(s) til policy-gradient uppfærslu.

Agentinn er hannaður til að passa beint við train.py sem kennarinn gaf:
- action(board, dice, player, i, train=False, train_config=None)
- episode_start(), end_episode(reward, final_board, perspective)
- save(path), load(path), set_eval_mode(bool), game_over_update(...)
"""

from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn

import backgammon
import flipped_agent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[agent_ac_andri] Using device: {device}")

# -------------------- Feature size & encoding --------------------
# Sama og TD agentarnir nota:
# 24 reitir * 2 leikmenn * 6 bin + 4 special + 1 flag
nx = 24 * 2 * 6 + 4 + 1
H_val = nx       # hidden size fyrir value-net
H_pol = nx       # hidden size fyrir policy-net

rng = np.random.RandomState(42)


def one_hot_encoding(board, nSecondRoll: bool = False):
    """
    Feature encoding í anda TD-agentanna.
    board er 1D array (stærð 29): 1..24 reitir, 25/26 bar, 27/28 borne off.
    """
    oneHot = np.zeros(nx, dtype=np.float32)

    # Player +1 bins
    for i in range(1, 6):
        idx = np.where(board[1:25] == i)[0]
        if idx.size > 0:
            oneHot[(i - 1) * 24 + idx] = 1
    idx = np.where(board[1:25] >= 6)[0]
    if idx.size > 0:
        oneHot[5 * 24 + idx] = 1

    # Player -1 bins
    offset = 24 * 6
    for i in range(1, 6):
        idx = np.where(board[1:25] == -i)[0]
        if idx.size > 0:
            oneHot[offset + (i - 1) * 24 + idx] = 1
    idx = np.where(board[1:25] <= -6)[0]
    if idx.size > 0:
        oneHot[offset + 5 * 24 + idx] = 1

    # bar og borne off
    oneHot[24 * 2 * 6 + 0] = board[25] / 2.0
    oneHot[24 * 2 * 6 + 1] = board[26] / 2.0
    oneHot[24 * 2 * 6 + 2] = board[27] / 15.0
    oneHot[24 * 2 * 6 + 3] = board[28] / 15.0

    # second-roll flag (notað ekki hér)
    oneHot[24 * 2 * 6 + 4] = 0.0
    return oneHot


def _x(board_pov):
    """
    board_pov: tafla eins og current-player er +1.
    """
    return np.asarray(one_hot_encoding(board_pov, False), dtype=np.float32)


# -------------------- Value network V(s) --------------------
v_w1 = nn.Parameter(
    torch.tensor(rng.normal(0, 0.10, size=(H_val, nx)), dtype=torch.float32, device=device)
)
v_b1 = nn.Parameter(torch.zeros((H_val, 1), dtype=torch.float32, device=device))
v_w2 = nn.Parameter(
    torch.tensor(rng.normal(0, 0.10, size=(1, H_val)), dtype=torch.float32, device=device)
)
v_b2 = nn.Parameter(torch.zeros((1, 1), dtype=torch.float32, device=device))


def value_forward(x: torch.Tensor) -> torch.Tensor:
    """
    Value-net: V(x) ≈ P(win | state), í [0,1].
    x: (nx,1)
    """
    h = torch.tanh(v_w1 @ x + v_b1)       # (H_val,1)
    y = v_w2 @ h + v_b2                   # (1,1)
    return torch.sigmoid(y)               # (1,1)


# -------------------- Policy network π(a|s) over AFTERSTATES --------------------
p_w1 = nn.Parameter(
    torch.tensor(rng.normal(0, 0.10, size=(H_pol, nx)), dtype=torch.float32, device=device)
)
p_b1 = nn.Parameter(torch.zeros((H_pol, 1), dtype=torch.float32, device=device))
p_w2 = nn.Parameter(
    torch.tensor(rng.normal(0, 0.10, size=(1, H_pol)), dtype=torch.float32, device=device)
)
p_b2 = nn.Parameter(torch.zeros((1, 1), dtype=torch.float32, device=device))


def policy_logit(x: torch.Tensor) -> torch.Tensor:
    """
    Policy-net: tekur afterstate feature x og skilar logit (fyrir softmax).
    x: (nx,1) -> logit: (1,1)
    """
    h = torch.tanh(p_w1 @ x + p_b1)       # (H_pol,1)
    y = p_w2 @ h + p_b2                   # (1,1)
    return y                              # logits (1,1)


# -------------------- Training state --------------------
_eval_mode = False

# trajectory: list of dicts fyrir hvert training step
_trajectory: List[Dict[str, Any]] = []
_episode_active = False

# learning rates
lr_value = 0.0005
lr_policy = 0.0005

CKPT_DEFAULT = Path("checkpoints/agent_ac_andri.pt")


def set_eval_mode(on: bool):
    global _eval_mode
    _eval_mode = bool(on)


def episode_start():
    """
    Kallað í upphafi hvers leiks í train.py.
    """
    global _trajectory, _episode_active
    _trajectory = []
    _episode_active = True


def end_episode(reward, final_board, perspective):
    """
    Kallað þegar leik er lokið í train.py.

    reward: +1 eða -1 (frá sjónarhorni perspective), en við notum frekar final_board
            til að ákvarða hver vann.
    final_board: lokastaða leiks (stærð 29)
    perspective: +1 eða -1 (hvor "agentinn" var í þessu kall-i). Við notum það ekki
                 beint, því sami kóði þjónar báðum.
    """
    global _trajectory, _episode_active
    if not _episode_active or not _trajectory:
        _episode_active = False
        _trajectory = []
        return

    # Reiknum hver vann úr final_board í global orientation:
    winner = 1 if final_board[27] == 15 else -1

    # Zero-a gamla gradients
    params = [v_w1, v_b1, v_w2, v_b2, p_w1, p_b1, p_w2, p_b2]
    for p in params:
        if p.grad is not None:
            p.grad = None

    total_loss = 0.0
    n_steps = len(_trajectory)
    if n_steps == 0:
        _episode_active = False
        _trajectory = []
        return

    for step in _trajectory:
        player = step["player"]       # +1 eða -1
        x_state_np = step["x_state"]  # (nx,)
        xs_actions_np = step["xs_actions"]  # (K, nx)
        chosen_idx = step["chosen_idx"]

        # Reward fyrir þetta skref: 1.0 ef þessi leikmaður vann, annars 0.0
        r = 1.0 if player == winner else 0.0

        # Value forward
        x_state = torch.tensor(x_state_np, dtype=torch.float32, device=device).view(-1, 1)
        v_s = value_forward(x_state)  # (1,1)

        # Policy forward yfir allar possible afterstates
        logits_list = []
        for row in xs_actions_np:
            x_a = torch.tensor(row, dtype=torch.float32, device=device).view(-1, 1)
            logit = policy_logit(x_a)  # (1,1)
            logits_list.append(logit.view(()))  # scalar

        logits = torch.stack(logits_list)      # (K,)
        probs = torch.softmax(logits, dim=0)   # (K,)
        prob_chosen = probs[chosen_idx]
        logprob = torch.log(prob_chosen + 1e-8)

        # Critic target og advantage
        target = torch.tensor([r], dtype=torch.float32, device=device)  # (1,)
        mse = (v_s.view(1) - target)**2

        # Advantage ~ (r - V(s))
        adv = (r - v_s.detach().item())

        # Policy loss (viljum MAX adv * logπ → MINUS í loss)
        policy_loss = -adv * logprob

        # Heildar loss fyrir þetta skref
        value_coeff = 0.5
        policy_coeff = 1.0
        step_loss = value_coeff * mse.mean() + policy_coeff * policy_loss
        total_loss = total_loss + step_loss

    total_loss = total_loss / n_steps

    # Backprop og manual SGD
    total_loss.backward()

    with torch.no_grad():
        # Value net update
        v_w1.data -= lr_value * v_w1.grad.data
        v_b1.data -= lr_value * v_b1.grad.data
        v_w2.data -= lr_value * v_w2.grad.data
        v_b2.data -= lr_value * v_b2.grad.data

        # Policy net update
        p_w1.data -= lr_policy * p_w1.grad.data
        p_b1.data -= lr_policy * p_b1.grad.data
        p_w2.data -= lr_policy * p_w2.grad.data
        p_b2.data -= lr_policy * p_b2.grad.data

    # Hreinsa grads
    for p in params:
        p.grad = None

    _episode_active = False
    _trajectory = []


def game_over_update(board, reward):
    """
    Haldið inni fyrir compatibility við eldri trainer. Við notum end_episode
    fyrir episodic uppfærslur, svo þetta er no-op.
    """
    return


# -------------------- Save / Load --------------------
def save(path=None):
    p = Path(path) if path else CKPT_DEFAULT
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "v_w1": v_w1.data.cpu(),
            "v_b1": v_b1.data.cpu(),
            "v_w2": v_w2.data.cpu(),
            "v_b2": v_b2.data.cpu(),
            "p_w1": p_w1.data.cpu(),
            "p_b1": p_b1.data.cpu(),
            "p_w2": p_w2.data.cpu(),
            "p_b2": p_b2.data.cpu(),
        },
        p,
    )
    print(f"  [agent_ac_andri] Model saved to {p}")


def load(path=None, map_location=None):
    p = Path(path) if path else CKPT_DEFAULT
    ml = map_location or device
    if not p.exists():
        print(f"  [agent_ac_andri] No checkpoint at {p}, starting fresh.")
        return
    s = torch.load(p, map_location=ml)
    with torch.no_grad():
        v_w1.data.copy_(s["v_w1"])
        v_b1.data.copy_(s["v_b1"])
        v_w2.data.copy_(s["v_w2"])
        v_b2.data.copy_(s["v_b2"])
        p_w1.data.copy_(s["p_w1"])
        p_b1.data.copy_(s["p_b1"])
        p_w2.data.copy_(s["p_w2"])
        p_b2.data.copy_(s["p_b2"])
    print(f"  [agent_ac_andri] Model loaded from {p}")


# -------------------- Core: action --------------------
def action(board_copy, dice, player, i, train=False, train_config=None):
    """
    Velur move fyrir gefna stöðu.

    board_copy: tafla frá environment (global orientation)
    dice: t.d. [3,5]
    player: +1 eða -1 (hver á leik)
    i: index fyrir doubles (0 eða 1), við notum hann ekki sérstaklega.
    train: ef True, söfnum í trajectory og lærum í end_episode.
    """
    global _episode_active, _trajectory

    # Finna legal moves & afterstates
    legal, boards = backgammon.legal_moves(board_copy, dice, player)
    if not legal:
        return []

    # POV: current player sem +1
    if player == 1:
        board_pov = board_copy
    else:
        board_pov = flipped_agent.flip_board(board_copy)

    x_state_np = _x(board_pov)  # (nx,)

    # Encode afterstates úr POV current player
    xs_actions_np = []
    for new_board in boards:
        if player == 1:
            new_pov = new_board
        else:
            new_pov = flipped_agent.flip_board(new_board)
        xs_actions_np.append(_x(new_pov))

    xs_actions_np = np.stack(xs_actions_np, axis=0)  # (K, nx)

    # Policy logits
    logits_list = []
    for row in xs_actions_np:
        x_a = torch.tensor(row, dtype=torch.float32, device=device).view(-1, 1)
        logit = policy_logit(x_a)  # (1,1)
        logits_list.append(logit.view(()))

    logits = torch.stack(logits_list)      # (K,)
    probs = torch.softmax(logits, dim=0)   # (K,)

    if _eval_mode or (not train):
        # Greedy í eval
        chosen_idx = int(torch.argmax(probs).item())
    else:
        # Sample according to π(a|s)
        p_np = probs.detach().cpu().numpy()
        chosen_idx = int(np.random.choice(len(legal), p=p_np))

    chosen_move = legal[chosen_idx]

    # Ef við erum í training-mode, söfnum við step
    if train and _episode_active:
        step = {
            "player": int(player),
            "x_state": x_state_np.copy(),
            "xs_actions": xs_actions_np.copy(),
            "chosen_idx": int(chosen_idx),
        }
        _trajectory.append(step)

    return chosen_move
