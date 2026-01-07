#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agent_sarsa.py

On-policy SARSA-like value learning for Backgammon using AFTERSTATES.

Við lærum V(s_pov) = líkurnar á sigri fyrir þann leikmann sem er að fara að
gera næsta leik (player-to-move, túlkaður sem +1 í POV).

Hugmynd:
- Environment kallar legal_moves(board, dice, player) og skilar afterstates.
- Eftir að við veljum move, gerum við ráð fyrir að ANDSTÆÐINGUR sé næstur.
- Við skoðum því stöðuna úr POV "next player" (andstæðingsins) með flip_board,
  reiknum V(next_pov) = líkur á SIGRI þess leikmanns og okkar sigur-líkur eru
  u.þ.b. 1 - V(next_pov).

Trainerinn sem kennarinn gaf:
- kallar play_one_game(agent, agent, training=True, ...)
- kallar svo end_episode(+1 eða -1, final_board, perspective=+1/-1)

Við túlkum reward > 0 sem "win" => target = 1.0 og reward <= 0 sem "loss" => target = 0.0.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import backgammon
import flipped_agent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[agent_sarsa] Using device: {device}")

# -------------------- Hyperparameters --------------------
alpha1 = 0.0005  # step-size for hidden layers
alpha2 = 0.0005  # step-size for output layer
gamma = 1.0      # discount (episodic leikur, svo oft 1.0)

epsilon_start = 0.30   # byrjum með góðan skammt exploration
epsilon_min   = 0.05   # förum ekki alveg niður í 0
epsilon_decay_games = 100_000  # fjöldi LEIKJA til að fara úr start í min

# -------------------- Feature size --------------------
# Sama encoding og í TD-agentunum:
# 24 reitir * 2 leikmenn * 6 bin + 4 special + 1 flag
nx = 24 * 2 * 6 + 4 + 1
H1 = nx          # fyrra falið lag
H2 = nx // 2     # seinna falið lag

# -------------------- Encoding --------------------
def one_hot_encoding(board, nSecondRoll: bool):
    """
    Feature encoding í anda þeirra agent-a sem fylgja verkefninu.
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

    # second-roll flag: við notum hann ekki í þessari SARSA-útgáfu
    oneHot[24 * 2 * 6 + 4] = 0.0
    return oneHot

def _x(board_pov):
    """
    board_pov: tafla í POV (orientation) þess leikmanns sem við metum sem +1.
    """
    return torch.tensor(
        one_hot_encoding(board_pov, False),
        dtype=torch.float32,
        device=device
    ).view(-1, 1)

# -------------------- Value network V(s_pov) --------------------
rng = np.random.RandomState(0)

# Tvö falin lög: nx -> H1 -> H2 -> 1
w1 = nn.Parameter(torch.tensor(rng.normal(0, 0.10, size=(H1, nx)), dtype=torch.float32, device=device))
b1 = nn.Parameter(torch.zeros((H1, 1), dtype=torch.float32, device=device))

w2 = nn.Parameter(torch.tensor(rng.normal(0, 0.10, size=(H2, H1)), dtype=torch.float32, device=device))
b2 = nn.Parameter(torch.zeros((H2, 1), dtype=torch.float32, device=device))

w3 = nn.Parameter(torch.tensor(rng.normal(0, 0.10, size=(1, H2)), dtype=torch.float32, device=device))
b3 = nn.Parameter(torch.zeros((1, 1), dtype=torch.float32, device=device))

def V(x):
    """
    V(x) = sigmoid(MLP(x)) ≈ P(win | state_pov)
    """
    h1 = torch.tanh(w1 @ x + b1)      # (H1, 1)
    h2 = torch.tanh(w2 @ h1 + b2)     # (H2, 1)
    y  = w3 @ h2 + b3                 # (1, 1)
    return torch.sigmoid(y)           # (1,1)

# -------------------- Training state --------------------
_eval_mode = False
current_epsilon  = epsilon_start
_game_counter = 0        # fjöldi leikja (fyrir save/load)
_episode_active = False

# Fyrri afterstate (úr POV "next player") sem bíður eftir SARSA-update
_prev_x_nextplayer = None
_prev_pending = False

CKPT_DEFAULT = Path("checkpoints/agent_sarsa.pt")

# -------------------- Helpers --------------------
def set_eval_mode(on: bool):
    global _eval_mode
    _eval_mode = bool(on)

def update_epsilon_for_game(game_idx: int):
    """
    Kallað úr train.py EINU SINNI fyrir hvern leik:
        agent.update_epsilon_for_game(g)

    Þá er prentunin "games=..." í raun fjöldi leikja, ekki steps.
    """
    global current_epsilon, _game_counter
    if _eval_mode:
        return
    _game_counter = int(game_idx)
    prog = min(game_idx / float(epsilon_decay_games), 1.0)
    current_epsilon = max(
        epsilon_start - prog * (epsilon_start - epsilon_min),
        epsilon_min
    )
    if game_idx % 1000 == 0:
        print(f"    [epsilon] games={game_idx} epsilon={current_epsilon:.4f}")

def episode_start():
    global _episode_active
    global _prev_x_nextplayer, _prev_pending
    _episode_active = True
    _prev_x_nextplayer = None
    _prev_pending = False

def end_episode(reward, final_board, perspective):
    """
    Trainerinn kallar:
        end_episode(+1 if winner == 1 else -1, final_board, perspective=+1)
    fyrir "fyrri" agent, og öfugt fyrir hinn.

    Við túlkum reward > 0 sem SIGUR fyrir agentinn, annars TAP.
    target = 1.0 fyrir win, 0.0 fyrir loss.
    """
    global _episode_active
    global _prev_x_nextplayer, _prev_pending

    if not _episode_active:
        return

    if _prev_pending and not _eval_mode:
        # umritum reward (+1/-1) yfir í 1.0/0.0 target
        r = 1.0 if reward > 0 else 0.0
        with torch.enable_grad():
            prev_v = V(_prev_x_nextplayer)
            target = torch.tensor([[r]], dtype=torch.float32, device=device)
            delta = (target - prev_v).item()
            prev_v.backward()
        _manual_sgd(delta)

    _episode_active = False
    _prev_x_nextplayer = None
    _prev_pending = False

def game_over_update(board, reward):
    """
    Haldið inni fyrir compatibility við eldri trainer. Notum end_episode
    fyrir terminal updates, þannig að þetta er no-op hér.
    """
    return

# -------------------- Save / Load --------------------
def save(path=None):
    p = Path(path) if path else CKPT_DEFAULT
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "w1": w1.data.cpu(),
        "b1": b1.data.cpu(),
        "w2": w2.data.cpu(),
        "b2": b2.data.cpu(),
        "w3": w3.data.cpu(),
        "b3": b3.data.cpu(),
        "epsilon": current_epsilon,
        "games": _game_counter,
    }, p)
    print(f"  Model saved to {p}")

def load(path=None, map_location=None):
    global current_epsilon, _game_counter
    p = Path(path) if path else CKPT_DEFAULT
    ml = map_location or device
    if p.exists():
        s = torch.load(p, map_location=ml)
        with torch.no_grad():
            w1.data.copy_(s["w1"])
            b1.data.copy_(s["b1"])
            w2.data.copy_(s["w2"])
            b2.data.copy_(s["b2"])
            w3.data.copy_(s["w3"])
            b3.data.copy_(s["b3"])
        current_epsilon = float(s.get("epsilon", epsilon_start))
        _game_counter = int(s.get("games", 0))
        print(f"  Model loaded from {p}")

# -------------------- Manual SGD --------------------
def _manual_sgd(delta: float):
    """Bein SGD uppfærsla með gradientunum sem eru í .grad."""
    with torch.no_grad():
        # Hidden lög
        w1.data += alpha1 * delta * w1.grad.data
        b1.data += alpha1 * delta * b1.grad.data
        w2.data += alpha1 * delta * w2.grad.data
        b2.data += alpha1 * delta * b2.grad.data
        # Output lag
        w3.data += alpha2 * delta * w3.grad.data
        b3.data += alpha2 * delta * b3.grad.data
    w1.grad = None
    b1.grad = None
    w2.grad = None
    b2.grad = None
    w3.grad = None
    b3.grad = None

# --------------- Core: action + SARSA update --------------------
def action(board_copy, dice, player, i, train=False, train_config=None):
    """
    Velur move fyrir gefna stöðu.

    board_copy: tafla frá environment
    dice: tveggja staka vektor (t.d. [3,5])
    player: +1 eða -1 (hver á leik)
    i: index fyrir doubles (0 eða 1), við notum hann ekki beint
    train: ef True, uppfærum við (SARSA-style), annars bara greedy/ε-greedy
    """
    global _episode_active
    global _prev_x_nextplayer, _prev_pending

    # Finna legal moves & afterstates í ORIGINAL orientation
    legal, boards = backgammon.legal_moves(board_copy, dice, player)
    if not legal:
        return []

    candidates = []
    for move, new_board in zip(legal, boards):
        # Eftir okkar move er andstæðingur næstur.
        # Við skoðum stöðuna úr POV "next player" með flip_board.
        next_board_pov = flipped_agent.flip_board(new_board)
        x_next = _x(next_board_pov)

        with torch.no_grad():
            v_nextplayer = V(x_next).item()  # líkur á SIGRI þess sem á leik næstur

        # Okkar líkur ≈ 1 - v_nextplayer
        score = 1.0 - v_nextplayer
        candidates.append((score, x_next, move))

    # ε-greedy val (epsilon er uppfært per leik með update_epsilon_for_game)
    if _eval_mode or (not train):
        best = max(candidates, key=lambda t: t[0])
    else:
        if np.random.rand() < current_epsilon:
            best = candidates[np.random.randint(len(candidates))]
        else:
            best = max(candidates, key=lambda t: t[0])

    chosen_score, chosen_x_nextplayer, chosen_move = best

    # SARSA update á previous afterstate
    if train and not _eval_mode and _episode_active:
        if _prev_pending:
            with torch.enable_grad():
                prev_v = V(_prev_x_nextplayer)
                target = torch.tensor(
                    [[float(chosen_score)]],
                    dtype=torch.float32,
                    device=device
                )
                delta = (target - prev_v).item()
                prev_v.backward()
            _manual_sgd(delta)

    # Vista núverandi "next player" afterstate fyrir næsta uppfærslu
    if train and _episode_active:
        _prev_x_nextplayer = chosen_x_nextplayer
        _prev_pending = True
    else:
        _prev_x_nextplayer = None
        _prev_pending = False

    # move er í original koordinötum (legal_moves skilar þannig), svo við þurfum
    # ekki að flip'a move til baka.
    return chosen_move
