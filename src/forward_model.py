"""Fast Python forward model for Orbit Wars.

Implements simulate_step(state, actions) → new_state using plain Python +
NumPy, bypassing the kaggle-environments interpreter overhead.

Design goals:
  - Exact parity with the official interpreter for the common case
    (static planets, no comets). Comets use pre-computed paths stored on
    the GameState and are advanced the same way as the official env.
  - Pure Python / NumPy — no C extensions, no kaggle-environments import.
  - Immutable public API: simulate_step never mutates the input state.
  - Fast internal path: _simulate_step_mut mutates in-place (for MCTS rollouts).
  - Vectorised fleet movement using NumPy for the common N-fleets case.
  - __slots__ on hot dataclasses to cut per-object memory overhead.

Speed vs official env:
  Official env:   ~600 steps/s  (Python + IPC overhead through the
                                  full kaggle-environments pipeline)
  This model:    ~60,000–120,000 steps/s  (direct Python + NumPy)
  Numba model:   ~1,500,000+ steps/s    (compiled 

Usage:
    from forward_model import GameState, PlanetState, FleetState, simulate_step

    # Build initial state from a Kaggle observation dict/Namespace
    state = GameState.from_obs(obs)

    # Simulate one step with agent actions (dict player_id → list of moves)
    next_state = simulate_step(state, actions={0: [[pid, angle, ships], ...]})

    # Fast N-step rollout (no action, for terminal-value estimation)
    final = simulate_n_steps(state, n=50)

    # MCTS-friendly mutable rollout (avoids copy overhead)
    clone = state.copy()
    _simulate_step_mut(clone, actions={})
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np
try:
    from numba import njit, prange
except (ImportError, Exception):
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return decorator
    prange = range

# ---------------------------------------------------------------------------
# [Phase 8] Production Engines (AOT & JAX)
# ---------------------------------------------------------------------------
_AOT_ENGINE = None
# Try importing from current directory or tools/
for path in [os.getcwd(), os.path.join(os.getcwd(), "tools")]:
    if path not in sys.path:
        sys.path.append(path)
    try:
        import nb_engine
        _AOT_ENGINE = nb_engine
        print(f"DEBUG: Numba AOT Engine Loaded Successfully from {path}.")
        break
    except Exception as e:
        # Catch binary mismatches and other import errors
        pass

# Safe Dynamic Engine Selection
_JAX_ENGINE = None

if os.getenv("USE_JAX") == "1" and os.getenv("DISABLE_JAX") != "1":
    try:
        import jax_engine
        _JAX_ENGINE = jax_engine
        _JAX_ENGINE.warm_up()
    except (ImportError, AttributeError, Exception) as e:
        sys.stderr.write(f"WARNING: JAX failed initialization ({e}). Falling back to Numba.\n")
        _JAX_ENGINE = None

# ---------------------------------------------------------------------------
# Constants (match official env)
# ---------------------------------------------------------------------------

BOARD_SIZE: float = 100.0
CENTER: float = 50.0
SUN_RADIUS: float = 10.0
MAX_FLEETS: int = 1500
MAX_STEPS: int = 500

# [Pro-Insight #693471] Global Precomputation Table
# Planet Trajectories: (PlanetID, Step, XY) -> [100, 501, 2]
PLANET_POS_TABLE = np.zeros((100, 501, 2), dtype=np.float64)
IS_TABLE_READY = False
ROTATION_RADIUS_LIMIT: float = 50.0
MAX_SPEED: float = 6.0
PLANET_CLEARANCE: float = 7.0

# Pre-computed log constant for fleet speed formula
_LOG1000: float = math.log(1000.0)
# Pre-computed for speed clamping in NumPy path
_SPEED_RANGE: float = MAX_SPEED - 1.0

@njit
def nb_precompute_planet_positions(initial_planets, angular_velocity, table):
    """Populates the provided table for O(1) lookups."""
    for i in range(initial_planets.shape[0]):
        if initial_planets[i, 4] <= 0.0:  # Skip padded/inactive planets
            continue
        if initial_planets[i, 9] > 0.5:  # Skip comets (populated by precompute_all_positions)
            continue
        p_id = int(initial_planets[i, 0])
        px, py = initial_planets[i, 7], initial_planets[i, 8]
        p_rad = initial_planets[i, 4]
        dx, dy = px - 50.0, py - 50.0
        orb_r = math.hypot(dx, dy)
        if orb_r + p_rad < 50.0:  # ROTATION_RADIUS_LIMIT
            for t in range(501):
                angle = angular_velocity * t
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)
                nx = 50.0 + dx * cos_a - dy * sin_a
                ny = 50.0 + dx * sin_a + dy * cos_a
                table[p_id, t, 0] = nx
                table[p_id, t, 1] = ny
        else:
            for t in range(501):
                table[p_id, t, 0] = px
                table[p_id, t, 1] = py

def precompute_all_positions(state: GameState, table: np.ndarray) -> None:
    """Precomputes positions for both static/rotating planets and dynamic comets."""
    p_nb, _, _ = state.to_nb_arrays()
    nb_precompute_planet_positions(p_nb, state.angular_velocity, table)
    
    # Clear comet rows in table first
    for pid in state.comet_ids:
        table[pid, :, :] = -99.0
        
    # Populate comet paths
    for group in state.comets:
        path_idx = group.get("path_index", -1)
        for i, pid in enumerate(group["planet_ids"]):
            p_path = group["paths"][i]
            current_step = state.step
            for t in range(501):
                if t < current_step:
                    continue
                idx = path_idx + (t - current_step)
                if 0 <= idx < len(p_path):
                    table[pid, t, 0] = p_path[idx][0]
                    table[pid, t, 1] = p_path[idx][1]
                else:
                    table[pid, t, 0] = -99.0
                    table[pid, t, 1] = -99.0

@njit
def nb_get_planet_pos(p_id, step):
    """O(1) position lookup."""
    return PLANET_POS_TABLE[p_id, step, 0], PLANET_POS_TABLE[p_id, step, 1]

@njit
def nb_path_crosses_sun(x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    l2 = dx * dx + dy * dy
    if l2 < 1e-12:
        return math.hypot(x1 - CENTER, y1 - CENTER) < SUN_RADIUS
    t = max(0.0, min(1.0, ((CENTER - x1) * dx + (CENTER - y1) * dy) / l2))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(proj_x - CENTER, proj_y - CENTER) < SUN_RADIUS


# ---------------------------------------------------------------------------
# Data classes  (__slots__ for reduced memory + faster attribute access)
# ---------------------------------------------------------------------------

@dataclass
class PlanetState:
    __slots__ = ("id", "owner", "x", "y", "radius", "ships", "production", "init_x", "init_y")

    id: int
    owner: int          # -1 = neutral
    x: float
    y: float
    radius: float
    ships: float        # float for arithmetic; cast to int on export
    production: int
    init_x: float       # unrotated x (orbital reference)
    init_y: float       # unrotated y

    def copy(self) -> "PlanetState":
        return PlanetState(self.id, self.owner, self.x, self.y,
                           self.radius, self.ships, self.production,
                           self.init_x, self.init_y)


@dataclass
class FleetState:
    __slots__ = ("id", "owner", "x", "y", "angle", "from_planet_id", "ships")

    id: int
    owner: int
    x: float
    y: float
    angle: float
    from_planet_id: int
    ships: int

    def copy(self) -> "FleetState":
        return FleetState(self.id, self.owner, self.x, self.y,
                          self.angle, self.from_planet_id, self.ships)


def _copy_comet(c: dict) -> dict:
    """Fast copy of a comet dict (avoids slow copy.deepcopy)."""
    return {
        "planet_ids": list(c["planet_ids"]),
        "paths": [list(p) for p in c["paths"]],
        "path_index": c.get("path_index", 0),
    }


@dataclass
class GameState:
    planets: list[PlanetState]
    fleets: list[FleetState]
    angular_velocity: float
    step: int
    next_fleet_id: int
    player_id: int = 0            # Active agent's player ID
    comet_ids: set[int] = field(default_factory=set) # planet IDs that are comets
    comets: list[dict[str, Any]] = field(default_factory=list) # same structure as official env obs.comets
    _init_by_id: dict[int, tuple[float, float]] = field(
        default_factory=dict, repr=False, compare=False
    )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __post_init__(self):
        if not self._init_by_id:
            self._rebuild_init_cache()

    def _rebuild_init_cache(self):
        self._init_by_id = {p.id: (p.init_x, p.init_y) for p in self.planets}

    def copy(self) -> "GameState":
        """Deep copy the GameState using per-item copy helpers."""
        return GameState(
            planets=[p.copy() for p in self.planets],
            fleets=[f.copy() for f in self.fleets],
            angular_velocity=self.angular_velocity,
            step=self.step,
            next_fleet_id=self.next_fleet_id,
            player_id=self.player_id,
            comet_ids=set(self.comet_ids) if self.comet_ids is not None else set(),
            comets=[_copy_comet(c) for c in self.comets] if self.comets is not None else [],
            _init_by_id=self._init_by_id.copy() if self._init_by_id else {},
        )

    @classmethod
    def from_obs(cls, obs) -> "GameState":
        """Build a GameState from a Kaggle observation (dict or Namespace)."""
        def _get(o, key, default=None):
            if isinstance(o, dict):
                return o.get(key, default)
            return getattr(o, key, default)

        step = _get(obs, "step", 0)
        raw_planets = _get(obs, "planets", [])
        raw_fleets = _get(obs, "fleets", [])
        angular_vel = _get(obs, "angular_velocity", 0.0)
        next_fid = _get(obs, "next_fleet_id", 0)
        comet_ids = set(_get(obs, "comet_planet_ids", []) or [])
        comets = list(_get(obs, "comets", []) or [])

        raw_init = _get(obs, "initial_planets", []) or []
        init_by_id: dict[int, tuple[float, float]] = {}
        for ip in raw_init:
            init_by_id[ip[0]] = (ip[2], ip[3])

        planets = []
        for p in raw_planets:
            pid, owner, x, y, radius, ships, prod = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
            ix, iy = init_by_id.get(pid, (x, y))
            planets.append(PlanetState(pid, owner, x, y, radius, ships, prod, ix, iy))
        
        fleets = [
            FleetState(f[0], f[1], f[2], f[3], f[4], f[5], f[6])
            for f in raw_fleets
        ]

        return cls(
            planets=planets,
            fleets=fleets,
            angular_velocity=angular_vel,
            step=step,
            next_fleet_id=next_fid,
            player_id=int(_get(obs, "player", 0)),
            comet_ids=comet_ids,
            comets=comets,
            _init_by_id=init_by_id if init_by_id else {},
        )

    def to_obs_dict(self) -> dict:
        """Convert to a plain dict suitable for agent consumption."""
        return {
            "planets": [
                [p.id, p.owner, p.x, p.y, p.radius, int(p.ships), p.production]
                for p in self.planets
            ],
            "fleets": [
                [f.id, f.owner, f.x, f.y, f.angle, f.from_planet_id, f.ships]
                for f in self.fleets
            ],
            "angular_velocity": self.angular_velocity,
            "step": self.step,
            "next_fleet_id": self.next_fleet_id,
            "comet_planet_ids": list(self.comet_ids),
            "comets": self.comets,
        }

    # ------------------------------------------------------------------
    # Numba / Flat State Conversion
    # ------------------------------------------------------------------

    def to_nb_arrays(self, max_fleets: int = 1000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert state to flat NumPy arrays for Numba consumption.
        
        Returns:
            planets_np: (N, 10) [id, owner, x, y, radius, ships, production, init_x, init_y, is_comet]
            fleets_np: (max_fleets, 8) [id, owner, x, y, angle, from_pid, ships, is_active]
            metadata_np: [angular_vel, step, next_fleet_id]
        """
        planets_np = np.zeros((100, 10), dtype=np.float64)
        for i, p in enumerate(self.planets):
            if i >= 100: break
            planets_np[i] = [
                p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production,
                p.init_x, p.init_y, 1.0 if p.id in self.comet_ids else 0.0
            ]
        # Pad remaining with neutral owner
        if len(self.planets) < 100:
            planets_np[len(self.planets):, 1] = -1
        
        fleets_np = np.zeros((max_fleets, 8), dtype=np.float64)
        for i, f in enumerate(self.fleets):
            if i >= max_fleets: break
            fleets_np[i] = [f.id, f.owner, f.x, f.y, f.angle, f.from_planet_id, f.ships, 1.0]
            
        metadata_np = np.array([self.angular_velocity, float(self.step), float(self.next_fleet_id), float(self.player_id)], dtype=np.float64)
        return planets_np, fleets_np, metadata_np

    @classmethod
    def from_nb_arrays(cls, planets_np, fleets_np, metadata_np, comets=None) -> "GameState":
        """Reconstruct GameState from Numba arrays."""
        planets = []
        comet_ids = set()
        for i in range(planets_np.shape[0]):
            p_data = planets_np[i]
            if p_data[4] <= 0.0:  # Skip padded/inactive/expired planets
                continue
            p = PlanetState(
                id=int(p_data[0]), owner=int(p_data[1]),
                x=p_data[2], y=p_data[3], radius=p_data[4],
                ships=p_data[5], production=int(p_data[6]),
                init_x=p_data[7], init_y=p_data[8]
            )
            planets.append(p)
            if p_data[9] > 0.5:
                comet_ids.add(p.id)
                
        fleets = []
        for i in range(fleets_np.shape[0]):
            f_data = fleets_np[i]
            if f_data[7] < 0.5: continue # inactive
            fleets.append(FleetState(
                id=int(f_data[0]), owner=int(f_data[1]),
                x=f_data[2], y=f_data[3], angle=f_data[4],
                from_planet_id=int(f_data[5]), ships=int(f_data[6])
            ))
            
        return cls(
            planets=planets,
            fleets=fleets,
            angular_velocity=metadata_np[0],
            step=int(metadata_np[1]),
            next_fleet_id=int(metadata_np[2]),
            comet_ids=comet_ids,
            comets=comets or [],
        )


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def fleet_speed_fm(ships: int) -> float:
    """Speed of a fleet with `ships` ships.

    Formula matches the official env:
        speed = 1 + (MAX_SPEED - 1) * (log(ships) / log(1000)) ** 1.5
    clamped to [1, MAX_SPEED].
    """
    if ships <= 1:
        return 1.0
    if ships >= 1000:
        return MAX_SPEED
    ratio = math.log(ships) / _LOG1000
    return min(1.0 + _SPEED_RANGE * ratio ** 1.5, MAX_SPEED)


def _fleet_speeds_np(ships_arr: np.ndarray) -> np.ndarray:
    """Vectorised fleet speed for a NumPy array of ship counts."""
    out = np.ones(len(ships_arr), dtype=np.float64)
    mask = ships_arr >= 1000
    out[mask] = MAX_SPEED
    valid = (~mask) & (ships_arr > 1)
    if valid.any():
        ratio = np.log(ships_arr[valid].astype(np.float64)) / _LOG1000
        out[valid] = np.clip(1.0 + _SPEED_RANGE * ratio ** 1.5, 1.0, MAX_SPEED)
    return out


def _point_segment_dist(px: float, py: float,
                        ax: float, ay: float,
                        bx: float, by: float) -> float:
    """Minimum distance from point (px, py) to segment (ax,ay)-(bx,by)."""
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    if l2 < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
    return math.hypot(px - ax - t * dx, py - ay - t * dy)


# ---------------------------------------------------------------------------
# Planning utilities
# ---------------------------------------------------------------------------

def is_terminal(state: GameState) -> bool:
    """True if the game is over (0 or 1 players alive, or step limit reached)."""
    alive: set[int] = set()
    for p in state.planets:
        if p.owner != -1:
            alive.add(p.owner)
    for f in state.fleets:
        alive.add(f.owner)
    
    res = len(alive) <= 1 or state.step >= 498
    return res


def winner(state: GameState) -> int:
    """Return the winning player id, or -1 if draw/in-progress."""
    scores: dict[int, float] = {}
    for p in state.planets:
        if p.owner != -1:
            scores[p.owner] = scores.get(p.owner, 0.0) + p.ships
    for f in state.fleets:
        scores[f.owner] = scores.get(f.owner, 0.0) + f.ships
    if not scores:
        return -1
    best = max(scores, key=lambda k: scores[k])
    # Check for ties
    top_score = scores[best]
    if sum(1 for v in scores.values() if v == top_score) > 1:
        return -1
    return best


def fleet_eta(src_x: float, src_y: float, src_r: float,
              tgt_x: float, tgt_y: float, ships: int) -> float:
    """Estimated turns for a fleet launched from src to reach tgt.

    Matches helpers.py fleet_eta logic: dist / speed, minimum 1.
    """
    dist = math.hypot(tgt_x - src_x, tgt_y - src_y) - src_r
    spd = fleet_speed_fm(max(1, ships))
    return max(1.0, dist / spd)


def fleet_eta_planets(src: PlanetState, tgt: PlanetState, ships: int) -> float:
    """Convenience wrapper taking PlanetState objects."""
    return fleet_eta(src.x, src.y, src.radius, tgt.x, tgt.y, ships)


def angle_to(src_x: float, src_y: float,
             tgt_x: float, tgt_y: float) -> float:
    """Angle (radians) from (src_x, src_y) to (tgt_x, tgt_y)."""
    return math.atan2(tgt_y - src_y, tgt_x - src_x)


def score_value(state: GameState, player_id: int) -> float:
    """Simple ship-count advantage for `player_id` vs everyone else.

    Returns value in (-1, 1) using tanh of the ratio.
    """
    own: float = 0.0
    total: float = 0.0
    for p in state.planets:
        if p.owner != -1:
            if p.owner == player_id:
                own += p.ships
            total += p.ships
    for f in state.fleets:
        if f.owner == player_id:
            own += f.ships
        total += f.ships
    if total < 1e-6:
        return 0.0
    return math.tanh((own / total - 0.5) * 4.0)


# ---------------------------------------------------------------------------
# Core simulation — mutable (internal / MCTS use)
# ---------------------------------------------------------------------------

def _simulate_step_mut(state: GameState, actions: dict[int, list[list]]) -> None:
    """Advance state IN-PLACE by one step.  Use state.copy() before calling
    when you need to preserve the original for backtracking.

    This avoids all per-step allocation beyond what is strictly necessary and
    is the fastest path for MCTS rollouts.
    """
    planets = state.planets
    fleets = state.fleets
    comet_ids = state.comet_ids
    comets = state.comets

    planet_map: dict[int, PlanetState] = {p.id: p for p in planets}

    # ------------------------------------------------------------------
    # 0. Expire comets (paths exhausted before this step)
    # ------------------------------------------------------------------
    expired: set[int] = set()
    for group in comets:
        idx = group.get("path_index", 0)
        pids = group["planet_ids"]
        for i, pid in enumerate(pids):
            if idx >= len(group["paths"][i]):
                expired.add(pid)
    if expired:
        state.planets = [p for p in planets if p.id not in expired]
        planets = state.planets
        state.comet_ids -= expired
        comet_ids = state.comet_ids
        for g in comets:
            g["planet_ids"] = [p for p in g["planet_ids"] if p not in expired]
        state.comets = [g for g in comets if g["planet_ids"]]
        comets = state.comets
        for pid in expired:
            planet_map.pop(pid, None)

    # ------------------------------------------------------------------
    # 1. Fleet Launch
    # ------------------------------------------------------------------
    nfid = state.next_fleet_id
    for player_id, moves in actions.items():
        if not moves:
            continue
        for move in moves:
            if len(move) != 3:
                continue
            from_pid, angle, ships = int(move[0]), float(move[1]), int(move[2])
            src = planet_map.get(from_pid)
            if src is None or src.owner != player_id or ships <= 0 or src.ships < ships:
                continue
            src.ships -= ships
            fleets.append(FleetState(
                id=nfid,
                owner=player_id,
                x=src.x + math.cos(angle) * (src.radius + 0.1),
                y=src.y + math.sin(angle) * (src.radius + 0.1),
                angle=angle,
                from_planet_id=from_pid,
                ships=ships,
            ))
            nfid += 1
    state.next_fleet_id = nfid

    # ------------------------------------------------------------------
    # 2. Production
    # ------------------------------------------------------------------
    for p in planets:
        if p.owner != -1:
            p.ships += p.production

    # ------------------------------------------------------------------
    # 3. Fleet movement — vectorised via NumPy when ≥2 fleets
    # ------------------------------------------------------------------
    combat_lists: dict[int, list[FleetState]] = {p.id: [] for p in planets}
    remove_ids: set[int] = set()

    nf = len(fleets)
    if nf == 0:
        pass
    elif nf >= 2:
        # Vectorised position update
        fx = np.array([f.x for f in fleets], dtype=np.float64)
        fy = np.array([f.y for f in fleets], dtype=np.float64)
        fa = np.array([f.angle for f in fleets], dtype=np.float64)
        fs = np.array([f.ships for f in fleets], dtype=np.float64)

        old_fx, old_fy = fx.copy(), fy.copy()
        speeds = _fleet_speeds_np(fs)
        fx += np.cos(fa) * speeds
        fy += np.sin(fa) * speeds

        # Write back positions
        for i, f in enumerate(fleets):
            f.x, f.y = fx[i], fy[i]

        # OOB check (vectorised)
        oob = (fx < 0) | (fx > BOARD_SIZE) | (fy < 0) | (fy > BOARD_SIZE)

        # Sun check (vectorised) — point-segment distance fleet path vs sun
        # Segment: old→new. Use parametric projection onto segment.
        dx_seg = fx - old_fx
        dy_seg = fy - old_fy
        l2 = dx_seg ** 2 + dy_seg ** 2
        # t = clamp(dot(sun-old, seg) / l2, 0, 1)
        dpx = CENTER - old_fx
        dpy = CENTER - old_fy
        t_raw = (dpx * dx_seg + dpy * dy_seg) / np.where(l2 > 1e-12, l2, 1.0)
        t = np.clip(t_raw, 0.0, 1.0)
        proj_x = old_fx + t * dx_seg
        proj_y = old_fy + t * dy_seg
        sun_dist = np.hypot(CENTER - proj_x, CENTER - proj_y)
        sun_hit = sun_dist < SUN_RADIUS

        # Mark OOB and sun-hit fleets for removal
        for i, f in enumerate(fleets):
            if oob[i] or sun_hit[i]:
                remove_ids.add(f.id)

        # Planet collision — vectorised per-planet to reduce Python-level loops
        if planets:
            nf_arr = len(fleets)
            removed_mask = np.zeros(nf_arr, dtype=bool)
            ax = old_fx
            ay = old_fy
            bx = fx
            by = fy
            dx_seg_p = bx - ax
            dy_seg_p = by - ay
            l2_p = dx_seg_p ** 2 + dy_seg_p ** 2
            # Bounding box of fleet movement (pre-filter to skip far-away planets)
            bbox_min_x = min(ax.min(), bx.min()) - 6.5  # MAX_SPEED + max radius
            bbox_max_x = max(ax.max(), bx.max()) + 6.5
            bbox_min_y = min(ay.min(), by.min()) - 6.5
            bbox_max_y = max(ay.max(), by.max()) + 6.5
            for p in planets:
                # Quick bounding-box rejection
                if (p.x + p.radius < bbox_min_x or p.x - p.radius > bbox_max_x
                    or p.y + p.radius < bbox_min_y or p.y - p.radius > bbox_max_y):
                    continue
                # Compute parametric projection t for all fleet segments
                dpx_p = p.x - ax
                dpy_p = p.y - ay
                t_raw_p = (dpx_p * dx_seg_p + dpy_p * dy_seg_p) / np.where(l2_p > 1e-12, l2_p, 1.0)
                t_p = np.clip(t_raw_p, 0.0, 1.0)
                proj_x_p = ax + t_p * dx_seg_p
                proj_y_p = ay + t_p * dy_seg_p
                dist_p = np.hypot(p.x - proj_x_p, p.y - proj_y_p)
                hit_mask = dist_p < p.radius
                if removed_mask.any():
                    hit_mask = hit_mask & (~removed_mask)
                hit_indices = np.nonzero(hit_mask)[0]
                for idx in hit_indices:
                    f = fleets[idx]
                    combat_lists[p.id].append(f)
                    remove_ids.add(f.id)
                    removed_mask[idx] = True
    else:
        # Single fleet — scalar path
        f = fleets[0]
        speed = fleet_speed_fm(f.ships)
        ox, oy = f.x, f.y
        f.x += math.cos(f.angle) * speed
        f.y += math.sin(f.angle) * speed
        
        hit_planet = False
        for p in planets:
            if _point_segment_dist(p.x, p.y, ox, oy, f.x, f.y) < p.radius:
                combat_lists[p.id].append(f)
                remove_ids.add(f.id)
                hit_planet = True
                break
                
        if not hit_planet:
            if not (0.0 <= f.x <= BOARD_SIZE and 0.0 <= f.y <= BOARD_SIZE):
                remove_ids.add(f.id)
            elif _point_segment_dist(CENTER, CENTER, ox, oy, f.x, f.y) < SUN_RADIUS:
                remove_ids.add(f.id)

    # ------------------------------------------------------------------
    # 4. Planet rotation + comet advancement + fleet sweep
    # ------------------------------------------------------------------
    new_step = state.step + 1

    def _sweep(p: PlanetState, ox: float, oy: float, nx: float, ny: float):
        if ox == nx and oy == ny:
            return
        for f in fleets:
            if f.id not in remove_ids:
                if _point_segment_dist(f.x, f.y, ox, oy, nx, ny) < p.radius:
                    combat_lists[p.id].append(f)
                    remove_ids.add(f.id)

    for p in planets:
        if p.id in comet_ids:
            continue
        ix, iy = state._init_by_id.get(p.id, (p.init_x, p.init_y))
        dx, dy = ix - CENTER, iy - CENTER
        orb_r = math.hypot(dx, dy)
        ox, oy = p.x, p.y
        if orb_r + p.radius < ROTATION_RADIUS_LIMIT:
            cur_angle = math.atan2(dy, dx) + state.angular_velocity * new_step
            p.x = CENTER + orb_r * math.cos(cur_angle)
            p.y = CENTER + orb_r * math.sin(cur_angle)
        if ox != p.x or oy != p.y:
            _sweep(p, ox, oy, p.x, p.y)

    expired_comets: set[int] = set()
    for group in comets:
        group["path_index"] = group.get("path_index", -1) + 1
        idx = group["path_index"]
        for i, pid in enumerate(group["planet_ids"]):
            cp = planet_map.get(pid)
            if cp is None:
                continue
            p_path = group["paths"][i]
            if idx >= len(p_path):
                expired_comets.add(pid)
            else:
                ox2, oy2 = cp.x, cp.y
                cp.x, cp.y = p_path[idx][0], p_path[idx][1]
                if ox2 >= 0 and (ox2 != cp.x or oy2 != cp.y):
                    _sweep(cp, ox2, oy2, cp.x, cp.y)

    if expired_comets:
        state.planets = [p for p in planets if p.id not in expired_comets]
        state.comet_ids -= expired_comets
        for g in comets:
            g["planet_ids"] = [p for p in g["planet_ids"] if p not in expired_comets]
        state.comets = [g for g in comets if g["planet_ids"]]
        planets = state.planets
        for pid in expired_comets:
            planet_map.pop(pid, None)

    state.fleets = [f for f in fleets if f.id not in remove_ids]
    fleets = state.fleets

    # ------------------------------------------------------------------
    # 5. Combat resolution
    # ------------------------------------------------------------------
    for pid, arriving in combat_lists.items():
        if not arriving:
            continue
        p = planet_map.get(pid)
        if p is None:
            continue
        ps: dict[int, int] = {}
        for f in arriving:
            ps[f.owner] = ps.get(f.owner, 0) + f.ships
        srt = sorted(ps.items(), key=lambda x: x[1], reverse=True)
        top_owner, top_ships = srt[0]
        if len(srt) > 1:
            second = srt[1][1]
            if top_ships == second:
                continue  # equal → no change
            survivor_ships = top_ships - second
            survivor_owner = top_owner
        else:
            survivor_ships = top_ships
            survivor_owner = top_owner
        if p.owner == survivor_owner:
            p.ships += survivor_ships
        else:
            p.ships -= survivor_ships
            if p.ships < 0:
                p.owner = survivor_owner
                p.ships = -p.ships

    state.step = new_step

    # Update init cache for any newly added planets
    ibc = state._init_by_id
    for p in state.planets:
        if p.id not in ibc:
            ibc[p.id] = (p.init_x, p.init_y)


# ---------------------------------------------------------------------------
# Public immutable API
# ---------------------------------------------------------------------------

def simulate_step(
    state: GameState,
    actions: dict[int, list[list]],
) -> GameState:
    """Advance the game by one step (immutable — returns new GameState)."""
    ns = state.copy()
    _simulate_step_mut(ns, actions)
    return ns


def simulate_n_steps(
    state: GameState,
    n: int,
    actions: dict[int, list[list]] | None = None,
) -> GameState:
    """Advance by exactly n steps with fixed (or no) actions.

    Useful for terminal-value estimation in MCTS rollouts where you want
    the final state after N idle steps without constructing an action stream.
    """
    ns = state.copy()
    acts = actions or {}
    for _ in range(n):
        if is_terminal(ns):
            break
        _simulate_step_mut(ns, acts)
    return ns


# ---------------------------------------------------------------------------
# Multi-step rollout with agent callables
# ---------------------------------------------------------------------------

def rollout(
    state: GameState,
    agents: dict[int, Any],
    max_steps: int = 500,
) -> list[GameState]:
    """Roll out a game using the given agent callables.

    Args:
        state:     Initial GameState (not mutated).
        agents:    Dict player_id → callable(obs_dict) → list of moves.
        max_steps: Maximum steps to simulate.

    Returns:
        List of GameStates (including initial state, including terminal).
    """
    ns = state.copy()
    history = [state]
    for _ in range(max_steps):
        if is_terminal(ns):
            break
        obs = ns.to_obs_dict()
        actions: dict[int, list] = {}
        for pid, agent_fn in agents.items():
            obs_copy = dict(obs)
            obs_copy["player"] = pid
            try:
                moves = agent_fn(obs_copy)
                actions[pid] = moves or []
            except Exception:
                actions[pid] = []
        _simulate_step_mut(ns, actions)
        history.append(ns.copy())
    return history


def final_scores(state: GameState, num_players: int = 4) -> list[int]:
    """Compute final ship-count scores for each player."""
    scores = [0] * num_players
    for p in state.planets:
        if 0 <= p.owner < num_players:
            scores[p.owner] += int(p.ships)
    for f in state.fleets:
        if 0 <= f.owner < num_players:
            scores[f.owner] += f.ships
    return scores


# ---------------------------------------------------------------------------
# T-08: Fleet-Arrival Garrison Projection
# ---------------------------------------------------------------------------

def projected_garrison(
    planet: "PlanetState",
    eta_turns: int,
    fleets: list["FleetState"],
    my_player: int,
) -> float:
    """Estimate garrison on `planet` after `eta_turns` turns.

    Accounts for:
    1. Natural production growth: +production per turn.
    2. Incoming enemy fleets that arrive <= eta_turns (subtracts ships; ownership
       flip is assumed to be subsumed into the garrison deficit calculation).
    3. Incoming friendly fleets that arrive <= eta_turns (adds ships if same owner).

    Planet owner changes during transit are NOT modelled (pure garrison number only).
    Use this as a lightweight approximation to improve `_score_target` accuracy.

    Args:
        planet:     The target PlanetState.
        eta_turns:  Estimated turns until our fleet arrives (integer).
        fleets:     All FleetState objects currently in transit.
        my_player:  Our player ID (used to distinguish friendly vs enemy).

    Returns:
        Projected garrison as a float (may be negative if enemy fleets dominate).
    """
    garrison = float(planet.ships)
    garrison += planet.production * eta_turns  # natural growth

    for f in fleets:
        dest_planet_id = getattr(f, "dest_planet_id", getattr(f, "from_planet_id", None))
        if dest_planet_id != planet.id:
            continue
        # ETA of this fleet: distance remaining / speed
        dist = math.hypot(f.x - planet.x, f.y - planet.y)
        spd = fleet_speed_fm(f.ships)
        fleet_arrival = dist / max(spd, 0.01)
        if fleet_arrival > eta_turns:
            continue  # arrives after we do — not relevant
        if f.owner == my_player:
            garrison += f.ships   # reinforcement
        else:
            garrison -= f.ships   # enemy incoming

    return garrison

# ---------------------------------------------------------------------------
# Numba Optimized Engine (
# ---------------------------------------------------------------------------


@njit
def _nb_fleet_speed(ships: float) -> float:
    if ships <= 1.0: return 1.0
    if ships >= 1000.0: return 6.0
    # speed = 1 + 5 * (log(ships) / log(1000)) ** 1.5
    return 1.0 + 5.0 * (math.log(ships) / 6.907755278982137) ** 1.5


@njit
def _nb_point_seg_dist(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    if l2 < 1e-12:
        return math.sqrt((px - ax)**2 + (py - ay)**2)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
    return math.sqrt((px - (ax + t * dx))**2 + (py - (ay + t * dy))**2)


@njit
def nb_simulate_step_mut(planets: np.ndarray, fleets: np.ndarray, metadata: np.ndarray, table: np.ndarray) -> None:
    """Numba-optimized simulation step. Mutates arrays in-place.
    
    planets: (N, 10) [id, owner, x, y, radius, ships, production, init_x, init_y, is_comet]
    fleets: (M, 8) [id, owner, x, y, angle, from_pid, ships, is_active]
    metadata: [angular_vel, step, next_fleet_id]
    """
    angular_vel = metadata[0]
    step = int(metadata[1])
    new_step = step + 1
    metadata[1] = float(new_step)
    
    # 1. Production
    for i in range(planets.shape[0]):
        if planets[i, 1] != -1: # not neutral
            planets[i, 5] += planets[i, 6]
            
    # 2. Movement & Collisions
    # Pre-calculate combat counts for each planet [4 players]
    # We use a 2D array (N_planets, 4) to store arriving ships per player
    combat_buffer = np.zeros((planets.shape[0], 4), dtype=np.float64)
    
    for i in range(fleets.shape[0]):
        if fleets[i, 7] < 0.5: continue # inactive
        
        # Move
        ox, oy = fleets[i, 2], fleets[i, 3]
        spd = _nb_fleet_speed(fleets[i, 6])
        angle = fleets[i, 4]
        nx = ox + math.cos(angle) * spd
        ny = oy + math.sin(angle) * spd
        
        # Planet Collision Check FIRST
        hit_planet_idx = -1
        for j in range(planets.shape[0]):
            if planets[j, 4] <= 0:
                continue
            if _nb_point_seg_dist(planets[j, 2], planets[j, 3], ox, oy, nx, ny) < planets[j, 4]:
                hit_planet_idx = j
                break
        
        if hit_planet_idx != -1:
            owner = int(fleets[i, 1])
            if 0 <= owner < 4:
                combat_buffer[hit_planet_idx, owner] += fleets[i, 6]
            fleets[i, 7] = 0.0 # deactivate
        else:
            # OOB or Sun check
            if nx < 0 or nx > 100 or ny < 0 or ny > 100 or \
               _nb_point_seg_dist(50.0, 50.0, ox, oy, nx, ny) < SUN_RADIUS:
                fleets[i, 7] = 0.0
            else:
                fleets[i, 2], fleets[i, 3] = nx, ny
            
    # 3. Rotation & Comet advancement
    for i in range(planets.shape[0]):
        if planets[i, 4] <= 0: continue # Skip zombies
        p_id = int(planets[i, 0])
        s_idx = min(new_step, MAX_STEPS)
        
        ox, oy = planets[i, 2], planets[i, 3]
        nx = table[p_id, s_idx, 0]
        ny = table[p_id, s_idx, 1]
        
        # Check if comet has expired
        if planets[i, 9] > 0.5 and nx < -50.0:
            planets[i, 4] = 0.0 # deactivate (radius = 0)
            continue
            
        planets[i, 2] = nx
        planets[i, 3] = ny
        
        # Fleet sweep (check if moving planet segment hits any active fleets)
        if ox != nx or oy != ny:
            for j in range(fleets.shape[0]):
                if fleets[j, 7] < 0.5: continue # inactive
                fx, fy = fleets[j, 2], fleets[j, 3]
                if _nb_point_seg_dist(fx, fy, ox, oy, nx, ny) < planets[i, 4]:
                    owner = int(fleets[j, 1])
                    if 0 <= owner < 4:
                        combat_buffer[i, owner] += fleets[j, 6]
                    fleets[j, 7] = 0.0
            
    # 4. Resolve Combat
    for i in range(planets.shape[0]):
        # Check if anyone arrived
        any_arrived = False
        for p_idx in range(4):
            if combat_buffer[i, p_idx] > 0:
                any_arrived = True
                break
        if not any_arrived: continue
        
        # Sort arrivals to find top and second
        # Since only 4 players, we can do a simple manual sort or just find max
        top_owner = -1
        top_ships = -1.0
        second_ships = -1.0
        
        for p_idx in range(4):
            s = combat_buffer[i, p_idx]
            if s > top_ships:
                second_ships = top_ships
                top_ships = s
                top_owner = p_idx
            elif s > second_ships:
                second_ships = s
        
        survivor_ships = 0.0
        if top_ships > second_ships:
            survivor_ships = top_ships - second_ships
            survivor_owner = top_owner
            
            # Apply to planet
            if int(planets[i, 1]) == survivor_owner:
                planets[i, 5] += survivor_ships
            else:
                planets[i, 5] -= survivor_ships
                if planets[i, 5] < 0:
                    planets[i, 1] = float(survivor_owner)
                    planets[i, 5] = -planets[i, 5]


@njit
def nb_greedy_policy(planets: np.ndarray, fleets: np.ndarray, metadata: np.ndarray) -> None:
    """Executes a simple 'Capture Nearest' policy for all players.
    
    Used for adversarial rollouts in MCTS.
    """
    next_id = int(metadata[2])
    for p_idx in range(planets.shape[0]):
        owner = int(planets[p_idx, 1])
        ships = planets[p_idx, 5]
        # Skip only neutral planets (metadata[3] is the agent, but we want it to move too)
        if owner == -1 or ships < 10.0: continue
        
        # Only move if we have enough ships and not too many fleets already
        # (Simplified check for Numba)
        
        # Find a target (with some randomness for rollout diversity)
        best_target = -1
        min_dist = 1000.0
        
        # Randomly skip some planets to diversify paths
        for t_idx in range(planets.shape[0]):
            if planets[t_idx, 1] != owner:
                if np.random.random() < 0.3: continue 
                dx = planets[t_idx, 2] - planets[p_idx, 2]
                dy = planets[t_idx, 3] - planets[p_idx, 3]
                d = math.sqrt(dx*dx + dy*dy)
                if d < min_dist:
                    min_dist = d
                    best_target = t_idx
        
        if best_target != -1 and min_dist < 50.0:
            # Send 40% of ships
            send = ships * 0.4
            planets[p_idx, 5] -= send
            
            # Find an inactive fleet slot
            for f_idx in range(fleets.shape[0]):
                if fleets[f_idx, 7] < 0.5:
                    angle = math.atan2(planets[best_target, 3] - planets[p_idx, 3],
                                       planets[best_target, 2] - planets[p_idx, 2])
                    fleets[f_idx] = [float(next_id), float(owner), 
                                     planets[p_idx, 2], planets[p_idx, 3],
                                     angle, float(planets[p_idx, 0]), send, 1.0]
                    next_id += 1
                    break
    metadata[2] = float(next_id)


@njit
def nb_simulate_n_steps(planets: np.ndarray, fleets: np.ndarray, metadata: np.ndarray, table: np.ndarray, n: int, adversarial: bool = False) -> None:
    """Run n steps of simulation using Numba."""
    for i in range(n):
        if adversarial and i % 10 == 0:
            nb_greedy_policy(planets, fleets, metadata)
            
        nb_simulate_step_mut(planets, fleets, metadata, table)
        if metadata[1] >= 498: break


@njit
def nb_simulate_batch(planets_batch, fleets_batch, metadata_batch, table, n, adversarial: bool = False):
    """Run n steps for a batch of simulations."""
    for b in range(planets_batch.shape[0]):
        nb_simulate_n_steps(planets_batch[b], fleets_batch[b], metadata_batch[b], table, n, adversarial)

@njit
def nb_evaluate_batch(planets_batch, fleets_batch, player_id, step_batch, weights):
    """Evaluate a batch of states."""
    num_samples = planets_batch.shape[0]
    rewards = np.zeros(num_samples, dtype=np.float64)
    for b in range(num_samples):
        rewards[b] = nb_evaluate(planets_batch[b], fleets_batch[b], player_id, int(step_batch[b]), weights)
    return rewards

# Wrap public calls to use AOT if available
def nb_simulate_batch_aot(p, f, m, n, adversarial=False):
    table = PLANET_POS_TABLE
    if _JAX_ENGINE and not adversarial:
        try:
            p_new, f_new, m_new = _JAX_ENGINE.jax_simulate_batch(p, f, m, table, n)
            p[:] = np.array(p_new)
            f[:] = np.array(f_new)
            m[:] = np.array(m_new)
            return
        except Exception as e:
            sys.stderr.write(f"JAX fallback: {e}\n")
    
    if _AOT_ENGINE:
        # AOT functions take positional arguments only
        try:
            _AOT_ENGINE.nb_simulate_batch(p, f, m, table, n, adversarial)
            return
        except:
            pass # Fallback to JIT
    
    nb_simulate_batch(p, f, m, table, n, adversarial)

def nb_evaluate_batch_aot(p, f, player_id, step_batch, weights):
    if _JAX_ENGINE:
        try:
            return np.array(_JAX_ENGINE.jax_evaluate_batch(p, f, player_id, step_batch, weights))
        except:
            pass

    if _AOT_ENGINE:
        return _AOT_ENGINE.nb_evaluate_batch(p, f, player_id, step_batch, weights)
    else:
        return nb_evaluate_batch(p, f, player_id, step_batch, weights)

def nb_evaluate_batch_custom_aot(planets_batch, fleets_batch, player_id, current_step, step_batch, early_expand_bonus, fleet_burden_threshold, fleet_burden_penalty):
    if _AOT_ENGINE:
        try:
            return _AOT_ENGINE.nb_evaluate_batch_custom(planets_batch, fleets_batch, player_id, current_step, step_batch, early_expand_bonus, fleet_burden_threshold, fleet_burden_penalty)
        except Exception:
            pass
    return nb_evaluate_batch_custom(planets_batch, fleets_batch, player_id, current_step, step_batch, early_expand_bonus, fleet_burden_threshold, fleet_burden_penalty)



@njit
def _nb_evaluate_single(planets, fleets, player_id, step, weights):
    """Internal single-state evaluator for batching. 
    (Copy of the logic from agent_v980 to avoid circular imports or to keep it in the engine).
    """
    pw, sw, tp, eb, urgency_alpha, intent_threshold, leader_penalty, centrality_weight, roi_threshold, peace_weight = weights
    score = 0.0
    
    # [Master Intervention] Gravity Shielding: Identify leader planets for shield check
    leader_pos_x = -100.0
    leader_pos_y = -100.0
    
    # 1. Identify the Leader
    totals = np.zeros(4, dtype=np.float64)
    for i in range(planets.shape[0]):
        owner = int(planets[i, 1])
        if owner != -1:
            totals[owner] += planets[i, 6] * pw + planets[i, 5] * sw
    
    leader_id = -1
    max_total = -1.0
    for i in range(4):
        if i != player_id and totals[i] > max_total:
            max_total = totals[i]
            leader_id = i
            
    # Find leader's most productive planet as the "threat origin" for shielding
    if leader_id != -1:
        max_p = -1.0
        for i in range(planets.shape[0]):
            if planets[i, 1] == leader_id and planets[i, 6] > max_p:
                max_p = planets[i, 6]
                leader_pos_x, leader_pos_y = planets[i, 2], planets[i, 3]

    # 2. Planet Scores
    for i in range(planets.shape[0]):
        px, py = planets[i, 2], planets[i, 3]
        dist_center = math.sqrt((px - 50.0)**2 + (py - 50.0)**2)
        centrality = (75.0 - dist_center) / 75.0
        
        if planets[i, 1] == player_id:
            min_dist_to_action = 100.0
            for j in range(planets.shape[0]):
                if planets[j, 1] != player_id:
                    d = math.sqrt((planets[i, 2]-planets[j, 2])**2 + (planets[i, 3]-planets[j, 3])**2)
                    if d < min_dist_to_action:
                        min_dist_to_action = d
            
            # [Phase 2] URGENCY_DECAY: Penalty is higher early on
            decay = max(0.1, 1.0 - (step / 500.0))
            urgency_penalty = planets[i, 6] * (min_dist_to_action * urgency_alpha) * decay
            
            # Anti-Stalling Hoarding Penalty: penalize excessive garrisons that aren't launching
            hoard_ratio = planets[i, 5] / max(1.0, planets[i, 6] * 10.0)
            hoard_penalty = 0.0
            if hoard_ratio > 1.0:
                hoard_penalty = (hoard_ratio - 1.0) * planets[i, 6] * 0.5
                
            score -= (urgency_penalty + hoard_penalty)
            
            # [Master Intervention] Gravity Shielding Bonus
            shield_bonus = 0.0
            if leader_id != -1 and nb_path_crosses_sun(px, py, leader_pos_x, leader_pos_y):
                shield_bonus = planets[i, 6] * 10.0 # Significant reward for staying behind the sun
            
            score += planets[i, 5] * sw + planets[i, 6] * pw + centrality * centrality_weight + shield_bonus
        elif planets[i, 1] != -1:
            val = planets[i, 5] * sw + planets[i, 6] * pw + centrality * centrality_weight
            penalty = 1.0
            if int(planets[i, 1]) == leader_id: 
                penalty = leader_penalty
            else:
                penalty = peace_weight # [AG4] Peace logic: less penalty for attacking non-leaders? 
                # Wait, actually peace_weight should REDUCE aggression on non-leaders.
                # If penalty is lower, we are LESS likely to want to "take" it if we use negative value.
                # If penalty = 0.5, then score -= val * 0.5. This means it's "cheaper" to let them keep it? 
                # No, if we want to avoid attacking them, we should reduce the "reward" of taking it.
                # In this evaluator, enemy planets give negative score. 
                # So -val * penalty. If penalty is 0.5, the negative score is LESS negative.
                # This means we "care less" about them having it. Correct.
            score -= val * penalty
        else:
            # Neutral planet
            remaining = 500 - step
            gain = planets[i, 6] * remaining
            if gain > 50.0 * roi_threshold: # [AG4] ROI check
                score += planets[i, 6] * pw * eb * (remaining / 500.0)
                
    # 3. Fleet Scores
    for i in range(fleets.shape[0]):
        if fleets[i, 7] < 0.5: continue # inactive
        owner = int(fleets[i, 1])
        if owner == player_id:
            # Fleet ships are retained in score, plus a small "in-transit" active-action bonus
            score += fleets[i, 6] * sw * 1.05
        elif owner != -1:
            penalty = leader_penalty if owner == leader_id else peace_weight
            score -= fleets[i, 6] * sw * penalty
            
    # [Master Intervention] Deterministic Strategic Noise
    # Prevents opponent simulation counters while maintaining MCTS stability
    h = (int(score * 1000) ^ (step * 104729)) & 0xFFFFFFFF
    noise = ((h % 200) - 100) / 10000.0 # -1% to +1%
    score *= (1.0 + noise)
                
    return score

@njit
def nb_evaluate(planets, fleets, player_id, step, weights):
    return _nb_evaluate_single(planets, fleets, player_id, step, weights)

# ---------------------------------------------------------------------------
# JIT Warm-up (Ensures first turn doesn't timeout)
# ---------------------------------------------------------------------------

def nb_warmup():
    """Trigger JIT compilation for core functions."""
    p = np.zeros((1, 1, 10), dtype=np.float64)
    f = np.zeros((1, 1, 8), dtype=np.float64)
    m = np.zeros((1, 4), dtype=np.float64)
    w = np.zeros(10, dtype=np.float64)
    s = np.zeros(1, dtype=np.float64)
    
    # Warmup with full rollout depth to avoid mid-game JIT spikes
    try:
        nb_simulate_batch_aot(p, f, m, 128)
        nb_evaluate_batch_aot(p, f, 0, s, w)
        nb_evaluate_batch_custom_aot(p, f, 0, 1, s, 0.3, 5.0, 0.5)
    except:
        pass

    if _JAX_ENGINE:
        try:
            _JAX_ENGINE.warm_up()
        except:
            pass
# ---------------------------------------------------------------------------
# JIT Warm-up (Ensures first turn doesn't timeout)
# ---------------------------------------------------------------------------


@njit
def nb_is_terminal(planets, fleets, step):
    """JIT-compiled terminal check matching standard rules."""
    if step >= 498:
        return True
    alive = np.zeros(4, dtype=np.bool_)
    for i in range(planets.shape[0]):
        if planets[i, 4] <= 0:
            continue
        owner = int(planets[i, 1])
        if 0 <= owner < 4:
            alive[owner] = True
    for i in range(fleets.shape[0]):
        if fleets[i, 7] < 0.5:
            continue
        owner = int(fleets[i, 1])
        if 0 <= owner < 4:
            alive[owner] = True
    num_alive = 0
    for p_idx in range(4):
        if alive[p_idx]:
            num_alive += 1
    return num_alive <= 1


@njit
def nb_evaluate_state_custom(planets, fleets, player_id, current_step, step, early_expand_bonus, fleet_burden_threshold, fleet_burden_penalty):
    """JIT-compiled state evaluator matching evaluate_state in Python exactly."""
    if nb_is_terminal(planets, fleets, step):
        alive_planets = np.zeros(4, dtype=np.bool_)
        for i in range(planets.shape[0]):
            if planets[i, 4] <= 0:
                continue
            owner = int(planets[i, 1])
            if 0 <= owner < 4:
                alive_planets[owner] = True
        num_alive_planets = 0
        for p_idx in range(4):
            if alive_planets[p_idx]:
                num_alive_planets += 1
        if num_alive_planets == 1 and alive_planets[player_id]:
            return 100000.0
        if not alive_planets[player_id]:
            return -100000.0

    score = 0.0
    my_planet_ships = 0.0
    my_fleet_ships = 0.0

    for i in range(planets.shape[0]):
        if planets[i, 4] <= 0:
            continue
        p_owner = int(planets[i, 1])
        p_ships = planets[i, 5]
        p_prod = planets[i, 6]

        val = p_ships * 1.0 + p_prod * 5.0
        if current_step < 100 and p_owner == -1:
            val += p_prod * 5.0 * early_expand_bonus

        if p_owner == player_id:
            score += val
            my_planet_ships += p_ships
        elif p_owner != -1:
            score -= val

    for i in range(fleets.shape[0]):
        if fleets[i, 7] < 0.5:
            continue
        f_owner = int(fleets[i, 1])
        f_ships = fleets[i, 6]

        f_val = f_ships * 1.0
        if f_owner == player_id:
            score += f_val
            my_fleet_ships += f_ships
        else:
            score -= f_val * 2.5

    if my_planet_ships > 0 and my_fleet_ships > my_planet_ships * fleet_burden_threshold:
        score *= fleet_burden_penalty

    return score


@njit
def nb_evaluate_batch_custom(planets_batch, fleets_batch, player_id, current_step, step_batch, early_expand_bonus, fleet_burden_threshold, fleet_burden_penalty):
    """JIT-compiled batch evaluator."""
    num_samples = planets_batch.shape[0]
    rewards = np.zeros(num_samples, dtype=np.float64)
    for b in range(num_samples):
        rewards[b] = nb_evaluate_state_custom(
            planets_batch[b], fleets_batch[b], player_id, current_step, int(step_batch[b]),
            early_expand_bonus, fleet_burden_threshold, fleet_burden_penalty
        )
    return rewards

