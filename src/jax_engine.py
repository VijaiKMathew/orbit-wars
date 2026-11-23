import jax
import jax.numpy as jnp
import numpy as np

# Auto-detect backend dtype: Metal needs float32, CPU can use float64
_JAX_DEVICES = jax.devices()
_ON_METAL = any(d.platform.upper() == "METAL" for d in _JAX_DEVICES)
FLOAT_DTYPE = jnp.float32 if _ON_METAL else jnp.float64
NP_FLOAT_DTYPE = np.float32 if _ON_METAL else np.float64

# Physical Constants
CENTER = 50.0
SUN_RADIUS = 10.0
MAX_SPEED = 6.0
ANGULAR_VEL = 0.01

@jax.jit
def get_fleet_speed(ships):
    log_ships = jnp.log(jnp.maximum(1.0, ships))
    ratio = log_ships / 6.907755278982137
    speed = 1.0 + 5.0 * (ratio ** 1.5)
    speed = jnp.where(ships <= 1.0, 1.0, speed)
    speed = jnp.where(ships >= 1000.0, 6.0, speed)
    return jnp.clip(speed, 1.0, 6.0)

@jax.jit
def point_seg_dist_sq(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    l2 = dx*dx + dy*dy
    t = jnp.where(l2 < 1e-12, 0.0, jnp.clip(((px - x1) * dx + (py - y1) * dy) / l2, 0.0, 1.0))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return (px - proj_x)**2 + (py - proj_y)**2

@jax.jit
def jax_simulate_step(planets, fleets, metadata, pos_table):
    """
    planets: (N, 10) [id, owner, x, y, radius, ships, prod, ix, iy, comet]
    fleets: (M, 8) [id, owner, x, y, angle, from_pid, ships, active]
    metadata: (3,) [ang_vel, step, next_id]
    """
    ang_vel, step, next_id = metadata[:3]
    new_step = step + 1
    
    # 1. Production
    owners = planets[:, 1]
    prod = planets[:, 6]
    ships = planets[:, 5]
    new_ships = ships + jnp.where(owners != -1, prod, 0.0)
    planets = planets.at[:, 5].set(new_ships)
    
    # 2. Movement
    f_active = fleets[:, 7] > 0.5
    f_ships = fleets[:, 6]
    f_speed = get_fleet_speed(f_ships)
    f_angle = fleets[:, 4]
    
    ox, oy = fleets[:, 2], fleets[:, 3]
    nx = ox + jnp.where(f_active, jnp.cos(f_angle) * f_speed, 0.0)
    ny = oy + jnp.where(f_active, jnp.sin(f_angle) * f_speed, 0.0)
    
    # Sun / OOB Check
    dist_sun_sq = point_seg_dist_sq(CENTER, CENTER, ox, oy, nx, ny)
    killed = ((nx < 0) | (nx > 100) | (ny < 0) | (ny > 100) | (dist_sun_sq < SUN_RADIUS**2)) & f_active
    
    # Planet Collision
    # (M, N) distance matrix
    def dist_to_all_planets(fx, fy, fnx, fny):
        return jax.vmap(lambda px, py: point_seg_dist_sq(px, py, fx, fy, fnx, fny))(planets[:, 2], planets[:, 3])
    
    all_dists_sq = jax.vmap(dist_to_all_planets)(ox, oy, nx, ny)
    collision_mask = (all_dists_sq < (planets[:, 4]**2)) & f_active[:, None] & (planets[:, 4] > 0.0)[None, :]
    
    is_hit_any = jnp.any(collision_mask, axis=1)
    hit_planet_idx = jnp.argmax(collision_mask, axis=1)
    
    # Update Fleets
    new_f_active = fleets[:, 7] * (1.0 - killed.astype(jnp.float32)) * (1.0 - is_hit_any.astype(jnp.float32))
    fleets = fleets.at[:, 2].set(jnp.where(is_hit_any | killed, ox, nx))
    fleets = fleets.at[:, 3].set(jnp.where(is_hit_any | killed, oy, ny))
    fleets = fleets.at[:, 7].set(new_f_active)
    
    # 3. Rotation (Vectorized using O(1) lookup table) & Sweep Check
    # Use new_step (clamped to MAX_STEPS)
    step_idx = jnp.clip(new_step, 0, 500).astype(jnp.int32)
    p_ids = planets[:, 0].astype(jnp.int32)
    
    rot_x = pos_table[p_ids, step_idx, 0]
    rot_y = pos_table[p_ids, step_idx, 1]
    
    is_zombie = (planets[:, 4] <= 0.0)
    is_comet = (planets[:, 9] > 0.5)
    comet_expired = is_comet & (rot_x < -50.0)
    is_zombie_after = is_zombie | comet_expired
    
    px_old, py_old = planets[:, 2], planets[:, 3]
    px_new = jnp.where(is_zombie_after, px_old, rot_x)
    py_new = jnp.where(is_zombie_after, py_old, rot_y)
    
    # Distance from all fleets (points) to each planet segment
    def dist_sweep_single_planet(pxo, pyo, pxn, pyn):
        return jax.vmap(lambda f_x, f_y: point_seg_dist_sq(f_x, f_y, pxo, pyo, pxn, pyn))(fleets[:, 2], fleets[:, 3])
        
    sweep_dists_sq = jax.vmap(dist_sweep_single_planet)(px_old, py_old, px_new, py_new) # shape: (N, M)
    
    # Check if planet actually moved
    planet_moved = (px_old != px_new) | (py_old != py_new)
    
    # Sweep collision matrix of shape (N, M)
    f_active_current = fleets[:, 7] > 0.5 # shape: (M,)
    sweep_collision_mask = (sweep_dists_sq < (planets[:, 4]**2)[:, None]) & \
                           planet_moved[:, None] & \
                           (~is_zombie_after)[:, None] & \
                           f_active_current[None, :]
                           
    fleet_sweep_hit = jnp.any(sweep_collision_mask, axis=0) # shape: (M,)
    fleet_sweep_planet_idx = jnp.argmax(sweep_collision_mask, axis=0) # shape: (M,)
    
    # Update fleet active status for sweep hits
    new_f_active_sweep = fleets[:, 7] * (1.0 - fleet_sweep_hit.astype(jnp.float32))
    fleets = fleets.at[:, 7].set(new_f_active_sweep)
    
    # Update planets
    planets = planets.at[:, 4].set(jnp.where(comet_expired, 0.0, planets[:, 4]))
    planets = planets.at[:, 2].set(px_new)
    planets = planets.at[:, 3].set(py_new)
    
    # 4. Combat Resolution (Numba-Parity Survivor Logic)
    combat_buffer = jnp.zeros((planets.shape[0], 4))
    f_owners = fleets[:, 1].astype(jnp.int32)
    f_hit_idx = hit_planet_idx.astype(jnp.int32)
    
    # Sum arrivals (only for players 0-3)
    f_is_player = (f_owners >= 0) & (f_owners <= 3)
    valid_f_owners = jnp.clip(f_owners, 0, 3)
    
    # Direct hits
    combat_buffer = combat_buffer.at[f_hit_idx, valid_f_owners].add(jnp.where(is_hit_any & f_is_player & f_active, fleets[:, 6], 0.0))
    # Sweep hits
    combat_buffer = combat_buffer.at[fleet_sweep_planet_idx, valid_f_owners].add(jnp.where(fleet_sweep_hit & f_is_player & f_active_current, fleets[:, 6], 0.0))
    
    # Find top attacker
    sorted_arrivals = jnp.sort(combat_buffer, axis=1)
    top1 = sorted_arrivals[:, 3]
    top2 = sorted_arrivals[:, 2]
    
    survivor_ships = top1 - top2
    survivor_owner = jnp.argmax(combat_buffer, axis=1).astype(jnp.float32)
    
    # Apply to planet
    p_owner = planets[:, 1]
    p_ships = planets[:, 5]
    
    is_same_owner = (p_owner == survivor_owner)
    new_ships_val = jnp.where(is_same_owner, p_ships + survivor_ships, p_ships - survivor_ships)
    
    # Handle capture
    owner_changed = (new_ships_val < 0)
    final_p_owner = jnp.where(owner_changed, survivor_owner, p_owner)
    final_p_ships = jnp.abs(new_ships_val)
    
    # Only update if someone arrived
    any_arrived = (top1 > 0)
    planets = planets.at[:, 1].set(jnp.where(any_arrived, final_p_owner, p_owner))
    planets = planets.at[:, 5].set(jnp.where(any_arrived, final_p_ships, p_ships))
    
    new_metadata = metadata.at[1].set(new_step)
    
    return planets, fleets, new_metadata

from functools import partial

@partial(jax.jit, static_argnums=(4,))
def jax_simulate_batch(planets_batch, fleets_batch, metadata_batch, pos_table, n):
    """Run n steps for a batch of states using vmap and scan."""
    def step_fn(carry, _):
        p, f, m = carry
        # Vectorize across the batch dimension
        p_new, f_new, m_new = jax.vmap(jax_simulate_step, in_axes=(0, 0, 0, None))(p, f, m, pos_table)
        return (p_new, f_new, m_new), None
    
    (planets_batch, fleets_batch, metadata_batch), _ = jax.lax.scan(
        step_fn, (planets_batch, fleets_batch, metadata_batch), jnp.arange(n)
    )
    return planets_batch, fleets_batch, metadata_batch

@jax.jit
def jax_evaluate_single(planets, fleets, player_id, step, weights):
    """Vectorized evaluation of a single state."""
    pw, sw, tp, eb, urgency_alpha, intent_threshold, leader_penalty, centrality_weight, roi_threshold, peace_weight = weights
    
    p_owners = planets[:, 1]
    p_ships = planets[:, 5]
    p_prod = planets[:, 6]
    p_x, p_y = planets[:, 2], planets[:, 3]
    
    # 1. Identify Leader
    p_val = p_prod * pw + p_ships * sw
    
    def get_player_total(pid):
        return jnp.where(p_owners == pid, p_val, 0.0).sum()
    
    totals = jax.vmap(get_player_total)(jnp.arange(4.0))
    totals_masked = jnp.where(jnp.arange(4.0) == player_id, -1.0, totals)
    max_total = jnp.max(totals_masked)
    leader_id = jnp.argmax(totals_masked)
    
    is_leader_planet = (p_owners == leader_id)
    leader_prod_masked = jnp.where(is_leader_planet, p_prod, -1.0)
    best_leader_planet_idx = jnp.argmax(leader_prod_masked)
    
    has_leader = (max_total > 0.0)
    leader_pos_x = jnp.where(has_leader, p_x[best_leader_planet_idx], -100.0)
    leader_pos_y = jnp.where(has_leader, p_y[best_leader_planet_idx], -100.0)
    
    # 2. Planet Scores
    dist_center = jnp.sqrt((p_x - 50.0)**2 + (p_y - 50.0)**2)
    centrality = (75.0 - dist_center) / 75.0
    
    dx = p_x[:, None] - p_x[None, :]
    dy = p_y[:, None] - p_y[None, :]
    dist_matrix = jnp.sqrt(dx**2 + dy**2)
    
    is_mine = (p_owners == player_id)
    is_not_mine_all = (p_owners != player_id)
    
    dist_to_action = jnp.where(is_not_mine_all[None, :], dist_matrix, 100.0)
    min_dist_to_action = jnp.min(dist_to_action, axis=1)
    
    decay = jnp.maximum(0.1, 1.0 - (step / 500.0))
    urgency_penalty = p_prod * (min_dist_to_action * urgency_alpha) * decay
    
    hoard_ratio = p_ships / jnp.maximum(1.0, p_prod * 10.0)
    hoard_penalty = jnp.where(hoard_ratio > 1.0, (hoard_ratio - 1.0) * p_prod * 0.5, 0.0)
    
    # Gravity Shielding
    dist_sun_sq = jax.vmap(lambda px, py: point_seg_dist_sq(50.0, 50.0, px, py, leader_pos_x, leader_pos_y))(p_x, p_y)
    path_crosses_sun = dist_sun_sq < SUN_RADIUS**2
    shield_bonus = jnp.where(has_leader & path_crosses_sun, p_prod * 10.0, 0.0)
    
    my_planet_score = p_ships * sw + p_prod * pw + centrality * centrality_weight + shield_bonus - urgency_penalty - hoard_penalty
    my_planet_total = jnp.where(is_mine, my_planet_score, 0.0).sum()
    
    enemy_base_val = p_ships * sw + p_prod * pw + centrality * centrality_weight
    is_leader = (p_owners == leader_id)
    penalty = jnp.where(is_leader, leader_penalty, peace_weight)
    enemy_planet_total = jnp.where((p_owners != -1) & ~is_mine, -enemy_base_val * penalty, 0.0).sum()
    
    remaining = 500.0 - step
    gain = p_prod * remaining
    neutral_val = p_prod * pw * eb * (remaining / 500.0)
    neutral_planet_total = jnp.where((p_owners == -1) & (gain > 50.0 * roi_threshold), neutral_val, 0.0).sum()
    
    # 3. Fleet Scores
    f_owners = fleets[:, 1]
    f_ships = fleets[:, 6]
    f_active = fleets[:, 7] > 0.5
    
    my_fleet_score = jnp.where(f_active & (f_owners == player_id), f_ships * sw * 1.05, 0.0).sum()
    f_penalty = jnp.where(f_owners == leader_id, leader_penalty, peace_weight)
    enemy_fleet_score = jnp.where(f_active & (f_owners != player_id) & (f_owners != -1), -f_ships * sw * f_penalty, 0.0).sum()
    
    total_score = my_planet_total + enemy_planet_total + neutral_planet_total + my_fleet_score + enemy_fleet_score
    return total_score

@jax.jit
def jax_evaluate_batch(planets_batch, fleets_batch, player_id, step_batch, weights):
    return jax.vmap(jax_evaluate_single, in_axes=(0, 0, None, 0, None))(
        planets_batch, fleets_batch, player_id, step_batch, weights
    )

@jax.jit
def jax_get_candidate_actions(planets, player_id, step, pos_table):
    p_owners = planets[:, 1]
    p_ships = planets[:, 5]
    p_x = planets[:, 2]
    p_y = planets[:, 3]
    
    # 1. Identify source planets owned by player_id
    is_mine = (p_owners == player_id)
    # Sort source planets by ships descending
    sorted_my_indices = jnp.argsort(jnp.where(is_mine, -p_ships, 1e9))
    src1_idx = sorted_my_indices[0]
    src2_idx = sorted_my_indices[1]
    
    # 2. Identify target planets (not mine, not comet, not padding)
    is_target = (p_owners != player_id) & (planets[:, 9] < 0.5) & (p_owners != -1e9)
    
    # For src1:
    dist1 = jnp.sqrt((p_x - p_x[src1_idx])**2 + (p_y - p_y[src1_idx])**2)
    sorted_tgt1_indices = jnp.argsort(jnp.where(is_target, dist1, 1e9))
    
    # For src2:
    dist2 = jnp.sqrt((p_x - p_x[src2_idx])**2 + (p_y - p_y[src2_idx])**2)
    sorted_tgt2_indices = jnp.argsort(jnp.where(is_target, dist2, 1e9))
    
    # Actions represented as: [src_id, angle, ships]
    actions = jnp.zeros((8, 3)) - 1.0
    
    # Helper to calculate and format actions
    def make_action(src_idx, tgt_idx):
        src_id = planets[src_idx, 0]
        src_ships = planets[src_idx, 5]
        src_prod = planets[src_idx, 6]
        src_safety = src_prod * 4.0
        src_surplus = jnp.maximum(0.0, src_ships - src_safety)
        
        tgt_owner = planets[tgt_idx, 1]
        tgt_ships = planets[tgt_idx, 5]
        tgt_id = planets[tgt_idx, 0].astype(jnp.int32)
        tgt_prod = planets[tgt_idx, 6]
        
        # Fixed point iteration (3 steps)
        # Iteration 1
        spd1 = get_fleet_speed(src_surplus)
        dist_init = jnp.sqrt((p_x[tgt_idx] - p_x[src_idx])**2 + (p_y[tgt_idx] - p_y[src_idx])**2)
        eta1 = jnp.ceil(jnp.maximum(1.0, (dist_init - planets[src_idx, 4]) / spd1))
        
        # Iteration 2
        future_step2 = jnp.clip(step + eta1, 0, 500).astype(jnp.int32)
        tx2 = pos_table[tgt_id, future_step2, 0]
        ty2 = pos_table[tgt_id, future_step2, 1]
        dist2 = jnp.sqrt((tx2 - p_x[src_idx])**2 + (ty2 - p_y[src_idx])**2)
        
        needed2 = jnp.where(tgt_owner == -1.0, tgt_ships + 1.0, tgt_ships + tgt_prod * eta1 + 1.0)
        send2 = jnp.clip(needed2, 1.0, src_surplus)
        spd2 = get_fleet_speed(send2)
        eta2 = jnp.ceil(jnp.maximum(1.0, (dist2 - planets[src_idx, 4]) / spd2))
        
        # Iteration 3
        future_step3 = jnp.clip(step + eta2, 0, 500).astype(jnp.int32)
        tx3 = pos_table[tgt_id, future_step3, 0]
        ty3 = pos_table[tgt_id, future_step3, 1]
        dist3 = jnp.sqrt((tx3 - p_x[src_idx])**2 + (ty3 - p_y[src_idx])**2)
        
        needed3 = jnp.where(tgt_owner == -1.0, tgt_ships + 1.0, tgt_ships + tgt_prod * eta2 + 1.0)
        send3 = jnp.clip(needed3, 1.0, src_surplus)
        spd3 = get_fleet_speed(send3)
        eta3 = jnp.ceil(jnp.maximum(1.0, (dist3 - planets[src_idx, 4]) / spd3))
        
        # Final position & angle
        future_step_final = jnp.clip(step + eta3, 0, 500).astype(jnp.int32)
        tx_f = pos_table[tgt_id, future_step_final, 0]
        ty_f = pos_table[tgt_id, future_step_final, 1]
        
        is_kep = planets[tgt_idx, 9] < 0.5
        tx = jnp.where(is_kep, tx_f, p_x[tgt_idx])
        ty = jnp.where(is_kep, ty_f, p_y[tgt_idx])
        
        dx = tx - p_x[src_idx]
        dy = ty - p_y[src_idx]
        angle = jnp.arctan2(dy, dx)
        send_ships = jnp.where(tgt_owner == -1.0, jnp.minimum(src_surplus, tgt_ships + 1.0), send3)
        
        valid = is_mine[src_idx] & (src_surplus > 0.0) & is_target[tgt_idx]
        return jnp.where(valid, jnp.array([src_id, angle, send_ships]), jnp.array([-1.0, -1.0, -1.0]))
        
    actions = actions.at[1].set(make_action(src1_idx, sorted_tgt1_indices[0]))
    actions = actions.at[2].set(make_action(src1_idx, sorted_tgt1_indices[1]))
    actions = actions.at[3].set(make_action(src1_idx, sorted_tgt1_indices[2]))
    actions = actions.at[4].set(make_action(src1_idx, sorted_tgt1_indices[3]))
    
    actions = actions.at[5].set(make_action(src2_idx, sorted_tgt2_indices[0]))
    actions = actions.at[6].set(make_action(src2_idx, sorted_tgt2_indices[1]))
    actions = actions.at[7].set(make_action(src2_idx, sorted_tgt2_indices[2]))
    
    return actions

@jax.jit
def jax_apply_action(planets, fleets, metadata, action, player_id):
    src_id, angle, ships_to_send = action
    
    valid_launch = (src_id >= 0.0) & (ships_to_send > 0.0)
    
    p_ids = planets[:, 0]
    p_match = (p_ids == src_id) & valid_launch
    planets = planets.at[:, 5].set(jnp.where(p_match, planets[:, 5] - ships_to_send, planets[:, 5]))
    
    src_idx = jnp.argmax(p_match)
    src_x = planets[src_idx, 2]
    src_y = planets[src_idx, 3]
    src_radius = planets[src_idx, 4]
    
    f_active = fleets[:, 7]
    free_slot_idx = jnp.argmax(f_active == 0.0)
    
    next_fid = metadata[2]
    new_fleet = jnp.array([
        next_fid,
        player_id,
        src_x + jnp.cos(angle) * (src_radius + 0.1),
        src_y + jnp.sin(angle) * (src_radius + 0.1),
        angle,
        src_id,
        ships_to_send,
        1.0
    ])
    
    should_launch = valid_launch & (f_active[free_slot_idx] == 0.0)
    fleets = fleets.at[free_slot_idx].set(jnp.where(should_launch, new_fleet, fleets[free_slot_idx]))
    
    new_metadata = metadata.at[2].set(jnp.where(should_launch, next_fid + 1.0, next_fid))
    
    return planets, fleets, new_metadata

@jax.jit
def jax_mcts_search(planets, fleets, metadata, pos_table, weights, player_id, num_iterations=100):
    MAX_NODES = 256
    MAX_BRANCHING = 8
    
    # Initialize flat tree arrays
    parent = jnp.zeros((MAX_NODES,), dtype=jnp.int32) - 1
    children = jnp.zeros((MAX_NODES, MAX_BRANCHING), dtype=jnp.int32) - 1
    num_children = jnp.zeros((MAX_NODES,), dtype=jnp.int32)
    visits = jnp.zeros((MAX_NODES,), dtype=FLOAT_DTYPE)
    value = jnp.zeros((MAX_NODES,), dtype=FLOAT_DTYPE)
    
    p_states = jnp.zeros((MAX_NODES, 100, 10), dtype=FLOAT_DTYPE)
    f_states = jnp.zeros((MAX_NODES, 200, 8), dtype=FLOAT_DTYPE)
    m_states = jnp.zeros((MAX_NODES, 4), dtype=FLOAT_DTYPE)
    
    untried_acts = jnp.zeros((MAX_NODES, MAX_BRANCHING, 3), dtype=FLOAT_DTYPE) - 1.0
    num_untried = jnp.zeros((MAX_NODES,), dtype=jnp.int32)
    nodes_action = jnp.zeros((MAX_NODES, 3), dtype=FLOAT_DTYPE) - 1.0
    
    # Initialize root (index 0)
    p_states = p_states.at[0].set(planets)
    f_states = f_states.at[0].set(fleets)
    m_states = m_states.at[0].set(metadata)
    
    root_acts = jax_get_candidate_actions(planets, player_id, metadata[1], pos_table)
    untried_acts = untried_acts.at[0].set(root_acts)
    num_untried = num_untried.at[0].set(MAX_BRANCHING)
    
    next_idx = 1
    
    # Loop state for MCTS iterations
    carry = (
        parent, children, num_children, visits, value,
        p_states, f_states, m_states,
        untried_acts, num_untried, nodes_action, next_idx
    )
    
    # Use jax.lax.fori_loop to run the iterations
    def loop_body(i, c):
        (
            parent, children, num_children, visits, value,
            p_states, f_states, m_states,
            untried_acts, num_untried, nodes_action, next_idx
        ) = c
        
        # 1. Selection
        curr_node = 0
        
        # Selection depth capped at 16
        def sel_step(d, node):
            step = m_states[node, 1]
            is_term = step >= 498.0
            has_ch = num_children[node] > 0
            has_untried = num_untried[node] > 0
            should_select = has_ch & (~has_untried) & (~is_term)
            
            ch_indices = children[node]
            ch_visits = visits[ch_indices]
            ch_values = value[ch_indices]
            parent_visit = visits[node]
            
            expl = 2.0 * jnp.sqrt(jnp.log(parent_visit + 1.0) / (ch_visits + 1e-6))
            ch_scores = (ch_values / (ch_visits + 1e-6)) + expl
            ch_scores = jnp.where(ch_indices >= 0, ch_scores, -1e9)
            
            best_child_slot = jnp.argmax(ch_scores)
            best_child_idx = ch_indices[best_child_slot]
            
            return jnp.where(should_select, best_child_idx, node)
            
        curr_node = jax.lax.fori_loop(0, 16, sel_step, curr_node)
        
        # 2. Expansion
        step = m_states[curr_node, 1]
        is_term = step >= 498.0
        has_untried = num_untried[curr_node] > 0
        should_expand = (~is_term) & has_untried & (next_idx < MAX_NODES)
        
        act_idx = jnp.maximum(0, num_untried[curr_node] - 1)
        action = untried_acts[curr_node, act_idx]
        
        p_curr = p_states[curr_node]
        f_curr = f_states[curr_node]
        m_curr = m_states[curr_node]
        
        p_next, f_next, m_next = jax_apply_action(p_curr, f_curr, m_curr, action, player_id)
        p_next, f_next, m_next = jax_simulate_step(p_next, f_next, m_next, pos_table)
        
        # Store new node if expanded
        p_states = p_states.at[next_idx].set(jnp.where(should_expand, p_next, p_states[next_idx]))
        f_states = f_states.at[next_idx].set(jnp.where(should_expand, f_next, f_states[next_idx]))
        m_states = m_states.at[next_idx].set(jnp.where(should_expand, m_next, m_states[next_idx]))
        parent = parent.at[next_idx].set(jnp.where(should_expand, curr_node, parent[next_idx]))
        nodes_action = nodes_action.at[next_idx].set(jnp.where(should_expand, action, nodes_action[next_idx]))
        
        new_acts = jax_get_candidate_actions(p_next, player_id, m_next[1], pos_table)
        untried_acts = untried_acts.at[next_idx].set(jnp.where(should_expand, new_acts, untried_acts[next_idx]))
        num_untried = num_untried.at[next_idx].set(jnp.where(should_expand, MAX_BRANCHING, num_untried[next_idx]))
        
        # Add new node to children of curr_node
        ch_slot = num_children[curr_node]
        children = children.at[curr_node, ch_slot].set(jnp.where(should_expand, next_idx, children[curr_node, ch_slot]))
        num_children = num_children.at[curr_node].add(jnp.where(should_expand, 1, 0))
        
        # Decrement untried actions for curr_node
        num_untried = num_untried.at[curr_node].set(jnp.where(has_untried, act_idx, num_untried[curr_node]))
        
        rollout_node = jnp.where(should_expand, next_idx, curr_node)
        next_idx = jnp.where(should_expand, next_idx + 1, next_idx)
        
        # 3. Rollout
        def rollout_step(step_i, carry_p_f_m):
            p, f, m = carry_p_f_m
            return jax_simulate_step(p, f, m, pos_table)
            
        p_roll, f_roll, m_roll = jax.lax.fori_loop(0, 16, rollout_step, (p_states[rollout_node], f_states[rollout_node], m_states[rollout_node]))
        
        # 4. Evaluation
        reward = jax_evaluate_single(p_roll, f_roll, player_id, m_roll[1], weights) / 5000.0
        
        # 5. Backpropagation
        bp_carry = (rollout_node, visits, value)
        def bp_step(step_i, carry_bp):
            node, v_arr, val_arr = carry_bp
            should_bp = node >= 0
            new_v_arr = v_arr.at[node].add(jnp.where(should_bp, 1.0, 0.0))
            new_val_arr = val_arr.at[node].add(jnp.where(should_bp, reward, 0.0))
            next_node = jnp.where(should_bp, parent[node], -1)
            return (next_node, new_v_arr, new_val_arr)
            
        _, visits, value = jax.lax.fori_loop(0, 16, bp_step, bp_carry)
        
        return (
            parent, children, num_children, visits, value,
            p_states, f_states, m_states,
            untried_acts, num_untried, nodes_action, next_idx
        )
        
    # Run iterations
    res = jax.lax.fori_loop(0, num_iterations, loop_body, carry)
    
    (
        parent, children, num_children, visits, value,
        p_states, f_states, m_states,
        untried_acts, num_untried, nodes_action, next_idx
    ) = res
    
    ch_indices = children[0]
    ch_visits = visits[ch_indices]
    ch_visits = jnp.where(ch_indices >= 0, ch_visits, -1.0)
    
    best_child_slot = jnp.argmax(ch_visits)
    best_child_idx = ch_indices[best_child_slot]
    
    best_action = nodes_action[best_child_idx]
    has_expanded_children = num_children[0] > 0
    final_action = jnp.where(has_expanded_children & (best_child_idx >= 0), best_action, jnp.array([-1.0, -1.0, -1.0]))
    
    return final_action

def warm_up():
    """Enable X64, configure persistent XLA compilation cache, and compile exact shapes.

    Auto-detects Metal (Apple GPU) vs CPU backend. Metal requires float32;
    CPU/Numba uses float64 for accuracy.
    """
    import os
    from jax import config

    # Detect backend
    _devices = jax.devices()
    _on_metal = any(d.platform.upper() == "METAL" for d in _devices)

    # 1. Identify cache directory
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except Exception:
        script_dir = os.path.abspath(".")
        
    cache_path = None
    
    # Check A: Flat jit_ files in script directory
    if os.path.exists(script_dir):
        try:
            if any(f.startswith("jit_") for f in os.listdir(script_dir)):
                cache_path = script_dir
        except Exception:
            pass
            
    # Check B: Flat jit_ files in current working directory
    if not cache_path and os.path.exists("."):
        try:
            if any(f.startswith("jit_") for f in os.listdir(".")):
                cache_path = os.path.abspath(".")
        except Exception:
            pass
            
    # Check C: jax_cache subdirectory in script directory
    if not cache_path:
        script_cache_dir = os.path.join(script_dir, "jax_cache")
        if os.path.exists(script_cache_dir):
            cache_path = script_cache_dir
            
    # Check D: jax_cache subdirectory in current working directory
    if not cache_path:
        cwd_cache_dir = os.path.abspath("jax_cache")
        if os.path.exists(cwd_cache_dir):
            cache_path = cwd_cache_dir
            
    # Check E: Fallback to home dev path
    if not cache_path:
        cache_path = os.path.join(os.path.expanduser("~"), ".gemini", "antigravity", "jax_cache")
        os.makedirs(cache_path, exist_ok=True)
        
    config.update("jax_compilation_cache_dir", cache_path)
    config.update("jax_persistent_cache_min_compile_time_secs", 0)

    # Metal does not support float64; CPU does
    if _on_metal:
        config.update("jax_enable_x64", False)
        _dtype = jnp.float32
        print(f"INFO: JAX running on Metal (Apple GPU) — using float32.")
    else:
        config.update("jax_enable_x64", True)
        _dtype = jnp.float64
        print(f"INFO: JAX running on CPU — using float64.")
    
    # MUST match exactly what's passed in main.py MCTS
    # We use 200 max_fleets instead of 1000 to massively speed up JAX CPU vectorization
    p = jnp.zeros((32, 100, 10), dtype=_dtype)
    f = jnp.zeros((32, 200, 8), dtype=_dtype)
    m = jnp.zeros((32, 4), dtype=_dtype)
    pos_table = jnp.zeros((100, 501, 2), dtype=_dtype)
    weights = jnp.zeros((10,), dtype=_dtype)
    
    try:
        # Pre-compile simulate step for both 17 steps (candidate rollout) and 64 steps (deep rollouts)
        p_res17, f_res17, m_res17 = jax_simulate_batch(p, f, m, pos_table, 17)
        p_res17.block_until_ready()

        p_res64, f_res64, m_res64 = jax_simulate_batch(p, f, m, pos_table, 64)
        p_res64.block_until_ready()
        
        rewards_res = jax_evaluate_batch(p, f, 0, m[:, 1], weights)
        rewards_res.block_until_ready()
        
        # Pre-compile the MCTS search using a single element of batch
        p_single = p[0]
        f_single = f[0]
        m_single = m[0]
        act_res = jax_mcts_search(p_single, f_single, m_single, pos_table, weights, 0, num_iterations=10)
        act_res.block_until_ready()
        
        print(f"JAX Engine Warmup Complete. Cache resolved at {cache_path}")
    except Exception as e:
        print(f"JAX Warmup Failed: {e}")

if __name__ == "__main__":
    warm_up()


