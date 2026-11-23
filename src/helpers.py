import math

# ---------------------------------------------------------------------------
# Constants (match official Orbit Wars env)
# ---------------------------------------------------------------------------
BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SPEED = 6.0
_LOG1000 = math.log(1000.0)
_SPEED_RANGE = MAX_SPEED - 1.0

# ---------------------------------------------------------------------------
# Geometry & Physics
# ---------------------------------------------------------------------------

def dist(p1, p2):
    """Euclidean distance between two objects with x, y attributes."""
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def fleet_speed(ships):
    """Official Orbit Wars fleet speed formula."""
    if ships <= 1:
        return 1.0
    if ships >= 1000:
        return MAX_SPEED
    ratio = math.log(ships) / _LOG1000
    return min(1.0 + _SPEED_RANGE * (ratio ** 1.5), MAX_SPEED)


def ships_for_speed(desired_speed):
    """Inverse of fleet_speed: minimum ships needed to reach a given speed.
    
    Mathematically verified roundtrip: s -> v(s) -> ships_for_speed(v) -> s
    is exact within 1 ship for all valid s in [1, 1000].
    """
    if desired_speed <= 1.0:
        return 1
    if desired_speed >= MAX_SPEED:
        return 1000
    ratio = ((desired_speed - 1.0) / _SPEED_RANGE) ** (2.0 / 3.0)
    return max(1, min(1000, int(math.ceil(1000.0 ** ratio))))


def ships_for_eta(distance, desired_eta):
    """Minimum ships to cover distance in desired_eta turns.
    
    ships = ships_for_speed(distance / desired_eta)
    """
    if desired_eta <= 0:
        return 1000
    return ships_for_speed(distance / desired_eta)

def fleet_eta(src, tgt, ships):
    """Estimated turns to arrival (dist / speed, min 1)."""
    d = math.hypot(src.x - tgt.x, src.y - tgt.y)
    s = fleet_speed(ships)
    return max(1.0, d / s)

def path_crosses_sun(x1, y1, x2, y2, margin=0.0):
    """True if the line segment from (x1,y1) to (x2,y2) intersects the sun."""
    r = SUN_RADIUS + margin
    dx, dy = x2 - x1, y2 - y1
    l2 = dx * dx + dy * dy
    if l2 < 1e-12:
        return math.hypot(x1 - CENTER, y1 - CENTER) < r
    t = max(0.0, min(1.0, ((CENTER - x1) * dx + (CENTER - y1) * dy) / l2))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(proj_x - CENTER, proj_y - CENTER) < r

def is_orbiting(planet):
    """True if planet is within the rotation radius limit."""
    px = planet.x if hasattr(planet, "x") else planet['x']
    py = planet.y if hasattr(planet, "y") else planet['y']
    return math.hypot(px - CENTER, py - CENTER) < ROTATION_RADIUS_LIMIT

def predict_planet_pos(planet, angular_vel, steps):
    """Predict planet position after N steps of rotation."""
    if not is_orbiting(planet) or steps <= 0:
        px = planet.x if hasattr(planet, "x") else planet['x']
        py = planet.y if hasattr(planet, "y") else planet['y']
        return px, py
    
    px = planet.x if hasattr(planet, "x") else planet['x']
    py = planet.y if hasattr(planet, "y") else planet['y']
    dx, dy = px - CENTER, py - CENTER
    r = math.hypot(dx, dy)
    cur_theta = math.atan2(dy, dx)
    new_theta = cur_theta + angular_vel * steps
    return CENTER + r * math.cos(new_theta), CENTER + r * math.sin(new_theta)

def solve_intercept(fx, fy, tx, ty, orbiting, omega, ships, iterations=25):
    """Iteratively solve for intercept point with an orbiting target.
    
    Returns (intercept_x, intercept_y, eta_turns).
    Uses fixed-point convergence on the ORIGINAL target position:
      1. Compute ETA from src to previous predicted position
      2. Predict where target will be (from ORIGINAL starting position) at that ETA
    This avoids compounding angular displacement across iterations.
    """
    d = math.hypot(tx - fx, ty - fy)
    t = d / fleet_speed(max(1, ships)) if ships > 0 else 999.0
    
    if not orbiting:
        return tx, ty, t
    
    # Pre-compute orbital parameters from original target position
    orig_dx = tx - CENTER
    orig_dy = ty - CENTER
    r = math.hypot(orig_dx, orig_dy)
    orig_theta = math.atan2(orig_dy, orig_dx)
    
    ix, iy = tx, ty  # predicted position (updated each iteration)
    for _ in range(iterations):
        # Predict from ORIGINAL target position at time t (never compounds)
        new_theta = orig_theta + omega * t
        ix = CENTER + r * math.cos(new_theta)
        iy = CENTER + r * math.sin(new_theta)
        
        d2 = math.hypot(ix - fx, iy - fy)
        t2 = d2 / fleet_speed(max(1, ships))
        if abs(t2 - t) < 0.05:
            break
        t = t2
    
    return ix, iy, max(t, 0.1)


def safe_angle(x1, y1, x2, y2, margin=1.5):
    """Compute launch angle that avoids the sun, going around if needed.
    
    If direct path doesn't cross sun, returns direct angle.
    Otherwise computes tangent angle to go around the sun (shortest arc).
    """
    direct = math.atan2(y2 - y1, x2 - x1)
    if not path_crosses_sun(x1, y1, x2, y2, margin):
        return direct
    d = math.hypot(x1 - CENTER, y1 - CENTER)
    if d <= SUN_RADIUS + 1.0:
        return direct
    half = math.asin(min(1.0, (SUN_RADIUS + 1.0) / d))
    to_sun = math.atan2(CENTER - y1, CENTER - x1)
    cw = to_sun + half
    ccw = to_sun - half
    def _adiff(a):
        dd = (a - direct) % (2 * math.pi)
        return min(dd, 2 * math.pi - dd)
    return cw if _adiff(cw) < _adiff(ccw) else ccw


def predict_fleet_target(fleet, planets, angular_vel, max_steps=200):
    """
    Predict which planet a fleet is targeting by simulating paths forward.
    Returns (planet_id, eta) or (None, None).
    """
    # handle object, dict, and tuple/list formats
    if isinstance(fleet, dict):
        fx, fy = fleet['x'], fleet['y']
        fa = fleet['angle']
        f_ships = fleet['ships']
    elif hasattr(fleet, "x"):
        fx, fy = fleet.x, fleet.y
        fa = fleet.angle
        f_ships = fleet.ships
    else:
        fx, fy = fleet[2], fleet[3]
        fa = fleet[4]
        f_ships = fleet[6]
    
    # Legacy prediction (forgiving angular cone)
    # This model is 'blind' to complex orbital intercepts but 
    # provides the strategic aggression needed for high-level play.
    dx_sun, dy_sun = fx - 50, fy - 50
    dist_to_sun = math.hypot(dx_sun, dy_sun)
    if dist_to_sun < SUN_RADIUS + 5: # Close to sun
        return -2, 5
        
    f_speed = fleet_speed(f_ships)
    for p in planets:
        if isinstance(p, dict):
            px, py, pid = p['x'], p['y'], p['id']
        elif hasattr(p, "x"):
            px, py, pid = p.x, p.y, p.id
        else:
            px, py, pid = p[2], p[3], p[0]
            
        dx, dy = px - fx, py - fy
        angle_to_p = math.atan2(dy, dx)
        diff = abs(angle_to_p - fa)
        while diff > math.pi: diff = abs(diff - 2 * math.pi)
        
        if diff < 0.15: # Cone filter
            dist = math.hypot(dx, dy)
            eta = dist / max(f_speed, 0.1)
            return pid, int(eta)
            
    return None, None

def fleets_targeting(planet, fleets, angular_vel=0.0, planets=None, turns_ahead=None):
    """Filter fleets that appear to be moving toward the given planet."""
    results = []
    p_id = planet.id if hasattr(planet, "id") else planet[0]
    
    for f in fleets:
        if angular_vel == 0 or planets is None:
            # Fallback to static angle check
            desired = math.atan2(planet.y - f.y, planet.x - f.x)
            diff = abs(math.remainder(f.angle - desired, 2 * math.pi))
            if diff < 0.1:
                if turns_ahead is not None:
                    d = math.hypot(f.x - planet.x, f.y - planet.y)
                    f_ships = f.ships if hasattr(f, "ships") else f[6]
                    eta = d / fleet_speed(f_ships)
                    if eta > turns_ahead:
                        continue
                results.append(f)
        else:
            # Use dynamic prediction
            target_id, eta = predict_fleet_target(f, planets, angular_vel)
            if target_id == p_id:
                if turns_ahead is not None and eta > turns_ahead:
                    continue
                results.append(f)
    return results

def is_fleet_threatening(fleet, planet, threshold_deg=15.0):
    """
    Trajectory analysis: Returns True if fleet is moving toward the planet
    within a specified angular cone.
    """
    if isinstance(fleet, dict):
        fx, fy, fa = fleet['x'], fleet['y'], fleet['angle']
    elif hasattr(fleet, "x"):
        fx, fy, fa = fleet.x, fleet.y, fleet.angle
    else:
        fx, fy, fa = fleet[2], fleet[3], fleet[4]
        
    if isinstance(planet, dict):
        px, py = planet['x'], planet['y']
    elif hasattr(planet, "x"):
        px, py = planet.x, planet.y
    else:
        px, py = planet[2], planet[3]

    dx, dy = px - fx, py - fy
    d = math.hypot(dx, dy)
    if d < 1e-6: return True
    
    # Dot product approach
    cos_fa, sin_fa = math.cos(fa), math.sin(fa)
    dot = (dx/d) * cos_fa + (dy/d) * sin_fa
    
    # threshold_deg = 45 -> cos(45) = 0.707
    threshold_cos = math.cos(math.radians(threshold_deg))
    return dot > threshold_cos

# ---------------------------------------------------------------------------
# Advanced Forecasts (for compatibility with existing tests)
# ---------------------------------------------------------------------------

def forecast_planet_ownership(planet, incoming_fleets, turns):
    """Estimate (owner, ships) after N turns considering incoming fleets."""
    # Simple model: resolve production, then combat
    ships = float(planet.ships)
    if planet.owner != -1:
        ships += planet.production * turns
    
    # Resolve fleets in arrival order
    arrivals = []
    for f in incoming_fleets:
        d = math.hypot(f.x - planet.x, f.y - planet.y)
        eta = d / fleet_speed(f.ships)
        if eta <= turns:
            arrivals.append((eta, f))
    
    arrivals.sort(key=lambda x: x[0])
    
    current_owner = planet.owner
    current_ships = ships
    
    for _, f in arrivals:
        if f.owner == current_owner:
            current_ships += f.ships
        else:
            current_ships -= f.ships
            if current_ships < 0:
                current_owner = f.owner
                current_ships = -current_ships
                
    return (current_owner, current_ships)

def project_net_garrison(planet, incoming_fleets, turns, perspective_owner):
    """Relative ship count for perspective_owner after N turns."""
    owner, ships = forecast_planet_ownership(planet, incoming_fleets, turns)
    return ships if owner == perspective_owner else -ships

def oob_clamp_target_pos(src_x, src_y, aim_x, aim_y, tgt_x, tgt_y, ships, planet_radius=0.0, board_size=100.0):
    """If the fleet would OOB on its terminal step near a board-edge planet,
    clamp aim to the target's exact center.

    The engine checks OOB before planet collision.  A high-speed fleet near the
    board edge can overshoot in its final tick and be destroyed before capture
    registers.  Clamping to the planet center eliminates the overshoot.

    Returns:
        (aim_x, aim_y) if safe, else (tgt_x, tgt_y).
    """
    speed = fleet_speed(max(1, ships))
    approach_to_edge = min(
        tgt_x, board_size - tgt_x,
        tgt_y, board_size - tgt_y
    )
    if approach_to_edge > speed + planet_radius + 1.0:
        return (aim_x, aim_y)
    dx, dy = aim_x - src_x, aim_y - src_y
    if math.hypot(dx, dy) < 1e-9:
        return (tgt_x, tgt_y)
    return (tgt_x, tgt_y)

