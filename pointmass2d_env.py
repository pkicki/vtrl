import math
import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import VectorEnv


class PointMass2D(VectorEnv):
    """
    Vectorised 2-D point-mass environment implementing the Gymnasium
    ``VectorEnv`` interface with batched PyTorch operations (GPU-friendly).

    The mass starts at a random position in the arena and must navigate to a
    fixed goal (default: origin).  All physics is batched over ``num_envs``
    independent copies.

    Single-env spaces
    -----------------
    Observation : Box(4,)  – [dx, dy, vx, vy]
                              where dx = x − goal_x,  dy = y − goal_y
    Action      : Box(2,)  – [ax, ay]  force/acceleration
                              in [−max_accel, max_accel]

    Reward (sparse)
    ---------------
    +1  for every sub-step the mass is within ``success_radius`` of the goal,
     0  otherwise.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        num_envs: int = 1024,
        device: str = "cuda",
        max_episode_steps: int = 200,
        sim_dt: float = 0.002,
        actor_dt: float = 0.01,
        render_mode=None,
        max_pos: float = 1.0,
        max_vel: float = 2.0,
        max_accel: float = 1.0,
        success_radius: float = 0.05,
        goal: tuple[float, float] = (0.0, 0.0),
    ):
        self.max_pos = max_pos
        self.max_vel = max_vel
        self.max_accel = max_accel
        self.success_radius = success_radius
        self.dt = sim_dt
        assert actor_dt >= sim_dt, "Actor time step must be >= simulation time step"
        assert abs(actor_dt / sim_dt - round(actor_dt / sim_dt)) < 1e-6, "Actor time step must be an integer multiple of simulation time step"
        self.frame_skip = max(1, int(actor_dt / sim_dt))

        self.device = device
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        # Observation: [dx, dy, vx, vy]
        obs_high = np.array(
            [2 * max_pos, 2 * max_pos, max_vel, max_vel], dtype=np.float32
        )
        single_obs_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)
        single_act_space = spaces.Box(
            low=-max_accel, high=max_accel, shape=(2,), dtype=np.float32
        )

        super().__init__(
            num_envs=num_envs,
            observation_space=single_obs_space,
            action_space=single_act_space,
        )

        # State: [x, y, vx, vy]; goal is shared across all envs
        self.state = torch.zeros((num_envs, 4), dtype=torch.float32, device=device)
        self.goal = torch.tensor(
            [goal[0], goal[1]], dtype=torch.float32, device=device
        ).expand(num_envs, -1).clone()

        self._step_counts = torch.zeros(num_envs, dtype=torch.long, device=device)

        # Rendering helpers
        self._last_action: torch.Tensor = torch.zeros(
            (num_envs, 2), dtype=torch.float32, device=device
        )
        self._pygame_initialized: bool = False
        self._render_history: dict = {"dist": [], "vx": [], "vy": []}
        self._render_history_maxlen: int = 256
        self._trail: list = []
        self._trail_maxlen: int = 150

    # ------------------------------------------------------------------
    # Core VectorEnv interface
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            torch.manual_seed(seed[0] if isinstance(seed, (list, tuple)) else seed)

        # Random position in (−max_pos, max_pos)²; start with zero velocity
        pos = (
            torch.rand((self.num_envs, 2), device=self.device) * 2.0 - 1.0
        ) * self.max_pos
        vel = torch.zeros((self.num_envs, 2), device=self.device)
        self.state = torch.cat([pos, vel], dim=-1)

        self._step_counts.zero_()
        self._last_action.zero_()

        for key in self._render_history:
            self._render_history[key].clear()
        self._trail.clear()

        return self._get_obs(), {}

    def _get_obs(self) -> torch.Tensor:
        """Returns [dx, dy, vx, vy] where (dx, dy) = pos − goal."""
        pos = self.state[:, :2]
        vel = self.state[:, 2:]
        rel = pos - self.goal
        return torch.cat([rel, vel], dim=-1)

    def step(self, actions):
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        actions = actions.to(self.device).float()
        u = torch.clamp(actions, -self.max_accel, self.max_accel)
        self._last_action = u.clone()

        rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for _ in range(self.frame_skip):
            pos = self.state[:, :2]
            vel = self.state[:, 2:]

            # Euler integration; clamp velocity to avoid blow-up
            new_vel = torch.clamp(vel + self.dt * u, -self.max_vel, self.max_vel)
            new_pos = pos + self.dt * new_vel
            self.state = torch.cat([new_pos, new_vel], dim=-1)

            # Sparse reward: +1 each sub-step while inside the goal region
            dist = torch.norm(new_pos - self.goal, dim=-1)
            rewards += (dist < self.success_radius).float() * self.dt
            terminated |= dist < self.success_radius
            terminated |= torch.any(torch.abs(new_pos) > self.max_pos * 1.5, dim=-1)


        self._step_counts += 1
        truncated = self._step_counts >= self.max_episode_steps

        # Save final obs before auto-reset (for GAE value bootstrapping)
        final_obs = self._get_obs()

        # Auto-reset truncated environments
        if truncated.any():
            idx = truncated.nonzero(as_tuple=True)[0]
            pos_r = (
                torch.rand((len(idx), 2), device=self.device) * 2.0 - 1.0
            ) * self.max_pos
            vel_r = torch.zeros((len(idx), 2), device=self.device)
            self.state[idx] = torch.cat([pos_r, vel_r], dim=-1)
            self._step_counts[idx] = 0

        obs = self._get_obs()
        # final_obs[:, :2] = [dx, dy] (relative to goal, before auto-reset)
        final_dist = torch.norm(final_obs[:, :2], dim=-1)
        infos = {
            "final_observation": final_obs,
            "env_metrics": {
                "dist_to_goal": final_dist.mean().item(),
            },
        }
        return obs, rewards, terminated, truncated, infos

    def close(self):
        if self._pygame_initialized:
            import pygame

            pygame.quit()
            self._pygame_initialized = False

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _init_render(self) -> None:
        """Lazily initialise pygame display and font objects."""
        if self._pygame_initialized:
            return
        import pygame

        pygame.init()
        pygame.font.init()
        self._screen_w, self._screen_h = 960, 480
        if self.render_mode == "human":
            self._screen = pygame.display.set_mode(
                (self._screen_w, self._screen_h),
                pygame.HWSURFACE | pygame.DOUBLEBUF,
            )
            pygame.display.set_caption("PointMass2D")
        else:  # "rgb_array"
            self._screen = pygame.Surface((self._screen_w, self._screen_h))
        self._clock = pygame.time.Clock()
        self._font_sm = pygame.font.SysFont("monospace", 13)
        self._font_md = pygame.font.SysFont("monospace", 15, bold=True)
        self._pygame_initialized = True

    def render(self):
        if self.render_mode is None:
            return None
        import pygame

        self._init_render()
        screen = self._screen
        W, H = self._screen_w, self._screen_h

        # ── colour palette ────────────────────────────────────────────────
        C_BG        = (240, 240, 244)
        C_PANEL     = (255, 255, 255)
        C_GRID      = (220, 220, 220)
        C_BOUNDS    = (180, 180, 190)
        C_GOAL      = ( 60, 180,  75)
        C_GOAL_RING = (150, 215, 150)
        C_MASS      = (204,  77,  77)
        C_VEL       = ( 40, 160, 200)
        C_FORCE     = (220, 150,  40)
        C_TRAIL     = (204,  77,  77)
        C_TEXT      = ( 40,  40,  40)
        C_DIST      = ( 60, 160,  75)
        C_CVX       = ( 70, 130, 180)
        C_CVY       = (200, 120,  40)

        screen.fill(C_BG)

        # ── current env[0] values ─────────────────────────────────────────
        px     = float(self.state[0, 0])
        py     = float(self.state[0, 1])
        vx     = float(self.state[0, 2])
        vy     = float(self.state[0, 3])
        fa     = float(self._last_action[0, 0])   # applied force x
        fb     = float(self._last_action[0, 1])   # applied force y
        goal_x = float(self.goal[0, 0])
        goal_y = float(self.goal[0, 1])
        dist   = math.hypot(px - goal_x, py - goal_y)
        at_goal = dist < self.success_radius

        # ── update rolling state ──────────────────────────────────────────
        self._trail.append((px, py))
        if len(self._trail) > self._trail_maxlen:
            del self._trail[0]

        for key, val in (("dist", dist), ("vx", vx), ("vy", vy)):
            lst = self._render_history[key]
            lst.append(val)
            if len(lst) > self._render_history_maxlen:
                del lst[0]

        # ═════════════════════════════════════════════════════════════════
        # LEFT PANEL – 2-D arena
        # ═════════════════════════════════════════════════════════════════
        phys_w = W * 9 // 16
        pygame.draw.rect(screen, C_PANEL, (0, 0, phys_w, H))

        arena_pad  = 50
        arena_size = min(phys_w - 2 * arena_pad, H - 2 * arena_pad)
        arena_x0   = (phys_w - arena_size) // 2
        arena_y0   = (H - arena_size) // 2

        def to_screen(wx: float, wy: float) -> tuple[int, int]:
            """World coords → screen pixels.  Y-axis is flipped (screen ↓)."""
            sx = arena_x0 + int((wx / self.max_pos + 1.0) * 0.5 * arena_size)
            sy = arena_y0 + int((1.0 - wy / self.max_pos) * 0.5 * arena_size)
            return sx, sy

        # -- arena border --------------------------------------------------
        pygame.draw.rect(screen, C_BOUNDS,
                         (arena_x0, arena_y0, arena_size, arena_size), 2)

        # -- grid lines (including centre axes) ----------------------------
        for frac in (0.25, 0.5, 0.75):
            gx_px = arena_x0 + int(frac * arena_size)
            gy_px = arena_y0 + int(frac * arena_size)
            lw = 2 if frac == 0.5 else 1
            pygame.draw.line(screen, C_GRID,
                             (gx_px, arena_y0), (gx_px, arena_y0 + arena_size), lw)
            pygame.draw.line(screen, C_GRID,
                             (arena_x0, gy_px), (arena_x0 + arena_size, gy_px), lw)

        # -- goal ----------------------------------------------------------
        gsx, gsy = to_screen(goal_x, goal_y)
        sr_px = max(int(self.success_radius / self.max_pos * arena_size * 0.5), 6)
        pygame.draw.circle(screen, C_GOAL_RING, (gsx, gsy), sr_px)
        pygame.draw.circle(screen, C_GOAL,      (gsx, gsy), sr_px, 2)
        pygame.draw.circle(screen, C_GOAL,      (gsx, gsy), 5)

        # "GOAL" label
        glbl = self._font_sm.render("goal", True, C_GOAL)
        screen.blit(glbl, (gsx + sr_px + 3, gsy - glbl.get_height() // 2))

        # -- trail ---------------------------------------------------------
        if len(self._trail) >= 2:
            trail_pts = [to_screen(tx, ty) for tx, ty in self._trail]
            for i in range(1, len(trail_pts)):
                alpha = i / len(trail_pts)
                c = tuple(
                    int(C_TRAIL[j] * alpha + C_PANEL[j] * (1.0 - alpha))
                    for j in range(3)
                )
                pygame.draw.line(screen, c, trail_pts[i - 1], trail_pts[i], 2)

        # -- mass ----------------------------------------------------------
        msx, msy = to_screen(px, py)
        mass_color = C_GOAL if at_goal else C_MASS
        pygame.draw.circle(screen, mass_color, (msx, msy), 9)
        pygame.draw.circle(screen, (30, 30, 30), (msx, msy), 9, 1)

        # -- velocity arrow ------------------------------------------------
        vel_scale = arena_size * 0.18 / max(self.max_vel, 1e-6)
        speed = math.hypot(vx, vy)
        if speed > 0.01:
            vex = msx + int(vx * vel_scale)
            vey = msy - int(vy * vel_scale)
            pygame.draw.line(screen, C_VEL, (msx, msy), (vex, vey), 3)
            _dx, _dy = vex - msx, vey - msy
            _len = math.hypot(_dx, _dy) or 1.0
            _ux, _uy = _dx / _len, _dy / _len
            _px, _py = -_uy, _ux
            pygame.draw.polygon(screen, C_VEL, [
                (vex, vey),
                (int(vex - _ux * 9 + _px * 5), int(vey - _uy * 9 + _py * 5)),
                (int(vex - _ux * 9 - _px * 5), int(vey - _uy * 9 - _py * 5)),
            ])

        # -- applied force arrow -------------------------------------------
        force_mag = math.hypot(fa, fb)
        force_scale = arena_size * 0.13 / max(self.max_accel, 1e-6)
        if force_mag > 0.01:
            fex = msx + int(fa * force_scale)
            fey = msy - int(fb * force_scale)
            pygame.draw.line(screen, C_FORCE, (msx, msy), (fex, fey), 2)
            _dx, _dy = fex - msx, fey - msy
            _len = math.hypot(_dx, _dy) or 1.0
            _ux, _uy = _dx / _len, _dy / _len
            _px, _py = -_uy, _ux
            pygame.draw.polygon(screen, C_FORCE, [
                (fex, fey),
                (int(fex - _ux * 7 + _px * 4), int(fey - _uy * 7 + _py * 4)),
                (int(fex - _ux * 7 - _px * 4), int(fey - _uy * 7 - _py * 4)),
            ])

        # -- text overlays -------------------------------------------------
        title = self._font_md.render("PointMass2D  ·  env[0]", True, C_TEXT)
        screen.blit(title, (10, 8))

        goal_str = "★ AT GOAL" if at_goal else f"dist={dist:.3f}"
        goal_col  = C_GOAL if at_goal else C_TEXT
        status = self._font_sm.render(
            f"pos=({px:+.3f}, {py:+.3f})  vel=({vx:+.3f}, {vy:+.3f})"
            f"  {goal_str}  nenvs={self.num_envs}",
            True, goal_col,
        )
        screen.blit(status, (10, H - 22))

        # -- legend --------------------------------------------------------
        lv = self._font_sm.render("→ velocity", True, C_VEL)
        screen.blit(lv, (10, H - 42))
        lf = self._font_sm.render("→ force", True, C_FORCE)
        screen.blit(lf, (10 + lv.get_width() + 14, H - 42))

        # ═════════════════════════════════════════════════════════════════
        # RIGHT PANEL – rolling time-series plots
        # ═════════════════════════════════════════════════════════════════
        RX0 = phys_w + 8
        RW  = W - RX0 - 8
        T_TOP, T_BOT, T_GAP = 28, 14, 6
        sub_h = (H - T_TOP - T_BOT - T_GAP * 2) // 3

        series = [
            ("Distance to Goal", C_DIST, "dist"),
            ("Velocity X",       C_CVX,  "vx"),
            ("Velocity Y",       C_CVY,  "vy"),
        ]
        for idx, (label, color, key) in enumerate(series):
            sy0 = T_TOP + idx * (sub_h + T_GAP)
            r = pygame.Rect(RX0, sy0, RW, sub_h)
            pygame.draw.rect(screen, C_PANEL, r)
            pygame.draw.rect(screen, C_GRID,  r, 1)

            lbl_surf = self._font_sm.render(label, True, C_TEXT)
            screen.blit(lbl_surf, (RX0 + 4, sy0 + 2))

            data = self._render_history[key]
            if len(data) >= 2:
                d_lo = min(data)
                d_hi = max(data)
                span = max(d_hi - d_lo, 1e-4)
                pad  = 0.08 * span
                d_lo -= pad
                d_hi += pad
                span  = d_hi - d_lo

                # Use default-arg trick to avoid late-binding in loop closure
                def plot_y(val: float, _sy0: int = sy0, _sh: int = sub_h,
                           _lo: float = d_lo, _sp: float = span) -> int:
                    return _sy0 + _sh - 2 - int((val - _lo) / _sp * (_sh - 4))

                def plot_x(i: int, _n: int = len(data)) -> int:
                    return RX0 + 1 + int(i / max(_n - 1, 1) * (RW - 2))

                # Zero line
                if d_lo < 0 < d_hi:
                    zy = plot_y(0.0)
                    pygame.draw.line(screen, C_GRID,
                                     (RX0 + 1, zy), (RX0 + RW - 1, zy), 1)

                # For "dist" draw success-radius threshold line
                if key == "dist" and d_lo < self.success_radius < d_hi:
                    ty = plot_y(self.success_radius)
                    pygame.draw.line(screen, C_GOAL,
                                     (RX0 + 1, ty), (RX0 + RW - 1, ty), 1)

                pts = [(plot_x(j), plot_y(data[j])) for j in range(len(data))]
                pygame.draw.lines(screen, color, False, pts, 2)

                cv = self._font_sm.render(f"{data[-1]:+.3f}", True, color)
                screen.blit(cv, (RX0 + RW - cv.get_width() - 4, sy0 + 2))

        # ── event pump & display ──────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return None

        if self.render_mode == "human":
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
            return None
        else:  # "rgb_array"
            return np.transpose(pygame.surfarray.array3d(screen), (1, 0, 2))
