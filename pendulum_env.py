import math
import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import VectorEnv


class PendulumRK4(VectorEnv):
    """
    Vectorised Pendulum environment that implements the Gymnasium
    ``VectorEnv`` interface while keeping all computation as batched
    PyTorch operations (GPU-friendly).

    Single-env spaces
    -----------------
    Observation : Box(3,)  – [cos(theta), sin(theta), theta_dot]
    Action      : Box(1,)  – scalar torque in [-max_torque, max_torque]
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
        frame_skip: int = 1,
        render_mode=None,
    ):
        self.max_speed = 8.0
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = 9.81
        self.m = 1.0
        self.l = 1.0

        self.device = device
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.frame_skip = frame_skip

        # Define single spaces
        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        single_obs_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        single_act_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )

        super().__init__(
            num_envs=num_envs,
            observation_space=single_obs_space,
            action_space=single_act_space,
        )

        # State holds [theta, theta_dot]
        self.state = torch.zeros((num_envs, 2), dtype=torch.float32, device=device)
        self._step_counts = torch.zeros(num_envs, dtype=torch.long, device=device)

        # Rendering state
        self._last_action: torch.Tensor = torch.zeros(num_envs, device=device)
        self._pygame_initialized: bool = False
        self._render_history: dict = {"th": [], "thdot": [], "act": []}
        self._render_history_maxlen: int = 256

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            torch.manual_seed(seed[0] if isinstance(seed, (list, tuple)) else seed)

        # Default Pendulum initialization: theta in [-pi, pi], thetadot in [-1, 1]
        high = torch.tensor([torch.pi, 1.0], device=self.device)
        
        # Generate uniform distribution between [-high, high]
        rand_vals = torch.rand((self.num_envs, 2), device=self.device) * 2.0 - 1.0
        self.state = rand_vals * high
        
        self._step_counts.zero_()
        self._last_action.zero_()

        # Clear render history on global reset
        for key in self._render_history:
            self._render_history[key].clear()

        return self._get_obs(), {}

    def _dynamics(self, state, u):
        """Batched pendulum equations of motion."""
        th = state[:, 0]
        thdot = state[:, 1]
        
        dthdot = (
            3 * self.g / (2 * self.l) * torch.sin(th) + 
            3.0 / (self.m * self.l ** 2) * u
        )
        return torch.stack([thdot, dthdot], dim=-1)

    def _rk4_step(self, state, u, dt):
        """Batched RK4 integration step."""
        k1 = self._dynamics(state, u)
        k2 = self._dynamics(state + 0.5 * dt * k1, u)
        k3 = self._dynamics(state + 0.5 * dt * k2, u)
        k4 = self._dynamics(state + dt * k3, u)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _angle_normalize(self, x):
        """Normalizes the angle to the range [-pi, pi]."""
        return ((x + torch.pi) % (2 * torch.pi)) - torch.pi

    def _get_obs(self):
        """Returns the batched observations [cos(th), sin(th), thdot]."""
        th = self.state[:, 0]
        thdot = self.state[:, 1]
        return torch.stack([torch.cos(th), torch.sin(th), thdot], dim=-1)

    def step(self, actions):
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        actions = actions.to(self.device).float()

        if actions.ndim == 2 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)

        # Clip actions and save for rendering
        u = torch.clamp(actions, -self.max_torque, self.max_torque)
        self._last_action = u.clone()


        rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        for _ in range(self.frame_skip):
            th = self.state[:, 0]
            thdot = self.state[:, 1]
            # Calculate costs (rewards)
            costs = self._angle_normalize(th) ** 2 + 0.1 * (thdot ** 2) + 0.001 * (u ** 2)
            rewards += -costs

            # Integrate dynamics
            new_state = self._rk4_step(self.state, u, self.dt)
            new_th = new_state[:, 0]
            new_thdot = torch.clamp(new_state[:, 1], -self.max_speed, self.max_speed)

            self.state = torch.stack([new_th, new_thdot], dim=-1)

        # Handle episode limits and truncation
        self._step_counts += 1
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = self._step_counts >= self.max_episode_steps

        # Save final observation before auto-reset (for GAE value bootstrapping)
        final_obs = self._get_obs()

        # Auto-reset truncated environments
        if truncated.any():
            idx = truncated.nonzero(as_tuple=True)[0]
            high = torch.tensor([torch.pi, 1.0], device=self.device)
            rand_vals = torch.rand((len(idx), 2), device=self.device) * 2.0 - 1.0
            self.state[idx] = rand_vals * high
            self._step_counts[idx] = 0

        obs = self._get_obs()
        infos = {"final_observation": final_obs}

        return obs, rewards, terminated, truncated, infos

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
            pygame.display.set_caption("BatchedPendulumEnv")
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
        C_BG      = (240, 240, 244)
        C_PANEL   = (255, 255, 255)
        C_AXLE    = ( 50,  50,  50)
        C_ROD     = (204,  77,  77)
        C_FORCE   = ( 40, 160, 200)
        C_TEXT    = ( 40,  40,  40)
        C_GRID    = (220, 220, 220)
        C_THETA   = ( 70, 130, 180)
        C_THDOT   = ( 60, 160,  75)
        C_ACT     = (200,  40,  40)

        screen.fill(C_BG)

        # ── current env[0] values ─────────────────────────────────────────
        th = float(self.state[0, 0])
        thdot = float(self.state[0, 1])
        act = float(self._last_action[0])

        # ── update rolling history ────────────────────────────────────────
        for key, val in (("th", th), ("thdot", thdot), ("act", act)):
            lst = self._render_history[key]
            lst.append(val)
            if len(lst) > self._render_history_maxlen:
                del lst[0]

        # ═════════════════════════════════════════════════════════════════
        # LEFT PANEL – physics visualisation
        # ═════════════════════════════════════════════════════════════════
        phys_w = W * 9 // 16  # ≈ 540 px
        pygame.draw.rect(screen, C_PANEL, (0, 0, phys_w, H))

        offset_x = phys_w // 2
        offset_y = H // 2
        scale = 150  # rendering length of the rod

        # -- Pendulum Rod --------------------------------------------------
        # In Gym's pendulum, th=0 is straight UP. 
        end_x = offset_x + scale * math.sin(th)
        end_y = offset_y - scale * math.cos(th)

        pygame.draw.line(screen, C_ROD, (offset_x, offset_y), (end_x, end_y), 15)
        pygame.draw.circle(screen, C_ROD, (int(end_x), int(end_y)), 8)
        
        # -- Axle ----------------------------------------------------------
        pygame.draw.circle(screen, C_AXLE, (offset_x, offset_y), 10)

        # -- force arrow (Torque) ------------------------------------------
        if abs(act) > 0.05 * self.max_torque:
            arrow_px = int(np.clip(act / self.max_torque * 100, -100, 100))
            ay = H - 60
            pygame.draw.line(screen, C_FORCE,
                             (offset_x, ay), (offset_x + arrow_px, ay), 5)
            sign = 1 if arrow_px > 0 else -1
            pygame.draw.polygon(screen, C_FORCE, [
                (offset_x + arrow_px,              ay),
                (offset_x + arrow_px - sign * 9,   ay - 7),
                (offset_x + arrow_px - sign * 9,   ay + 7),
            ])
            flbl = self._font_sm.render("Torque", True, C_FORCE)
            screen.blit(flbl, (offset_x - flbl.get_width() // 2, ay - 22))

        # -- text overlays -------------------------------------------------
        title = self._font_md.render(
            "BatchedPendulumEnv  ·  env[0]", True, C_TEXT)
        screen.blit(title, (10, 8))
        status = self._font_sm.render(
            f"th={th:+.3f} rad   thdot={thdot:+.3f}   act={act:+.2f}   nenvs={self.num_envs}",
            True, C_TEXT,
        )
        screen.blit(status, (10, H - 22))

        # ═════════════════════════════════════════════════════════════════
        # RIGHT PANEL – rolling time-series plots
        # ═════════════════════════════════════════════════════════════════
        RX0 = phys_w + 8
        RW  = W - RX0 - 8
        T_TOP, T_BOT, T_GAP = 28, 14, 6
        sub_h = (H - T_TOP - T_BOT - T_GAP * 2) // 3

        series = [
            ("Angle (rad)", C_THETA, "th"),
            ("Angular Velocity", C_THDOT, "thdot"),
            ("Torque Action",   C_ACT, "act"),
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

                def vy(val: float, _sy0: int = sy0, _sh: int = sub_h,
                       _lo: float = d_lo, _sp: float = span) -> int:
                    return _sy0 + _sh - 2 - int((val - _lo) / _sp * (_sh - 4))

                def vx(i: int, _n: int = len(data)) -> int:
                    return RX0 + 1 + int(i / max(_n - 1, 1) * (RW - 2))

                if d_lo < 0 < d_hi:
                    zy = vy(0.0)
                    pygame.draw.line(screen, C_GRID,
                                     (RX0 + 1, zy), (RX0 + RW - 1, zy), 1)

                pts = [(vx(j), vy(data[j])) for j in range(len(data))]
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

    def close(self):
        if self._pygame_initialized:
            import pygame
            pygame.quit()
            self._pygame_initialized = False