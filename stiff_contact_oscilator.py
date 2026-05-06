import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import VectorEnv


class StiffContactOscillator(VectorEnv):
    """
    Vectorised stiff-contact oscillator that implements the Gymnasium
    ``VectorEnv`` interface while keeping all computation as batched
    PyTorch operations (GPU-friendly).

    Single-env spaces
    -----------------
    Observation : Box(2,)  – [position, velocity]
    Action      : Box(1,)  – scalar force in [-max_action, max_action]

    Batched spaces (set automatically by VectorEnv.__init__)
    ---------------------------------------------------------
    self.observation_space  → Box(num_envs, 2)
    self.action_space       → Box(num_envs, 1)
    self.single_observation_space / self.single_action_space  → single-env spaces
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        num_envs: int = 1024,
        dt: float = 0.002,       # high-frequency physics (500 Hz)
        frame_skip: int = 1,     # control frequency knob
        device: str = "cuda",
        k: float = 5000.0,
        c: float = 5.0,
        lam: float = 2.0,
        max_action: float = 10.0,
        max_episode_steps: int = 1280,
        render_mode=None,
    ):
        # Define per-environment (single) spaces first, then pass to
        # VectorEnv which creates the batched spaces and sets:
        #   self.single_observation_space / single_action_space
        #   self.observation_space / action_space  (batch_space of the above)
        #   self.is_vector_env = True
        single_obs_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )
        single_act_space = spaces.Box(
            low=-max_action, high=max_action, shape=(1,), dtype=np.float32
        )
        super().__init__(
            num_envs=num_envs,
            observation_space=single_obs_space,
            action_space=single_act_space,
        )

        self.dt = dt
        self.frame_skip = frame_skip
        self.device = device
        self.max_action = max_action
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        self.k = k
        self.c = c
        self.lam = lam

        self.state = torch.zeros(num_envs, 2, device=device)
        self._step_counts = torch.zeros(num_envs, dtype=torch.long, device=device)

        # Rendering state
        self._last_action: torch.Tensor = torch.zeros(num_envs, device=device)
        self._pygame_initialized: bool = False
        self._render_history: dict = {"pos": [], "vel": [], "act": []}
        self._render_history_maxlen: int = 256

    # ------------------------------------------------------------------
    # VectorEnv interface
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        """Reset all environments and return initial observations.

        Parameters
        ----------
        seed : int | list[int] | None
            Global seed (int) or per-env seed list. Passed to the parent
            RNG; also forwarded to torch for reproducibility.
        """

        if seed is not None:
            # Use a single global seed (take first element if list given)
            torch.manual_seed(seed[0] if isinstance(seed, (list, tuple)) else seed)

        p = 0.1 * torch.randn(self.num_envs, device=self.device).abs()
        v = 0.1 * torch.randn(self.num_envs, device=self.device)
        self.state = torch.stack([p, v], dim=-1)
        self._step_counts.zero_()

        # infos dict follows the VectorEnv convention: keys map to
        # per-env arrays of shape (num_envs, ...)
        return self.state, {}

    def step(self, actions):
        """Step all environments with a batch of actions.

        Parameters
        ----------
        actions : torch.Tensor | np.ndarray – shape (num_envs,) or (num_envs, 1)

        Returns
        -------
        obs         : torch.Tensor  (num_envs, 2)
        rewards     : torch.Tensor  (num_envs,)
        terminated  : torch.Tensor  (num_envs,)  – always False (infinite horizon)
        truncated   : torch.Tensor  (num_envs,)  – always False
        infos       : dict
        """
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        actions = actions.to(self.device).float()

        # Accept both (num_envs,) and (num_envs, 1)
        if actions.ndim == 2 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)

        actions = torch.clamp(actions, -self.max_action, self.max_action)
        self._last_action = actions.clone()

        total_reward = torch.zeros(self.num_envs, device=self.device)

        for _ in range(self.frame_skip):
            self._physics_step(actions)

            p, v = self.state[:, 0], self.state[:, 1]
            penetration_step = (p < 0.0).float()

            reward = (
                -(p**2)
                #- 0.1 * (v**2)
                - 0.001 * (actions**2)
                - 10.0 * penetration_step
            )
            total_reward += reward

        # Average reward across internal steps
        total_reward = total_reward# / self.frame_skip

        self._step_counts += 1
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = self._step_counts >= self.max_episode_steps

        # Save the final observation *before* resetting so the caller can
        # bootstrap the value from it (required for correct GAE on truncation).
        final_obs = self.state.clone()

        # Auto-reset environments that were truncated
        if truncated.any():
            idx = truncated.nonzero(as_tuple=True)[0]
            p_new = 0.1 * torch.randn(len(idx), device=self.device).abs()
            v_new = 0.1 * torch.randn(len(idx), device=self.device)
            self.state[idx] = torch.stack([p_new, v_new], dim=-1)
            self._step_counts[idx] = 0

        # `final_observation` follows the VectorEnv auto-reset convention:
        # shape (num_envs, obs_dim), meaningful only where truncated=True.
        infos = {"final_observation": final_obs}
        return self.state, total_reward, terminated, truncated, infos

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _physics_step(self, action: torch.Tensor) -> None:
        p, v = self.state[:, 0], self.state[:, 1]

        # Contact force (wall at p = 0)
        contact = (p < 0.0).float()
        f_wall = contact * (-self.k * p - self.c * v)

        dp = v
        dv = action + f_wall + self.lam * p

        p = p + self.dt * dp
        v = v + self.dt * dv

        self.state = torch.stack([p, v], dim=-1)

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
            pygame.display.set_caption("StiffContactOscillator")
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
        C_CONTACT = (210, 210, 215)
        C_HATCH   = (175, 175, 182)
        C_WALL    = ( 50,  50,  50)
        C_TRACK   = (180, 180, 180)
        C_SPRING  = ( 80,  80,  80)
        C_FREE    = ( 70, 130, 180)  # steelblue
        C_COLL    = (220,  80,  60)  # tomato
        C_FORCE   = (200,  40,  40)  # crimson
        C_TEXT    = ( 40,  40,  40)
        C_GRID    = (220, 220, 220)
        C_POS     = ( 70, 130, 180)
        C_VEL     = ( 60, 160,  75)
        C_ACT     = (200,  40,  40)

        screen.fill(C_BG)

        # ── current env[0] values ─────────────────────────────────────────
        p = float(self.state[0, 0])
        v = float(self.state[0, 1])
        a = float(self._last_action[0])

        # ── update rolling history ────────────────────────────────────────
        for key, val in (("pos", p), ("vel", v), ("act", a)):
            lst = self._render_history[key]
            lst.append(val)
            if len(lst) > self._render_history_maxlen:
                del lst[0]

        # ═════════════════════════════════════════════════════════════════
        # LEFT PANEL – physics visualisation
        # ═════════════════════════════════════════════════════════════════
        phys_w = W * 9 // 16  # ≈ 540 px
        pygame.draw.rect(screen, C_PANEL, (0, 0, phys_w, H))

        MX = 50               # horizontal margin inside panel
        MY = 55               # vertical margin
        draw_w = phys_w - 2 * MX
        draw_h = H - 2 * MY
        y_mid  = MY + draw_h // 2

        X_MIN, X_MAX = -0.8, 1.6  # world-space visible range

        def wx(world_x: float) -> int:
            return MX + int((world_x - X_MIN) / (X_MAX - X_MIN) * draw_w)

        wall_sx = wx(0.0)

        # -- hatched contact region (x < 0) --------------------------------
        cr_w = max(0, wall_sx - MX)
        pygame.draw.rect(screen, C_CONTACT, (MX, MY, cr_w, draw_h))
        old_clip = screen.get_clip()
        screen.set_clip(pygame.Rect(MX, MY, cr_w, draw_h))
        for x0 in range(MX - draw_h, wall_sx, 14):
            pygame.draw.line(screen, C_HATCH,
                             (x0, MY), (x0 + draw_h, MY + draw_h), 1)
        screen.set_clip(old_clip)

        # -- track line ----------------------------------------------------
        pygame.draw.line(screen, C_TRACK, (MX, y_mid), (phys_w - MX, y_mid), 2)

        # -- wall ----------------------------------------------------------
        pygame.draw.line(screen, C_WALL,
                         (wall_sx, MY + 6), (wall_sx, MY + draw_h - 6), 3)
        wlbl = self._font_sm.render("wall", True, C_WALL)
        screen.blit(wlbl, (wall_sx - wlbl.get_width() // 2, MY - 18))

        # -- spring (zigzag between wall and mass) -------------------------
        mass_r  = 16
        mass_sx = wx(p)
        spr_end = mass_sx - mass_r - 2

        if spr_end > wall_sx + 8:
            N = 64
            n_coils = 8
            xs = np.linspace(wall_sx, spr_end, N)
            ys = y_mid + 11 * np.sin(np.linspace(0, n_coils * 2 * np.pi, N))
            spts = ([(wall_sx, y_mid)]
                    + [(int(xs[i]), int(ys[i])) for i in range(N)]
                    + [(spr_end, y_mid)])
            pygame.draw.lines(screen, C_SPRING, False, spts, 2)

        # -- mass ----------------------------------------------------------
        mc = C_COLL if p < 0 else C_FREE
        pygame.draw.circle(screen, mc, (mass_sx, y_mid), mass_r)
        pygame.draw.circle(screen, (0, 0, 0), (mass_sx, y_mid), mass_r, 2)

        # -- force arrow ---------------------------------------------------
        if abs(a) > 0.05 * self.max_action:
            arrow_px = int(np.clip(a / self.max_action * 55, -55, 55))
            ay = y_mid - mass_r - 16
            pygame.draw.line(screen, C_FORCE,
                             (mass_sx, ay), (mass_sx + arrow_px, ay), 3)
            sign = 1 if arrow_px > 0 else -1
            pygame.draw.polygon(screen, C_FORCE, [
                (mass_sx + arrow_px,              ay),
                (mass_sx + arrow_px - sign * 9,   ay - 5),
                (mass_sx + arrow_px - sign * 9,   ay + 5),
            ])

        # -- text overlays -------------------------------------------------
        title = self._font_md.render(
            "StiffContactOscillator  ·  env[0]", True, C_TEXT)
        screen.blit(title, (10, 8))
        status = self._font_sm.render(
            f"pos={p:+.3f}   vel={v:+.3f}   act={a:+.2f}   nenvs={self.num_envs}",
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
            ("Position", C_POS, "pos"),
            ("Velocity", C_VEL, "vel"),
            ("Action",   C_ACT, "act"),
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
            self._clock.tick(60)
            return None
        else:  # "rgb_array"
            return np.transpose(pygame.surfarray.array3d(screen), (1, 0, 2))

    def close(self):
        if self._pygame_initialized:
            import pygame
            pygame.quit()
            self._pygame_initialized = False