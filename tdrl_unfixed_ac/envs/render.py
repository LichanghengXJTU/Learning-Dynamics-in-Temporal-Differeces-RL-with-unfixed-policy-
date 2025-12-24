"""Optional pygame renderer for TorusGobletGhostEnv."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import pygame
except Exception:  # pragma: no cover - optional dependency
    pygame = None


class HeadlessRenderer:
    """No-op renderer for headless usage."""

    def __init__(self, env: Any) -> None:
        self.env = env

    def render(self, action: Optional[np.ndarray] = None) -> None:
        return None

    def close(self) -> None:
        return None


class FrameRecorder:
    """Optional frame recorder to gif/mp4 via imageio."""

    def __init__(self, path: Optional[str], fps: int) -> None:
        self.path = Path(path) if path else None
        self.fps = int(fps)
        self._writer = None
        self._imageio = None
        if self.path is not None:
            try:
                import imageio.v2 as imageio  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("imageio is required for recording frames.") from exc
            self._imageio = imageio
            self._writer = imageio.get_writer(str(self.path), fps=self.fps)

    def add(self, frame: np.ndarray) -> None:
        if self._writer is None:
            return
        self._writer.append_data(frame)

    def close(self) -> None:
        if self._writer is None:
            return
        self._writer.close()
        self._writer = None


class TorusRenderer:
    """Pygame renderer for the torus Goblet&Ghost environment."""

    def __init__(
        self,
        env: Any,
        *,
        width: int = 720,
        height: int = 720,
        fps: int = 30,
        show: bool = True,
        record_path: Optional[str] = None,
    ) -> None:
        if pygame is None:
            raise RuntimeError("pygame is required for rendering.")
        if not show:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        pygame.font.init()

        self.env = env
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.show = bool(show)
        self.padding = 40
        self.torus_size = float(getattr(env, "torus_size", 1.0))
        self.scale = (min(self.width, self.height) - 2 * self.padding) / self.torus_size
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont(None, 18)
        self._closed = False

        if self.show:
            self._window = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Torus Goblet&Ghost")
            self._surface = self._window
        else:
            self._window = None
            self._surface = pygame.Surface((self.width, self.height))

        self._recording = record_path is not None
        self._recorder = FrameRecorder(record_path, fps=self.fps)

        self._colors = {
            "bg": (16, 18, 24),
            "grid": (60, 70, 90),
            "adventurer": (70, 200, 255),
            "ghost": (255, 120, 140),
            "goblet_pos": (250, 200, 70),
            "goblet_neg": (120, 60, 200),
            "text": (230, 230, 230),
            "arrow": (180, 255, 140),
        }

    def render(self, action: Optional[np.ndarray] = None) -> None:
        if self._closed:
            return
        self._handle_events()
        self._surface.fill(self._colors["bg"])
        self._draw_grid()
        self._draw_goblets()
        self._draw_entities()
        self._draw_action(action)
        self._draw_events()

        if self.show and self._window is not None:
            pygame.display.flip()
            self._clock.tick(self.fps)

        if self._recording:
            frame = pygame.surfarray.array3d(self._surface)
            frame = np.transpose(frame, (1, 0, 2))
            self._recorder.add(frame)

    def close(self) -> None:
        self._recorder.close()
        if pygame is not None:
            pygame.quit()

    def _handle_events(self) -> None:
        if not self.show:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._closed = True

    def _world_to_screen(self, pos: np.ndarray) -> Tuple[int, int]:
        x, y = pos
        sx = int(self.padding + x * self.scale)
        sy = int(self.padding + y * self.scale)
        return sx, sy

    def _draw_grid(self) -> None:
        rect = pygame.Rect(
            self.padding,
            self.padding,
            int(self.torus_size * self.scale),
            int(self.torus_size * self.scale),
        )
        pygame.draw.rect(self._surface, self._colors["grid"], rect, width=2)
        steps = 6
        for i in range(1, steps):
            offset = int(i * rect.width / steps)
            pygame.draw.line(
                self._surface,
                self._colors["grid"],
                (rect.left + offset, rect.top),
                (rect.left + offset, rect.bottom),
                width=1,
            )
            pygame.draw.line(
                self._surface,
                self._colors["grid"],
                (rect.left, rect.top + offset),
                (rect.right, rect.top + offset),
                width=1,
            )

    def _draw_goblets(self) -> None:
        goblets = getattr(self.env, "goblets_pos", None)
        goblet_types = getattr(self.env, "goblets_type", None)
        if goblets is None or goblet_types is None:
            return
        for pos, gtype in zip(goblets, goblet_types):
            color = self._colors["goblet_pos"] if gtype > 0 else self._colors["goblet_neg"]
            center = self._world_to_screen(pos)
            pygame.draw.circle(self._surface, color, center, 6)
            pygame.draw.circle(self._surface, (20, 20, 20), center, 6, width=1)

    def _draw_entities(self) -> None:
        adv = getattr(self.env, "adventurer", None)
        ghost = getattr(self.env, "ghost", None)
        if adv is not None:
            pygame.draw.circle(self._surface, self._colors["adventurer"], self._world_to_screen(adv), 8)
            pygame.draw.circle(self._surface, (10, 10, 10), self._world_to_screen(adv), 8, width=1)
        if ghost is not None:
            pygame.draw.circle(self._surface, self._colors["ghost"], self._world_to_screen(ghost), 8)
            pygame.draw.circle(self._surface, (10, 10, 10), self._world_to_screen(ghost), 8, width=1)

    def _draw_action(self, action: Optional[np.ndarray]) -> None:
        if action is None:
            return
        adv = getattr(self.env, "adventurer", None)
        if adv is None:
            return
        action = np.asarray(action, dtype=float).reshape(2)
        v_max = float(getattr(self.env, "v_max", 1.0))
        if v_max <= 0.0:
            return
        norm = float(np.linalg.norm(action))
        if norm > v_max and norm > 0.0:
            action = action / norm * v_max
        arrow = action / v_max * (0.3 * self.torus_size)
        start = np.asarray(adv, dtype=float)
        end = start + arrow
        start_px = self._world_to_screen(start)
        end_px = self._world_to_screen(end)
        pygame.draw.line(self._surface, self._colors["arrow"], start_px, end_px, width=2)
        self._draw_arrow_head(start_px, end_px)

    def _draw_arrow_head(self, start: Tuple[int, int], end: Tuple[int, int]) -> None:
        vec = np.array([end[0] - start[0], end[1] - start[1]], dtype=float)
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-6:
            return
        direction = vec / norm
        perp = np.array([-direction[1], direction[0]])
        tip = np.array(end, dtype=float)
        left = tip - direction * 8 + perp * 5
        right = tip - direction * 8 - perp * 5
        pygame.draw.polygon(
            self._surface,
            self._colors["arrow"],
            [tip.tolist(), left.tolist(), right.tolist()],
        )

    def _draw_events(self) -> None:
        events = getattr(self.env, "last_events", {})
        lines = []
        if events.get("caught"):
            lines.append("Caught!")
        if events.get("picked"):
            pick_type = events.get("picked_type", 0.0)
            label = "Picked +" if pick_type > 0 else "Picked -"
            lines.append(label)
        if events.get("restart"):
            lines.append("Restart")
        step = getattr(self.env, "_step_count", None)
        if step is not None:
            lines.append(f"Step {step}")

        if not lines:
            return
        x = 12
        y = 10
        for line in lines:
            surface = self._font.render(line, True, self._colors["text"])
            self._surface.blit(surface, (x, y))
            y += surface.get_height() + 2
