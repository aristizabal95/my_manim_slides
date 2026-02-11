import os
import random
from typing import Literal

from manim import *

filepath = os.path.dirname(os.path.abspath(__file__))
head_path = os.path.join(filepath, "robot-head.svg")
body_path = os.path.join(filepath, "robot-body.svg")


class HeadLedMove(Animation):
    """Moves the robot horizontally with a head-led impulse: tilt towards goal,
    move, then straighten. Rotation and movement overlap for a fluid motion."""

    def __init__(
        self,
        mobject: Mobject,
        shift: np.ndarray | tuple | list,
        tilt_angle: float = 15 * DEGREES,
        **kwargs,
    ):
        super().__init__(mobject, **kwargs)
        self.shift_vector = np.array(shift, dtype=float)
        self.tilt_angle = tilt_angle
        self._start_center = np.array(mobject.get_center())
        self._last_angle = 0.0  # Track rotation ourselves (mobjects have no .angle)

    def _tilt_at(self, alpha: float) -> float:
        """Smooth tilt hump: sin(π·α) peaks at 0.5 with zero derivative there."""
        return np.sin(PI * alpha)

    def interpolate_mobject(self, alpha: float) -> None:
        alpha = self.rate_func(alpha)

        start_pos = self._start_center
        direction = np.sign(self.shift_vector[0]) if self.shift_vector[0] != 0 else 1
        max_tilt = -direction * self.tilt_angle

        # Both run across the full alpha range
        current_pos = start_pos + self.shift_vector * alpha
        tilt_amount = self._tilt_at(alpha)
        current_angle = max_tilt * tilt_amount

        self.mobject.move_to(current_pos)
        angle_delta = current_angle - self._last_angle
        self.mobject.rotate(angle_delta, about_point=current_pos)
        self._last_angle = current_angle


class Robot(VGroup):
    def __init__(self,
    render_body: bool = True,
    color: str = BLACK,
    fill_color: str = WHITE,
    fill_opacity: float = 1.0,
    stroke_width: float = 4,
    shadow: bool = True,
    shadow_color: str = GREY_D,
    shadow_opacity: float = 0.5,
    shadow_offset: np.ndarray | tuple | list = (DOWN + RIGHT) * 0.1,
    **kwargs
    ):
        super().__init__(**kwargs)

        self.head = SVGMobject(
            file_name=head_path,
            color=color,
            stroke_color=color,
            stroke_width=stroke_width,
            fill_color=fill_color,
            fill_opacity=fill_opacity,
        ).scale(0.5)
        if shadow:
            self.head_shadow = SVGMobject(
                file_name=head_path,
                color=shadow_color,
                stroke_color=shadow_color,
                stroke_width=0,
                fill_color=shadow_color,
                fill_opacity=shadow_opacity,
            ).scale(0.5)
            self.head_shadow.move_to(self.head.get_center() + shadow_offset)
            self.add(self.head_shadow)
        self.add(self.head)

        self.eyes_pos = self.head.get_center() + DOWN * 0.1
        self.eyes = VGroup(
            Circle(color=color, radius=0.1, stroke_width=stroke_width).shift(LEFT * 0.25),
            Circle(color=color, radius=0.1, stroke_width=stroke_width).shift(RIGHT * 0.25),
        ).move_to(self.eyes_pos)
        self.add(self.eyes)

        self.eyes_happy = VGroup(
            Arc(angle=PI, color=color, radius=0.1, stroke_width=stroke_width).shift(LEFT * 0.25),
            Arc(angle=PI, color=color, radius=0.1, stroke_width=stroke_width).shift(RIGHT * 0.25),
        ).move_to(self.eyes_pos)
        # self.add(self.eyes_happy)

        self.eyes_sad = VGroup(
            Arc(angle=PI, color=color, radius=0.1, start_angle=PI, stroke_width=stroke_width).shift(LEFT * 0.25),
            Arc(angle=PI, color=color, radius=0.1, start_angle=PI, stroke_width=stroke_width).shift(RIGHT * 0.25),
        ).move_to(self.eyes_pos)

        self._eyes_map: dict[str, VGroup] = {
            "default": self.eyes,
            "happy": self.eyes_happy,
            "sad": self.eyes_sad,
        }
        self._eyes_state: Literal["default", "happy", "sad"] = "default"

        if render_body:
            self.body = SVGMobject(
                file_name=body_path,
                color=color,
                stroke_color=color,
                stroke_width=stroke_width,
                fill_color=fill_color,
                fill_opacity=fill_opacity,
            )
            self.body.next_to(self.head, DOWN)

            if shadow:
                self.body_shadow = SVGMobject(
                    file_name=body_path,
                    color=shadow_color,
                    stroke_color=shadow_color,
                    stroke_width=0,
                    fill_color=shadow_color,
                    fill_opacity=shadow_opacity,
                )
                self.body_shadow.move_to(self.body.get_center() + shadow_offset)
                self.add(self.body_shadow)
            self.add(self.body)

    def set_eyes(
        self, state: Literal["default", "happy", "sad"]
    ) -> None:
        """Switch the robot's eyes to default (circles), happy, or sad."""
        if state == self._eyes_state:
            return
        old_eyes = self._eyes_map[self._eyes_state]
        new_eyes = self._eyes_map[state]
        new_eyes.move_to(old_eyes)
        self.remove(old_eyes)
        self.add(new_eyes)
        self._eyes_state = state

    def scale(self, scale_factor: float) -> None:
        for eye_state, eye in self._eyes_map.items():
            if eye_state == self._eyes_state:
                continue
            eye.scale(scale_factor)
        return super().scale(scale_factor)

    def animate_eyes(
        self, state: Literal["default", "happy", "sad"], **kwargs
    ) -> Animation:
        """Return an animation that transforms the eyes to the given state."""
        if state == self._eyes_state:
            return Wait(1e-6)
        old_eyes = self._eyes_map[self._eyes_state]
        new_eyes = self._eyes_map[state]
        new_eyes.move_to(old_eyes)
        self._eyes_state = state
        return ReplacementTransform(old_eyes, new_eyes, **kwargs)

    def head_led_move(
        self,
        shift: np.ndarray | tuple | list,
        tilt_angle: float = 15 * DEGREES,
        **kwargs,
    ) -> HeadLedMove:
        """Return a HeadLedMove animation for this robot."""
        return HeadLedMove(self, shift=shift, tilt_angle=tilt_angle, **kwargs)


class RobotExample(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        robot = Robot(render_body=True)
        self.add(robot)
        for _ in range(5):
            move_dir = random.choice([RIGHT, LEFT])
            eyes_state = random.choice(["default", "happy", "sad"])
            self.play(robot.head_led_move(move_dir * 2, run_time=0.5))
            self.play(robot.animate_eyes(eyes_state, run_time=0.3))