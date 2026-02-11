import os
import random
import sys
from datetime import date, timedelta
from pathlib import Path
from coin import Coin

# Allow importing rl.robot when this file is run directly by manim
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from manim import *
from manim_slides import Slide
from factored_manim.mobjects.slides import *
from factored_manim.mobjects import TextBox, UnorderedList, TwistingLines
import factored_manim.branding.colors as fcolors
from manim_ml.neural_network import Convolutional2DLayer, FeedForwardLayer, NeuralNetwork

from rl.robot.robot import Robot


class Wiggle(Animation):
    """Quick horizontal shake to indicate negative feedback."""

    def __init__(
        self,
        mobject: Mobject,
        amplitude: float = 0.08,
        num_wiggles: int = 4,
        **kwargs,
    ):
        super().__init__(mobject, **kwargs)
        self.amplitude = amplitude
        self.num_wiggles = num_wiggles
        self._start_center = np.array(mobject.get_center())

    def interpolate_mobject(self, alpha: float) -> None:
        alpha = self.rate_func(alpha)
        # One full sine cycle per wiggle
        offset = self.amplitude * np.sin(2 * np.pi * self.num_wiggles * alpha)
        self.mobject.move_to(self._start_center + RIGHT * offset)


class Intro(Slide):
    def _flip(self, coin: Coin, num_semiflips: int = 1, run_time: float = 1):
        cur_flip = 0
        individual_run_time = run_time / 2 / num_semiflips
        while cur_flip < num_semiflips:
            if cur_flip % 2 == 0:
                self.play(Rotate(coin, 90*DEGREES, axis=RIGHT, about_point=coin.get_center(), run_time=individual_run_time))
                coin.add_to_back(coin.edge, coin.top)
                self.play(Rotate(coin, 90*DEGREES, axis=RIGHT, about_point=coin.get_center(), run_time=individual_run_time))
            else:
                self.play(Rotate(coin, 90*DEGREES, axis=RIGHT, about_point=coin.get_center(), run_time=individual_run_time))
                coin.add_to_back(coin.edge, coin.bottom)
                self.play(Rotate(coin, 90*DEGREES, axis=RIGHT, about_point=coin.get_center(), run_time=individual_run_time))
            cur_flip += 1

    def title_slide(self):
        with fcolors.use_theme("dark", self.camera):
            title_slide = TitleSlide(
                title="Introduction to Reinforcement Learning",
                subtitle="Foundations Group - Cohort V",
                title_font_size=48,
            )
            self.play(Write(title_slide))
            self.next_slide()
            self.play(FadeOut(title_slide))

    def agenda(self):
        with fcolors.use_theme("light", self.camera):
            agenda_slide = AgendaLeftSlide(
                items=[
                    "Introduction to RL",
                    "Types of RL",
                    "Foundations Group\nObjectives",
                    "Cohort Logistics",
                ],
            )
            self.play(Write(agenda_slide))
            self.next_slide()
            self.play(FadeOut(agenda_slide))

    def overview_section(self):
        with fcolors.use_theme("dark", self.camera):
            overview_section = SectionSlide(
                idx=1,
                title="Overview of RL",
            )
            self.play(Write(overview_section))
            self.next_slide()
            self.play(FadeOut(overview_section))

    def rl_setting(self):
        with fcolors.use_theme("light", self.camera):
            slide = ContrastSegmentBGSlide(
                contrast_segment_width_ratio=0.4,
            )
            with fcolors.use_theme("dark"):
                title_text = TextBox(
                    title="The Reinforcement Learning Setting",
                    subtext=None,
                    title_font_size=36,
                    max_width=slide.containers.contrast_segment.width - 1,
                    separator=False
                ).move_to(slide.containers.contrast_segment.get_corner(UL) + DOWN * 0.5 + RIGHT * 0.5, aligned_edge=UL)
                ulist = UnorderedList(
                    items=[
                        "An agent and an environment",
                        "Observations, Actions and Rewards",
                        "Agent affects environment, which\naffects observations",
                        "Maximize total received rewards",
                        "Data collection is part of the learning process",
                    ]
                )
                ulist.move_to(slide.containers.contrast_segment.get_center())

            # Load the image (ImageMobject is not a VMobject, so keep it separate from the slide VGroup)
            image_path = os.path.join(os.path.dirname(__file__), "rl_setting.jpg")
            rl_setting_img = ImageMobject(image_path)
            rl_setting_img.scale_to_fit_width(slide.containers.background_segment.width + 2)
            rl_setting_img.set_z_index(-1)
            rl_setting_img.move_to(slide.containers.background_segment.get_center())

            self.play(Write(slide), Write(title_text), Write(ulist), FadeIn(rl_setting_img))
            self.next_slide()
            self.play(FadeOut(slide), FadeOut(title_text), FadeOut(ulist), FadeOut(rl_setting_img))

    def _robot_trial(
        self,
        start_idx: int,
        steps: list[int],
        left_arrow_mag: list[float],
        robot: Robot,
        squares: VGroup,
        cell_centers: list[np.ndarray],
    ) -> None:
            num_cells = len(cell_centers)
            robot.move_to(cell_centers[start_idx])
            path = [start_idx]
            pos = start_idx
            for i, step in enumerate(steps):
                pos = max(0, min(num_cells - 1, pos + step))
                path.append(pos)

            robot.set_eyes("default")
            self.play(
                FadeIn(robot),
            )

            reward_text = None

            # Animate random walk
            for k in range(1, len(path)):
                prev_l_mag = left_arrow_mag[k - 1]
                prev_r_mag = 1 - prev_l_mag
                robot_left = robot.get_edge_center(LEFT)
                robot_right = robot.get_edge_center(RIGHT)
                left_arrow = Arrow(start=robot_left, end=robot_left + LEFT * prev_l_mag * 3, color=BLACK)
                right_arrow = Arrow(start=robot_right, end=robot_right + RIGHT * prev_r_mag * 3, color=BLACK)
                self.play(FadeIn(left_arrow), FadeIn(right_arrow), run_time=0.2)
                self.next_slide()
                prev_idx, curr_idx = path[k - 1], path[k]
                shift = (cell_centers[curr_idx] - cell_centers[prev_idx])
                if shift[0] < 0:
                    self.play(Indicate(left_arrow, run_time=0.2))
                else:
                    self.play(Indicate(right_arrow), run_time=0.2)
                self.play(FadeOut(left_arrow), FadeOut(right_arrow), run_time=0.2)
                self.play(robot.head_led_move(shift, run_time=0.4))

                if curr_idx == 0:
                    # Reward -10, sad eyes (no animation), wiggle
                    reward_text = Text(
                        "-10",
                        font_size=48,
                        color=RED,
                        weight=BOLD,
                    ).next_to(squares[0], UP, buff=0.4)
                    robot.set_eyes("sad")
                    self.play(
                        FadeIn(reward_text, shift=UP * 0.5),
                        Wiggle(robot, amplitude=0.1, num_wiggles=4, run_time=0.5)
                    )
                    
                    break
                if curr_idx == num_cells - 1:
                    # Reward 10, happy eyes (no animation)
                    reward_text = Text(
                        "10",
                        font_size=48,
                        color=GREEN,
                        weight=BOLD,
                    ).next_to(squares[-1], UP, buff=0.4)
                    robot.set_eyes("happy")
                    self.play(
                        FadeIn(reward_text, shift=UP * 0.5),
                        Flash(robot),
                    )
                    break
            self.next_slide()
            self.play(FadeOut(reward_text), FadeOut(robot))

    def rl_example(self):
        with fcolors.use_theme("light", self.camera):
            slide = TitleBGSlide()
            title_text = TextBox(
                title="Example",
                subtext=None,
                title_font_size=36,
                separator=False
            ).move_to(slide.containers.main.get_corner(UL), aligned_edge=UL)

            self.play(Write(title_text), Write(slide))

            # 1x5 grid in the center of the slide
            num_cells = 5
            cell_side = 2
            grid_center = ORIGIN
            cell_centers = [
                grid_center + RIGHT * (i - (num_cells - 1) / 2) * cell_side
                for i in range(num_cells)
            ]

            squares = VGroup()
            for i, center in enumerate(cell_centers):
                sq = Square(side_length=cell_side, stroke_width=2, stroke_color=BLACK)
                sq.move_to(center)
                if i == 0:
                    sq.set_fill(RED, opacity=0.5)
                elif i == num_cells - 1:
                    sq.set_fill(GREEN, opacity=0.5)
                else:
                    sq.set_fill(WHITE, opacity=0.3)
                squares.add(sq)

            robot = Robot(render_body=True, color=GREY_D, stroke_width=2).scale(0.5)
            start_idx = 2
            robot.move_to(cell_centers[start_idx])

            self.play(
                FadeIn(squares, shift=RIGHT * 2),
                slide.submobjects[0].animate.shift(RIGHT * 2),
            )

            # Precompute random path until we land on left-most (index 0)
            trials_steps = [
                [-1, 1, -1, -1],
                [1, -1, -1, 1, 1, 1],
                [1, 1, 1],
            ]
            trials_left_arrow_mag = [
                [0.5, 0.5, 0.5, 0.5],
                [0.4, 0.5, 0.4, 0.25, 0.4, 0.5, 0.5],
                [0.25, 0.25, 0.25]
            ]
            num_trials = len(trials_steps)
            self.next_slide()
            for trial_idx in range(num_trials):
                self._robot_trial(
                    start_idx=start_idx,
                    steps=trials_steps[trial_idx],
                    left_arrow_mag=trials_left_arrow_mag[trial_idx],
                    robot=robot,
                    squares=squares,
                    cell_centers=cell_centers,
                )
            self.next_slide()
            self.play(FadeOut(squares), FadeOut(slide), FadeOut(title_text))

    def mdp(self):
        with fcolors.use_theme("light", self.camera):
            slide = ContrastSegmentBGSlide(
                contrast_segment_width_ratio=0.39,
            )

            font_size = 28
            with fcolors.use_theme("dark"):
                title_text = TextBox(
                    title="Markov Decision Process",
                    subtext=None,
                    title_font_size=36,
                    separator=False,
                    max_width=slide.containers.contrast_segment.width,
                ).move_to(slide.containers.contrast_segment.get_corner(UL), aligned_edge=UL)
                states_math_str = "S = \\{ s_1, s_2, \\ldots, s_n \\}"
                actions_math_str = "A = \\{ a_1, a_2, \\ldots, a_m \\}"
                transition_math_str = "P(s_{t+1} | s_t, a_t)"
                reward_math_str = "R(s_t, a_t, s_{t+1})"
                states_text = MathTex(
                    "\\text{States: }",
                    states_math_str,
                    font_size=28,
                    color=fcolors.current_theme.text
                ).next_to(title_text, DOWN * 2, aligned_edge=LEFT)
                actions_text = MathTex(
                    "\\text{Actions: } ",
                    actions_math_str,
                    font_size=28,
                    color=fcolors.current_theme.text
                ).next_to(states_text, DOWN, aligned_edge=LEFT)
                transition_text = MathTex(
                    "\\text{Transition Function: } ",
                    transition_math_str,
                    font_size=28,
                    color=fcolors.current_theme.text
                ).next_to(actions_text, DOWN, aligned_edge=LEFT)
                reward_text = MathTex(
                    "\\text{Reward Function: } ",
                    reward_math_str,
                    font_size=28,
                    color=fcolors.current_theme.text
                ).next_to(transition_text, DOWN, aligned_edge=LEFT)

                explanation_text_group = VGroup(states_text, actions_text, transition_text, reward_text)

            cool_color = BLUE_D
            hot_color = GOLD
            overheat_color = DARK_BROWN
            fast_color = RED_D
            slow_color = BLUE

            states_example_text = MathTex(
                "S = \\{ ", "Cool", ", ", "Hot", ", ", "Overheat", " \\}",
                font_size=28,
            ).move_to(slide.containers.background_segment.get_corner(UR), aligned_edge=UR)
            states_example_text.set_color_by_tex("Cool", cool_color)
            states_example_text.set_color_by_tex("Hot", hot_color)
            states_example_text.set_color_by_tex("Overheat", overheat_color)

            actions_example_text = MathTex("A = \\{ ", "Fast", ", ", "Slow", " \\}",
                font_size=28,
            ).next_to(states_example_text, DOWN, aligned_edge=RIGHT)
            actions_example_text.set_color_by_tex("Fast", fast_color)
            actions_example_text.set_color_by_tex("Slow", slow_color)


            cool_state = Circle(color=cool_color, radius=0.7, fill_opacity=0.0).move_to([-1, -2, 0])
            cool_text = Text("Cool", font_size=20, color=cool_color).move_to(cool_state.get_center())
            cool_group = VGroup(cool_state, cool_text)
            cool_group.og_color = cool_color

            hot_state = Circle(color=hot_color, radius=0.7, fill_opacity=0.0).move_to([3, 2, 0])
            hot_text = Text("Hot", font_size=20, color=hot_color).move_to(hot_state.get_center())
            hot_group = VGroup(hot_state, hot_text)
            hot_group.og_color = hot_color

            overheat_state = Circle(color=overheat_color, radius=0.7, fill_opacity=0.0).move_to([7, 0, 0])
            overheat_text = Text("Overheat", font_size=20, color=overheat_color).move_to(overheat_state.get_center())
            overheat_group = VGroup(overheat_state, overheat_text)
            overheat_group.og_color = overheat_color

            states_group = VGroup(cool_group, hot_group, overheat_group).scale(0.9)

        new_slide = ContrastSegmentBGSlide(
            contrast_segment_width_ratio=0.48,
        )

        # Cool transitions
        self_ref_r = 0.35
        cfh_arrow = CurvedArrow(
            start_point=cool_state.get_top(),
            end_point=hot_state.get_left(),
            radius=-7,
            color=fast_color
        )
        cfh_arrow.starting_point = cool_state.get_top()
        cfh_arrow.end_point = hot_state.get_left()
        cfh_prob = MathTex("0.5", font_size=28, color=fcolors.current_theme.text).move_to(cfh_arrow.get_center())
        cfh_reward = MathTex("+2", font_size=28, color=GREEN).move_to(cfh_arrow.get_center() + UL * 0.5)
        cfh_group = VGroup(cfh_arrow, cfh_prob, cfh_reward)

        cfc_arrow = Arc(
            radius=self_ref_r,
            start_angle=0*DEGREES,
            angle=235*DEGREES,
            color=fast_color
        ).move_to(cool_state.get_corner(UL) + [0.3, 0.05, 0]).add_tip()
        cfc_arrow.starting_point = cool_state.get_top()
        cfc_arrow.end_point = cool_state.point_at_angle(135*DEGREES)
        cfc_prob = MathTex("0.5", font_size=28, color=fcolors.current_theme.text).move_to(cfc_arrow.get_center())
        cfc_reward = MathTex("+2", font_size=28, color=GREEN).move_to(cfc_arrow.get_center() + UP * 0.8)
        cfc_group = VGroup(cfc_arrow, cfc_prob, cfc_reward)

        csc_arrow = Arc(
            radius=self_ref_r,
            start_angle=90*DEGREES,
            angle=235*DEGREES,
            color=slow_color
        ).move_to(cool_state.get_corner(DL) + [-0.05, 0.3, 0]).add_tip()
        csc_arrow.starting_point = cool_state.get_left()
        csc_arrow.end_point = cool_state.point_at_angle(225*DEGREES)

        csc_prob = MathTex("1.0", font_size=28, color=fcolors.current_theme.text).move_to(csc_arrow.get_center())
        csc_reward = MathTex("+1", font_size=28, color=GREEN).move_to(csc_arrow.get_center() + LEFT * 0.8)
        csc_group = VGroup(csc_arrow, csc_prob, csc_reward)

        cf_group = VGroup(cfc_group, cfh_group)

        cold_transitions_group = VGroup(cf_group, csc_group)
    
        # Hot transitions
        hsc_arrow = CurvedArrow(
            start_point=hot_state.get_bottom(),
            end_point=cool_state.get_right(),
            radius=-7,
            color=slow_color
        )
        hsc_arrow.starting_point = hot_state.get_bottom()
        hsc_arrow.end_point = cool_state.get_right()
        hsc_prob = MathTex("0.5", font_size=28, color=fcolors.current_theme.text).move_to(hsc_arrow.get_center())
        hsc_reward = MathTex("+1", font_size=28, color=GREEN).move_to(hsc_arrow.get_center() + DR * 0.5)
        hsc_group = VGroup(hsc_arrow, hsc_prob, hsc_reward)
    
        hfo_arrow = CurvedArrow(
            start_point=hot_state.get_right(),
            end_point=overheat_state.point_at_angle(135*DEGREES),
            radius=-7,
            color=fast_color
        )
        hfo_arrow.starting_point = hot_state.get_right()
        hfo_arrow.end_point = overheat_state.point_at_angle(135*DEGREES)
        hfo_prob = MathTex("1.0", font_size=28, color=fcolors.current_theme.text).move_to(hfo_arrow.get_center() + DOWN * 0.2)
        hfo_reward = MathTex("-10", font_size=28, color=RED).move_to(hfo_arrow.get_center() + UP * 0.5)
        hfo_group = VGroup(hfo_arrow, hfo_prob, hfo_reward)
    
        hsh_arrow = Arc(
            radius=self_ref_r,
            start_angle=180*DEGREES,
            angle=235*DEGREES,
            color=slow_color
        ).move_to(hot_state.get_corner(DR) + [-0.3, -0.05, 0]).add_tip()
        hsh_arrow.starting_point = hot_state.get_bottom()
        hsh_arrow.end_point = hot_state.point_at_angle(315*DEGREES)
        hsh_prob = MathTex("0.5", font_size=28, color=fcolors.current_theme.text).move_to(hsh_arrow.get_center())
        hsh_reward = MathTex("+1", font_size=28, color=GREEN).move_to(hsh_arrow.get_center() + DOWN * 0.8)
        hsh_group = VGroup(hsh_arrow, hsh_prob, hsh_reward)

        hs_group = VGroup(hsc_group, hsh_group)

        hot_transitions_group = VGroup(hs_group, hfo_group)

        mdp_group = VGroup(states_group, cold_transitions_group, hot_transitions_group).scale(0.8)

        # RL Framework
        agent_text = Text("Agent", font_size=40, color=YELLOW)
        agent_box = SurroundingRectangle(agent_text, color=YELLOW, buff=0.3)
        agent_group = VGroup(agent_box, agent_text)
        agent_group.shift(UP * 1)

        env_text = Text("Environment", font_size=40, color=MAROON)
        env_box = SurroundingRectangle(env_text, color=MAROON, buff=0.3)
        env_group = VGroup(env_box, env_text)
        env_group.next_to(agent_group, DOWN * 3)

        rl_framework_center = (agent_group.get_center() + env_group.get_center()) / 2

        # Action arrow
        action_arrow = VGroup(
            Line(
                start=agent_group.get_right(),
                end=agent_group.get_right() + RIGHT * 2,
                color=WHITE
            ),
            Line(
                start=agent_group.get_right() + RIGHT * 2,
                end=[(agent_group.get_right() + RIGHT * 2)[0], env_group.get_right()[1], 0],
                color=WHITE
            ),
            Line(
                start=[(agent_group.get_right() + RIGHT * 2)[0], env_group.get_right()[1], 0],
                end=env_group.get_right(),
                color=WHITE
            ).add_tip()
        )
        action_arrow_label = MathTex("a").next_to(action_arrow, RIGHT)
        action_arrow_group = VGroup(action_arrow, action_arrow_label)

        # Obs & Reward arrows
        epsilon = 0.2
        obs_arrow = VGroup(
            Line(
                start=env_group.get_left() + DOWN * epsilon,
                end=[(agent_group.get_left() + LEFT * 2 - epsilon)[0], (env_group.get_left() + DOWN * epsilon)[1], 0],
            ),
            Line(
                start=[(agent_group.get_left() + LEFT * 2 - epsilon)[0], (env_group.get_left() + DOWN * epsilon)[1], 0],
                end=agent_group.get_left() + LEFT * 2 + (LEFT + UP) * epsilon,
                color=WHITE
            ),
            Line(
                start=agent_group.get_left() + LEFT * 2 + (LEFT + UP) * epsilon,
                end=agent_group.get_left() + UP * epsilon,
                color=WHITE
            ).add_tip()
        )
        obs_arrow_label = MathTex("o").next_to(obs_arrow, LEFT)
        obs_arrow_group = VGroup(obs_arrow, obs_arrow_label)
        # TODO: finish function and use that instead
        epsilon = -epsilon
        reward_arrow = VGroup(
            Line(
                start=env_group.get_left() + DOWN * epsilon,
                end=[(agent_group.get_left() + LEFT * 2 - epsilon)[0], (env_group.get_left() + DOWN * epsilon)[1], 0],
            ),
            Line(
                start=[(agent_group.get_left() + LEFT * 2 - epsilon)[0], (env_group.get_left() + DOWN * epsilon)[1], 0],
                end=agent_group.get_left() + LEFT * 2 + (LEFT + UP) * epsilon,
                color=WHITE
            ),
            Line(
                start=agent_group.get_left() + LEFT * 2 + (LEFT + UP) * epsilon,
                end=agent_group.get_left() + UP * epsilon,
                color=WHITE
            ).add_tip()
        )
        reward_arrow_label = MathTex("r").next_to(reward_arrow, LEFT).shift(RIGHT * 0.8)
        reward_arrow_group = VGroup(reward_arrow, reward_arrow_label)

        arrows_group = VGroup(action_arrow_group, obs_arrow_group, reward_arrow_group)

        rl_framework_group = VGroup(
            agent_group,
            env_group,
            arrows_group
        ).scale(0.6).move_to(new_slide.containers.contrast_segment.get_center())

        # Robot
        robot = VGroup(
            Dot(color=WHITE, radius=0.1).shift(LEFT * 0.5),
            Dot(color=WHITE, radius=0.1).shift(RIGHT * 0.5),
            Dot(color=WHITE, radius=0.2).shift(UP * 1.2),
            RoundedRectangle(width=2, height=1, corner_radius=0.5, color=WHITE, stroke_width=4),
            Line(start=UP * 0.5, end=UP * 1.2, color=WHITE, stroke_width=4),
        ).scale(0.5)

        coin = Coin(radius=0.4, height=0.1)
        coin.next_to(robot, RIGHT)

        random_agent_group = VGroup(robot, coin)
        random_agent_group.move_to(agent_group.get_center())


        self.play(Write(slide), Write(title_text))
        self.play(Write(states_text))
        self.play(
            Create(cool_state),
            Create(hot_state),
            Create(overheat_state),
            Write(cool_text),
            Write(hot_text),
            Write(overheat_text),
            Write(states_example_text),
        )
        self.next_slide()
        self.play(Write(actions_text))
        self.play(Write(actions_example_text))
        self.next_slide()

        self.play(Write(transition_text))
        self.play(
            Create(cfh_arrow),
            Create(cfc_arrow),
            Create(csc_arrow),
            Create(hsc_arrow),
            Create(hfo_arrow),
            Create(hsh_arrow),
            Write(cfh_prob),
            Write(cfc_prob),
            Write(csc_prob),
            Write(hsc_prob),
            Write(hfo_prob),
            Write(hsh_prob),
            run_time=2
        )
        self.next_slide()

        self.play(Write(reward_text))
        self.play(
            Write(cfh_reward),
            Write(cfc_reward),
            Write(csc_reward),
            Write(hsc_reward),
            Write(hfo_reward),
            Write(hsh_reward),
        )
        self.next_slide()

        # Deterministic transitions
        self.play(
            Indicate(csc_group),
            Indicate(hfo_group),
            run_time=2,
        )
        self.next_slide()

        # Probabilistic transitions
        self.play(
            Indicate(cf_group),
            Indicate(hs_group),
            run_time=2,
        )
        self.next_slide()

        # Terminal States
        self.play(
            Indicate(overheat_group),
            run_time=2,
        )
        self.next_slide()

        self.play(
            FadeOut(explanation_text_group, shift=LEFT * 5),
            mdp_group.animate.shift(RIGHT).scale(0.9),
            Transform(slide, new_slide),
        )

        self.play(
            Create(agent_group),
            Create(env_group),
        )
        self.play(
            Create(action_arrow_group),
            Create(obs_arrow_group),
            Create(reward_arrow_group),
        )
        self.play(
            Wiggle(env_group),
            Wiggle(mdp_group),
            run_time=2
        )
        self.next_slide()

        # SIMULATE A GAME WITH RANDOM AGENT

        np.random.seed(42)
        current_state = cool_group
        active_state_color = YELLOW

        self.play(rl_framework_group.animate.shift(DOWN * 1.5))
        self.play(
            Create(robot),
            Create(coin),
            current_state.animate.set_color(active_state_color),
            FocusOn(current_state),
        )
        self.wait(1)

        while current_state != overheat_group:
            self.next_slide()
            old_state = current_state
            num_semiflips = np.random.randint(1, 11)
            action = "slow" if num_semiflips % 2 == 0 else "fast"
            if action == "slow" and current_state == cool_group:
                current_state = cool_group
                transition_arrow = csc_arrow
                reward = 1
            elif action == "fast" and current_state == cool_group:
                current_state = hot_group if np.random.random() < 0.5 else cool_group
                transition_arrow = cfh_arrow if current_state == hot_group else cfc_arrow
                reward = 2
            elif action == "slow" and current_state == hot_group:
                current_state = cool_group if np.random.random() < 0.5 else hot_group
                transition_arrow = hsc_arrow if current_state == cool_group else hfo_arrow
                reward = 1
            elif action == "fast" and current_state == hot_group:
                current_state = overheat_group
                transition_arrow = hfo_arrow
                reward = -10

            reward_text = Text(str(reward), font_size=40, color=GREEN if reward > 0 else RED)
            reward_text.next_to(robot, UP)

            transition_dot = Dot(
                color=active_state_color,
                radius=0.1,
            ).move_to(transition_arrow.point_from_proportion(0.0))
            state_clone = old_state.copy()
            self.add(state_clone)
            old_state.set_color(old_state.og_color)

            self._flip(coin, num_semiflips=num_semiflips, run_time=1)
            self.play(
                Transform(state_clone[0], transition_dot),
                state_clone[1].animate.set_color(old_state.og_color),
                run_time=0.5,
                rate_func=rate_functions.ease_in_sine
            )
            self.remove(state_clone)
            old_state[1].set_color(old_state.og_color)
            self.play(
                MoveAlongPath(transition_dot, transition_arrow, rate_func=linear),
                Succession(FadeIn(reward_text, shift=UP * 0.5), FadeOut(reward_text, shift=UP * 0.5)),
                run_time=0.5,
            )
            state_clone = current_state.copy()
            state_clone.set_color(active_state_color)
            self.play(
                Transform(transition_dot, state_clone[0]),
                current_state[1].animate.set_color(active_state_color),
                run_time=0.5,
                rate_func=rate_functions.ease_out_sine
            )
            current_state.set_color(active_state_color)
            self.remove(state_clone)
            self.remove(transition_dot)

        self.next_slide()
        self.play(
            FadeOut(rl_framework_group),
            FadeOut(robot),
            FadeOut(coin),
            FadeOut(title_text),
            FadeOut(slide),
            FadeOut(mdp_group),
            FadeOut(states_example_text),
            FadeOut(actions_example_text),
        )

    def types_intro(self):
        with fcolors.use_theme("dark", self.camera):
            slide = SectionSlide(
                idx=2,
                title="Types of RL",
            )
            self.play(Write(slide))
            self.next_slide()
            self.play(FadeOut(slide))

    def types_of_rl(self):
        with fcolors.use_theme("light", self.camera):
            slide = ContrastSegmentBGSlide(
                contrast_segment_width_ratio=0.5,
            )
            self.play(Write(slide))
            # Multi-Armed Bandits
            with fcolors.use_theme("dark"):
                mab_title_text = TextBox(
                    title="Multi-Armed Bandits",
                    subtext=None,
                    title_font_size=36,
                    separator=False,
                    max_width=slide.containers.contrast_segment.width,
                ).move_to(slide.containers.contrast_segment.get_corner(UL), aligned_edge=UL)
                image_path = os.path.join(os.path.dirname(__file__), "one-armed-bandit.jpg")
                mab_img = ImageMobject(image_path).scale(0.4)
                mab_img.next_to(mab_title_text, DOWN * 2, aligned_edge=LEFT)
                mab_ulist = UnorderedList(
                    items=[
                        "One-step decision problem",
                        "Identify best action to take",
                        "May be state-less (Multi-Armed Bandit)\nor state-based (Contextual Bandit)",
                        "Used in A/B testing, recommendation systems, etc.",
                    ],
                )
                mab_ulist.next_to(mab_img, DOWN * 2, aligned_edge=LEFT)
                image_shift = slide.containers.contrast_segment.get_center()[0] - mab_img.get_center()[0]
                mab_img.shift(image_shift * RIGHT)
            
            self.play(Write(mab_title_text), FadeIn(mab_img), Write(mab_ulist))
            self.next_slide()

            # Tabular RL
            with fcolors.use_theme("light"):
                tabular_rl_title_text = TextBox(
                    title="Tabular RL",
                    subtext=None,
                    title_font_size=36,
                    separator=False,
                    max_width=slide.containers.background_segment.width,
                ).move_to(slide.containers.background_segment.get_corner(UL), aligned_edge=UL)
                image_path = os.path.join(os.path.dirname(__file__), "frozenlake.png")
                frozenlake_img = ImageMobject(image_path).scale(0.6)
                frozenlake_img.next_to(tabular_rl_title_text, DOWN * 2, aligned_edge=LEFT)
                tabular_rl_ulist = UnorderedList(
                    items=[
                        "Multi-step state-based decision problem",
                        "Identify best action to take in each state (tabular Q-learning,\nSARSA, etc.)",
                        "Handles discrete states and actions",
                        "Used in inventory management, resource\n allocation, etc.",
                    ],
                )
                tabular_rl_ulist.next_to(frozenlake_img, DOWN * 2, aligned_edge=LEFT)
                image_shift = slide.containers.background_segment.get_center()[0] - frozenlake_img.get_center()[0]
                frozenlake_img.shift(image_shift * RIGHT)

            self.play(Write(tabular_rl_title_text), FadeIn(frozenlake_img), Write(tabular_rl_ulist))
            self.next_slide()

            # Deep RL
            with fcolors.use_theme("dark"):
                deep_rl_title_text = TextBox(
                    title="Deep RL",
                    subtext=None,
                    title_font_size=36,
                    separator=False,
                    max_width=slide.containers.contrast_segment.width,
                ).move_to(slide.containers.contrast_segment.get_corner(UL), aligned_edge=UL)
                nn = NeuralNetwork([
                        FeedForwardLayer(3),
                        FeedForwardLayer(5),
                        FeedForwardLayer(3),
                    ],
                    layer_spacing=0.25,
                ).scale(1.4)
                nn.next_to(deep_rl_title_text, DOWN * 2, aligned_edge=LEFT)
                deep_rl_ulist = UnorderedList(
                    items=[
                        "Deep neural networks for state-action mapping",
                        "Output best action / distribution over actions",
                        "Can handle continuous state & action spaces",
                        "Used in nuclear fusion, autonomous driving, robotics, etc.",
                    ],
                )
                deep_rl_ulist.next_to(nn, DOWN * 2, aligned_edge=LEFT)
                nn_shift = slide.containers.contrast_segment.get_center()[0] - nn.get_center()[0]
                nn.shift(nn_shift * RIGHT)
            self.play(FadeOut(mab_title_text), FadeOut(mab_img), FadeOut(mab_ulist))
            self.play(Write(deep_rl_title_text), Create(nn, run_time=0.1), Write(deep_rl_ulist))
            forward_pass = nn.make_forward_pass_animation()
            self.play(forward_pass)
            self.next_slide()

            # Hierarchical RL
            with fcolors.use_theme("light"):
                hierarchical_rl_title_text = TextBox(
                    title="Hierarchical RL",
                    subtext=None,
                    title_font_size=36,
                    separator=False,
                    max_width=slide.containers.background_segment.width,
                ).move_to(slide.containers.background_segment.get_corner(UL), aligned_edge=UL)
                image_path = os.path.join(os.path.dirname(__file__), "hierarchical-rl.png")
                hierarchical_rl_img = ImageMobject(image_path).scale(0.65)
                hierarchical_rl_img.next_to(hierarchical_rl_title_text, DOWN * 2, aligned_edge=LEFT)
                hierarchical_rl_ulist = UnorderedList(
                    items=[
                        "Build a hierarchy of policies",
                        "Learn a policy for the coordinator, which learns a policy\nfor the sub-policies",
                        "Aims to provide higher levels of abstraction for\ndecision making",
                    ],
                )
                hierarchical_rl_ulist.next_to(hierarchical_rl_img, DOWN * 2, aligned_edge=LEFT)
                image_shift = slide.containers.background_segment.get_center()[0] - hierarchical_rl_img.get_center()[0]
                hierarchical_rl_img.shift(image_shift * RIGHT)
            self.play(FadeOut(tabular_rl_title_text), FadeOut(tabular_rl_ulist), FadeOut(frozenlake_img))
            self.play(Write(hierarchical_rl_title_text), FadeIn(hierarchical_rl_img), Write(hierarchical_rl_ulist))
            self.next_slide()

            # Multi-Agent RL
            with fcolors.use_theme("dark"):
                multi_agent_rl_title_text = TextBox(
                    title="Multi-Agent RL",
                    subtext=None,
                    title_font_size=36,
                    separator=False,
                    max_width=slide.containers.contrast_segment.width,
                ).move_to(slide.containers.contrast_segment.get_corner(UL), aligned_edge=UL)
                image_path = os.path.join(os.path.dirname(__file__), "multi-agent-rl.jpg")
                multi_agent_rl_img = ImageMobject(image_path).scale(0.25)
                multi_agent_rl_img.next_to(multi_agent_rl_title_text, DOWN * 2, aligned_edge=LEFT)
                multi_agent_rl_ulist = UnorderedList(
                    items=[
                        "Learn to maximize reward on an environment with\nmultiple agents",
                        "Each agent may have its own policy and reward function",
                        "Can learn to cooperate or compete with other agents",
                        "Holy grail of RL"
                    ],
                )
                multi_agent_rl_ulist.next_to(multi_agent_rl_img, DOWN * 2, aligned_edge=LEFT)
                image_shift = slide.containers.contrast_segment.get_center()[0] - multi_agent_rl_img.get_center()[0]
                multi_agent_rl_img.shift(image_shift * RIGHT)
            self.play(FadeOut(deep_rl_title_text), FadeOut(deep_rl_ulist), FadeOut(nn))
            self.play(Write(multi_agent_rl_title_text), FadeIn(multi_agent_rl_img), Write(multi_agent_rl_ulist))
            self.next_slide()


            self.play(
                FadeOut(hierarchical_rl_title_text),
                FadeOut(hierarchical_rl_ulist),
                FadeOut(hierarchical_rl_img),
                FadeOut(multi_agent_rl_title_text),
                FadeOut(multi_agent_rl_ulist),
                FadeOut(multi_agent_rl_img),
                FadeOut(slide),
            )

    def objectives_intro(self):
        with fcolors.use_theme("dark", self.camera):
            slide = SectionSlide(
                idx=3,
                title="Foundations Group Objectives",
            )
            self.play(Write(slide))
            self.next_slide()
            self.play(FadeOut(slide))

    def objectives(self):
        with fcolors.use_theme("light", self.camera):
            decor = VGroup(
                TwistingLines(
                    start_points=[(0.5, 3.5, 0), (2, 1.5, 0)],
                    end_points=[(3.5, 1, 0), (4.5, 2, 0)],
                    color=fcolors.current_theme.decor,
                    spacing=0.6
                ),
                TwistingLines(
                    start_points=[(3.5, 1, 0), (4.5, 2, 0)],
                    end_points=[(5.5, 3, 0), (6, 1.5, 0)],
                    color=fcolors.current_theme.decor,
                    spacing=0.6,
                ),
            )
            title_text = TextBox(
                title="Cohort Objectives",
                subtext=None,
                separator=False,
            ).to_edge(UL)
            learning_ulist = UnorderedList(
                subtitle="Learning",
                items=[
                    "Cover both theory and practice",
                    "Main learning source: Coursera",
                    "Exercises provided by Factored",
                    "Learning schedule & accountability",
                ]
            )
            projects_ulist = UnorderedList(
                subtitle="Project Development",
                items=[
                    "Two milestone projects: Tabular and Deep RL",
                    "Additional project in the works: LLMs & RL",
                    "Projects in the form of internal competitions"
                ]
            )
            competitions_ulist = UnorderedList(
                subtitle="External Competitions",
                items=[
                    "Explore open RL competitions",
                    "Submit interesting & novel solutions",
                    "Opportunity to publish research!",
                ]
            )
            ulists = VGroup(learning_ulist, projects_ulist, competitions_ulist)
            ulists.arrange(RIGHT, aligned_edge=UP)
            ulists.move_to(ORIGIN + DOWN)
            self.play(Write(decor), Write(title_text), Write(ulists))
            self.next_slide()
            self.play(FadeOut(decor), FadeOut(title_text), FadeOut(ulists))

    def logistics_intro(self):
        with fcolors.use_theme("dark", self.camera):
            slide = SectionSlide(
                idx=4,
                title="Cohort Logistics",
            )
            self.play(Write(slide))
            self.next_slide()
            self.play(FadeOut(slide))

    def logistics(self):
        with fcolors.use_theme("dark", self.camera):
            start_date = date(2026, 3, 9)
            schedule_rows = [
                (0, "Course 1", "Fundamentals of Reinforcement Learning", "Complete modules 1, 2, 3.", None),
                (14, "Course 1", "Fundamentals of Reinforcement Learning", "Complete modules 4, 5.", None),
                (28, "Course 2", "Sample-based Learning Methods", "Complete modules 1, 2, 3.", None),
                (42, "Course 2", "Sample-based Learning Methods", "Complete modules 4, 5.", None),
                (49, "Live Coding Session", "Q-Learning", "Implement Q-learning from scratch", fcolors.BLUE_BG),
                (63, "Project", "Sample-based Learning Methods", "Use Q-learning and SARSA algorithms in basic environments", fcolors.current_theme.accent),
                (77, "Course 3", "Prediction and Control with Function Approximation", "Complete modules 1, 2, 3.", None),
                (91, "Course 3", "Prediction and Control with Function Approximation", "Complete modules 4, 5.", None),
                (105, "Course 4", "A Complete Reinforcement Learning System (Capstone)", "Discuss the setup results (lunar landing environment)", None),
                (112, "Live Coding Session", "Deep Q-Learning", "Solve the CartPole environment using Deep Q-learning", fcolors.BLUE_BG),
                (119, "Course 4", "A Complete Reinforcement Learning Systeme", "Review the approach and the solutions", None),
                (133, "Live Coding Session", "Actor Critic Methods", "Solve the Pendulum environment using PPO", fcolors.BLUE_BG),
                (147, "Project", "Prediction and Control with Function Approximation", "Capstone Project with Function Approximation.", fcolors.current_theme.accent),
                (161, "RL & LLMs", "RLHF & Verifiable RL", "Run an LLM training pipeline with GRPO", fcolors.BLUE_BG),
                (175, "Project", "LLMs for SQL generation", "Fine-tune LLMs for text-to-SQL using GRPO", fcolors.current_theme.accent),
            ]
            table_data = [
                [(start_date + timedelta(days=days)).strftime("%d/%m/%Y"), activity, topic, objective]
                for days, activity, topic, objective, color in schedule_rows
            ]
            table = Table(
                table_data,
                col_labels=[Text("Date"), Text("Activity"), Text("Topic"), Text("Objective")],
                line_config={"stroke_width": 1, "color": fcolors.current_theme.secondary},
            )
            for i, (*_, row_color) in enumerate(schedule_rows):
                if row_color is not None:
                    for j in range(len(table_data[0])):
                        table.add_highlighted_cell((i + 1, j + 1), color=row_color)
            table.remove(*table.get_vertical_lines())
            table.scale(0.3).move_to(ORIGIN)
            self.play(Write(table))
            self.next_slide()
            self.play(FadeOut(table))

    def thankyou(self):
        with fcolors.use_theme("blue", self.camera):
            slide = WhatsNextSlide(
                title="Thank you!",
                subtext="Questions?",
            )
            self.play(Write(slide))
            self.next_slide()
            self.play(FadeOut(slide))

    def construct(self):
        self.next_slide()
        self.title_slide()
        self.agenda()
        self.overview_section()
        self.rl_setting()
        self.rl_example()
        self.mdp()
        self.types_intro()
        self.types_of_rl()
        self.objectives_intro()
        self.objectives()
        self.logistics_intro()
        self.logistics()
        self.thankyou()