from manim import *
import numpy as np
from coin import Coin


class MarkovExample(Scene):

    def flip(self, coin: Coin, num_semiflips: int = 1, run_time: float = 1):
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

    def construct(self):
        states_math_str = "S = \\{ s_1, s_2, \\ldots, s_n \\}"
        actions_math_str = "A = \\{ a_1, a_2, \\ldots, a_m \\}"
        transition_math_str = "P(s_{t+1} | s_t, a_t)"
        reward_math_str = "R(s_t, a_t, s_{t+1})"
        states_text = MathTex("\\text{States: } ", states_math_str).to_edge(UL)
        actions_text = MathTex("\\text{Actions: } ", actions_math_str).next_to(states_text, DOWN, aligned_edge=LEFT)
        transition_text = MathTex("\\text{Transition Function: } ", transition_math_str).next_to(actions_text, DOWN, aligned_edge=LEFT)
        reward_text = MathTex("\\text{Reward Function: } ", reward_math_str).next_to(transition_text, DOWN, aligned_edge=LEFT)

        explanation_text_group = VGroup(states_text, actions_text, transition_text, reward_text)

        states_example_text = MathTex("S = \\{ ", "Cool", ", ", "Hot", ", ", "Overheat", " \\}").to_edge(UR)
        states_example_text.set_color_by_tex("Cool", GREEN)
        states_example_text.set_color_by_tex("Hot", ORANGE)
        states_example_text.set_color_by_tex("Overheat", GRAY)

        actions_example_text = MathTex("A = \\{ ", "Fast", ", ", "Slow", " \\}").next_to(states_example_text, DOWN, aligned_edge=RIGHT)
        actions_example_text.set_color_by_tex("Fast", RED)
        actions_example_text.set_color_by_tex("Slow", BLUE)


        cool_state = Circle(color=GREEN, radius=0.7, fill_opacity=0.0).move_to([-1, -2, 0])
        cool_text = Text("Cool", font_size=20, color=GREEN).move_to(cool_state.get_center())
        cool_group = VGroup(cool_state, cool_text)
        cool_group.og_color = GREEN

        hot_state = Circle(color=ORANGE, radius=0.7, fill_opacity=0.0).move_to([3, 2, 0])
        hot_text = Text("Hot", font_size=20, color=ORANGE).move_to(hot_state.get_center())
        hot_group = VGroup(hot_state, hot_text)
        hot_group.og_color = ORANGE

        overheat_state = Circle(color=GRAY, radius=0.7, fill_opacity=0.0).move_to([7, 0, 0])
        overheat_text = Text("Overheat", font_size=20, color=GRAY).move_to(overheat_state.get_center())
        overheat_group = VGroup(overheat_state, overheat_text)
        overheat_group.og_color = GRAY

        states_group = VGroup(cool_group, hot_group, overheat_group)
        
        # Cool transitions
        cfh_arrow = CurvedArrow(
            start_point=cool_state.get_top(),
            end_point=hot_state.get_left(),
            radius=-7,
            color=RED
        )
        cfh_arrow.starting_point = cool_state.get_top()
        cfh_arrow.end_point = hot_state.get_left()
        cfh_prob = MathTex("0.5", font_size=28, color=WHITE).move_to(cfh_arrow.get_center())
        cfh_reward = MathTex("+2", font_size=28, color=GREEN).move_to(cfh_arrow.get_center() + UL * 0.5)
        cfh_group = VGroup(cfh_arrow, cfh_prob, cfh_reward)

        cfc_arrow = Arc(
            radius=0.4,
            start_angle=0*DEGREES,
            angle=235*DEGREES,
            color=RED
        ).move_to(cool_state.get_corner(UL) + [0.3, 0.05, 0]).add_tip()
        cfc_arrow.starting_point = cool_state.get_top()
        cfc_arrow.end_point = cool_state.point_at_angle(135*DEGREES)
        cfc_prob = MathTex("0.5", font_size=28, color=WHITE).move_to(cfc_arrow.get_center())
        cfc_reward = MathTex("+2", font_size=28, color=GREEN).move_to(cfc_arrow.get_center() + UP * 0.8)
        cfc_group = VGroup(cfc_arrow, cfc_prob, cfc_reward)

        csc_arrow = Arc(
            radius=0.4,
            start_angle=90*DEGREES,
            angle=235*DEGREES,
            color=BLUE
        ).move_to(cool_state.get_corner(DL) + [-0.05, 0.3, 0]).add_tip()
        csc_arrow.starting_point = cool_state.get_left()
        csc_arrow.end_point = cool_state.point_at_angle(225*DEGREES)

        csc_prob = MathTex("1.0", font_size=28, color=WHITE).move_to(csc_arrow.get_center())
        csc_reward = MathTex("+1", font_size=28, color=GREEN).move_to(csc_arrow.get_center() + LEFT * 0.8)
        csc_group = VGroup(csc_arrow, csc_prob, csc_reward)

        cf_group = VGroup(cfc_group, cfh_group)

        cold_transitions_group = VGroup(cf_group, csc_group)
        
        # Hot transitions
        hsc_arrow = CurvedArrow(
            start_point=hot_state.get_bottom(),
            end_point=cool_state.get_right(),
            radius=-7,
            color=BLUE
        )
        hsc_arrow.starting_point = hot_state.get_bottom()
        hsc_arrow.end_point = cool_state.get_right()
        hsc_prob = MathTex("0.5", font_size=28, color=WHITE).move_to(hsc_arrow.get_center())
        hsc_reward = MathTex("+1", font_size=28, color=GREEN).move_to(hsc_arrow.get_center() + DR * 0.5)
        hsc_group = VGroup(hsc_arrow, hsc_prob, hsc_reward)
        
        hfo_arrow = CurvedArrow(
            start_point=hot_state.get_right(),
            end_point=overheat_state.point_at_angle(135*DEGREES),
            radius=-7,
            color=RED
        )
        hfo_arrow.starting_point = hot_state.get_right()
        hfo_arrow.end_point = overheat_state.point_at_angle(135*DEGREES)
        hfo_prob = MathTex("1.0", font_size=28, color=WHITE).move_to(hfo_arrow.get_center())
        hfo_reward = MathTex("-10", font_size=28, color=RED).move_to(hfo_arrow.get_center() + UP * 0.5)
        hfo_group = VGroup(hfo_arrow, hfo_prob, hfo_reward)
        
        hsh_arrow = Arc(
            radius=0.4,
            start_angle=180*DEGREES,
            angle=235*DEGREES,
            color=BLUE
        ).move_to(hot_state.get_corner(DR) + [-0.3, -0.05, 0]).add_tip()
        hsh_arrow.starting_point = hot_state.get_bottom()
        hsh_arrow.end_point = hot_state.point_at_angle(315*DEGREES)
        hsh_prob = MathTex("0.5", font_size=28, color=WHITE).move_to(hsh_arrow.get_center())
        hsh_reward = MathTex("+1", font_size=28, color=GREEN).move_to(hsh_arrow.get_center() + DOWN * 0.8)
        hsh_group = VGroup(hsh_arrow, hsh_prob, hsh_reward)

        hs_group = VGroup(hsc_group, hsh_group)

        hot_transitions_group = VGroup(hs_group, hfo_group)

        mdp_group = VGroup(states_group, cold_transitions_group, hot_transitions_group)

        
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

        rl_framework_group = VGroup(agent_group, env_group, arrows_group).to_edge(LEFT)

        # Robot
        robot = VGroup(
            Dot(color=WHITE, radius=0.1).shift(LEFT * 0.5),
            Dot(color=WHITE, radius=0.1).shift(RIGHT * 0.5),
            Dot(color=WHITE, radius=0.2).shift(UP * 1.2),
            RoundedRectangle(width=2, height=1, corner_radius=0.5, color=WHITE, stroke_width=4),
            Line(start=UP * 0.5, end=UP * 1.2, color=WHITE, stroke_width=4),
        ).scale(0.8)

        coin = Coin(radius=0.5, height=0.1)
        coin.next_to(robot, RIGHT)

        random_agent_group = VGroup(robot, coin)
        random_agent_group.move_to(agent_group.get_center()).shift(UP)

        # Animations
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
        self.next_section()

        self.play(Write(actions_text))
        self.play(Write(actions_example_text))
        self.next_section()

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
        self.next_section()

        self.play(Write(reward_text))
        self.play(
            Write(cfh_reward),
            Write(cfc_reward),
            Write(csc_reward),
            Write(hsc_reward),
            Write(hfo_reward),
            Write(hsh_reward),
        )
        self.next_section()

        # Deterministic transitions
        self.play(
            Indicate(csc_group),
            Indicate(hfo_group),
            run_time=2,
        )
        self.next_section()

        # Probabilistic transitions
        self.play(
            Indicate(cf_group),
            Indicate(hs_group),
            run_time=2,
        )
        self.next_section()

        # Terminal States
        self.play(
            Indicate(overheat_group),
            run_time=2,
        )
        self.next_section()

        self.play(
            FadeOut(explanation_text_group, shift=LEFT * 5),
            mdp_group.animate.shift(RIGHT).scale(0.9)
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
        self.next_section()

        # SIMULATE A GAME WITH RANDOM AGENT

        np.random.seed(42)
        current_state = cool_group

        self.play(rl_framework_group.animate.shift(DOWN))
        self.play(
            Create(robot),
            Create(coin),
            current_state.animate.set_color(YELLOW),
        )
        self.wait(1)

        while current_state != overheat_group:
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

            transition_dot = Dot(color=YELLOW, radius=0.1).move_to(transition_arrow.point_from_proportion(0.0))
            state_clone = old_state.copy()
            self.add(state_clone)
            old_state.set_color(old_state.og_color)

            self.flip(coin, num_semiflips=num_semiflips, run_time=1)
            self.play(
                Transform(state_clone[0], transition_dot),
                state_clone[1].animate.set_color(old_state.og_color),
                run_time=0.5,
                rate_func=rate_functions.ease_in_sine
            )
            self.remove(state_clone)
            old_state[1].set_color(old_state.og_color)
            self.play(
                MoveAlongPath(transition_dot, transition_arrow),
                run_time=0.5,
                rate_func=linear,
            )
            state_clone = current_state.copy()
            state_clone.set_color(YELLOW)
            self.play(
                Transform(transition_dot, state_clone[0]),
                current_state[1].animate.set_color(YELLOW),
                run_time=0.5,
                rate_func=rate_functions.ease_out_sine
            )
            current_state.set_color(YELLOW)
            self.remove(state_clone)
            self.remove(transition_dot)


        self.next_section()