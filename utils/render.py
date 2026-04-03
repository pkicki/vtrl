import jax
import jax.numpy as jnp
import imageio
import os

def record_final_evaluation_video(
    make_inference_fn,
    params,
    render_env,
    episode_length,
    action_repeat,
    seed,
    output_path,
    render_every=16,
    camera="side",
):

    policy = make_inference_fn(params, deterministic=True)
    policy = jax.jit(policy)

    rollout_key = jax.random.PRNGKey(seed + 10_000)
    act_key = jax.random.PRNGKey(seed + 20_000)

    state = render_env.reset(rollout_key)

    total_steps = episode_length
    num_frames = total_steps // render_every

    def step_fn(carry, t):
        state, key = carry

        key, key_sample = jax.random.split(key)
        action, _ = policy(state.obs, key_sample)

        # apply action_repeat
        def repeat_step(s, _):
            s = render_env.step(s, action)
            return s, None

        state, _ = jax.lax.scan(repeat_step, state, None, length=action_repeat)

        keep = (t % render_every) == 0

        return (state, key), (state, keep)

    (final_state, _), (states, mask) = jax.lax.scan(
        step_fn,
        (state, act_key),
        jnp.arange(total_steps // action_repeat),
    )

    states_subsampled = jax.tree_util.tree_map(
        lambda x: x[mask], states
    )

    # render() expects List[State]; convert batched state → list of individual states
    states_np = jax.device_get(states_subsampled)
    num_render_frames = jax.tree_util.tree_leaves(states_np)[0].shape[0]
    states_list = [
        jax.tree_util.tree_map(lambda x, i=i: x[i], states_np)
        for i in range(num_render_frames)
    ]

    D = 1
    frames = render_env.render(
        states_list,
        height=int(480 / D),
        width=int(640 / D),
        camera=camera,
    )

    # --- write video ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fps = int(1.0 / render_env.dt / render_every)

    writer = imageio.get_writer(output_path, fps=fps)

    for frame in frames:
        writer.append_data(frame)

    writer.close()

    return output_path