import collections
import time

import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import model as openpi_model
from openpi.models import pi0

from dirspread import scene
from dirspread import spread

DT = 1.0 / 20.0
SPEED_SWITCH = 0.08


def prepare(policy, image, wrist, state, prompt):
    raw = {
        "observation/image": image,
        "observation/wrist_image": wrist,
        "observation/state": np.asarray(state, np.float32),
        "prompt": prompt,
    }
    inputs = policy._input_transform(jax.tree.map(lambda item: item, raw))
    batched = jax.tree.map(lambda item: jnp.asarray(item)[None, ...], inputs)
    return openpi_model.Observation.from_dict(batched), np.asarray(batched["state"][0])


def state_vec(obs):
    return np.concatenate(
        (obs["robot0_eef_pos"], scene.quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
    )


def original(policy, suite, task_id, init_idx, camera, delta, prompt, limit):
    env, obs, _prompt = scene.open_env(suite, task_id, init_idx, camera, delta)
    policy._rng = jax.random.key(0)
    plan = collections.deque()
    photos_f, photos_w, places = [], [], []
    done_at = None
    step = 0
    frames = 0
    t0 = time.perf_counter()
    try:
        while frames < limit:
            if step < scene.WAIT:
                obs, _, _, _ = env.step(scene.DUMMY)
                step += 1
                continue
            image = scene.photo(obs, "agentview_image")
            hand = scene.photo(obs, "robot0_eye_in_hand_image")
            photos_f.append(image)
            photos_w.append(hand)
            places.append(scene.measure(env))
            if not plan:
                out = policy.infer(
                    {
                        "observation/image": image,
                        "observation/wrist_image": hand,
                        "observation/state": state_vec(obs),
                        "prompt": prompt,
                    }
                )
                plan.extend(out["actions"][: scene.REPLAN])
            frames += 1
            obs, _, done, _ = env.step(plan.popleft().tolist())
            if done and done_at is None:
                done_at = frames
                print("orig checker", done_at, flush=True)
            step += 1
    finally:
        env.close()
    print(f"orig wall {time.perf_counter() - t0:.1f}s frames {frames} done {done_at}", flush=True)
    return photos_f, photos_w, places, done_at


def update_phase(phase, streak, speed_change, plate_steps):
    switched = ""
    if phase == "看碗" and streak >= 2 and plate_steps == 0:
        phase = "看盘"
        switched = "grasp"
    elif phase == "看盘" and plate_steps >= 10 and speed_change is not None and speed_change >= SPEED_SWITCH:
        phase = "看碗"
        switched = f"speed {speed_change:.3f}"
    return phase, switched


def rollout(policy, model, run, state, stacked, sp, suite, task_id, init_idx, camera, delta, prompt, limit):
    env, obs, _prompt = scene.open_env(suite, task_id, init_idx, camera, delta)
    policy._rng = jax.random.key(0)
    plan = collections.deque()
    photos_f, photos_w, places, phases, mains = [], [], [], [], []
    done_at = None
    phase = "看碗"
    main_view = "固定机位"
    streak = 0
    last_query_eef = None
    prev_eef = None
    speeds = []
    plate_steps = 0
    latched_bowl = False
    step = 0
    frames = 0
    t0 = time.perf_counter()
    try:
        while frames < limit:
            if step < scene.WAIT:
                obs, _, _, _ = env.step(scene.DUMMY)
                step += 1
                continue
            image = scene.photo(obs, "agentview_image")
            hand = scene.photo(obs, "robot0_eye_in_hand_image")
            grip = np.asarray(obs["robot0_gripper_qpos"], np.float64)
            aperture = float(abs(grip[0] - grip[1])) if grip.size > 1 else float(abs(grip[0]))
            eef = np.asarray(obs["robot0_eef_pos"], np.float64)
            if prev_eef is not None:
                speeds.append(float(np.linalg.norm(eef - prev_eef) / DT))
            prev_eef = eef
            moved = 0.0 if last_query_eef is None else float(np.linalg.norm(eef - last_query_eef))
            if not plan:
                speed_change = None
                if len(speeds) >= 10:
                    speed_change = abs(float(np.mean(speeds[-5:]) - np.mean(speeds[-10:-5])))
                prepared, state_out = prepare(policy, image, hand, state_vec(obs), prompt)
                bowl, plate = spread.word_indexes(prepared.tokenized_prompt[0], sp)
                prepared = openpi_model.preprocess_observation(None, prepared, train=False)
                tokens, mask, ar = model.embed_prefix(prepared)
                tokens = tokens.astype(jnp.bfloat16)
                attn_mask = pi0.make_attn_mask(mask, ar)[:, None, :, :]
                positions = jnp.cumsum(mask, axis=1) - 1
                words = spread.read_words(tokens, attn_mask, positions, stacked, bowl, plate)
                bowl_share = spread.camera_share(words["bowl_peak"], words["bowl_conc"])
                plate_share = spread.camera_share(words["plate_peak"], words["plate_conc"])
                grasped = bowl_share[1] > bowl_share[0] and aperture < 0.02 and moved > 0.03
                entering = phase == "看碗" and not latched_bowl
                if entering and grasped:
                    streak += 1
                elif entering:
                    streak = 0
                phase, switched = update_phase(phase, streak, speed_change, plate_steps)
                if switched:
                    print(f"speed switch {switched} at step {frames + 1}", flush=True)
                    if switched.startswith("speed"):
                        latched_bowl = True
                chosen = bowl_share if phase == "看碗" else plate_share
                main_view = "固定机位" if float(chosen[0]) >= float(chosen[1]) else "抓夹机位"
                w_fixed, w_wrist = spread.widths_for(chosen)
                policy._rng, rng = jax.random.split(policy._rng)
                noise = jax.random.normal(rng, (prepared.state.shape[0], model.action_horizon, model.action_dim))
                actions = np.asarray(
                    run(
                        state,
                        prepared,
                        noise,
                        stacked,
                        jnp.asarray(spread.kernel_for(w_fixed)),
                        jnp.asarray(spread.kernel_for(w_wrist)),
                    )[0]
                )
                last_query_eef = eef
                out = policy._output_transform({"state": state_out, "actions": actions})
                plan.extend(np.asarray(out["actions"])[: scene.REPLAN])
                delta_txt = f"{speed_change:.3f}" if speed_change is not None else "-"
                print(
                    f"speed step {frames + 1:03d} {phase} share {chosen[0]:.2f}/{chosen[1]:.2f} "
                    f"spread {w_fixed:.2f}/{w_wrist:.2f} main {main_view} "
                    f"move {moved * 100:.1f}cm aperture {aperture:.3f} dv {delta_txt}",
                    flush=True,
                )
            if phase == "看盘":
                plate_steps += 1
            photos_f.append(image)
            photos_w.append(hand)
            places.append(scene.measure(env))
            phases.append(phase)
            mains.append(main_view)
            frames += 1
            obs, _, done, _ = env.step(plan.popleft().tolist())
            if done and done_at is None:
                done_at = frames
                print("speed checker", done_at, flush=True)
            step += 1
    finally:
        env.close()
    print(f"speed wall {time.perf_counter() - t0:.1f}s frames {frames} done {done_at}", flush=True)
    return photos_f, photos_w, places, phases, mains, done_at
