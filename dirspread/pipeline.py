import jax
import jax.numpy as jnp
import sentencepiece
from libero.libero import benchmark
from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as train_config

from dirspread import rollout
from dirspread import scene
from dirspread import spread
from dirspread import video


def load(checkpoint, tokenizer):
    print("loading", checkpoint, flush=True)
    policy = policy_config.create_trained_policy(train_config.get_config("pi05_libero"), checkpoint)
    tok_path = download.maybe_download(tokenizer, gs={"token": "anon"})
    sp = sentencepiece.SentencePieceProcessor(model_proto=open(tok_path, "rb").read())
    model = policy._model
    layers = jax.tree.map(lambda w: jnp.asarray(w), spread.pure_params(model)["PaliGemma"]["llm"]["layers"])
    stacked = spread.expert0_params(layers, slice(None))
    run, state = spread.make_runner(model)
    return policy, model, run, state, stacked, sp


def run_tasks(checkpoint, tokenizer, tasks, out_dir, frames, skip_existing):
    out_dir.mkdir(parents=True, exist_ok=True)
    policy, model, run, state, stacked, sp = load(checkpoint, tokenizer)
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    for task in tasks:
        path = out_dir / f"compare_seg{task['id']:02d}.mp4"
        if skip_existing and path.exists():
            print("skip", path, flush=True)
            continue
        camera = task["camera"]
        delta = None if task["delta"] is None else task["delta"]
        print(f"=== seg {task['id']} task {task['task_id']} init {task['init_idx']} {task['note']} ===", flush=True)
        prompt = scene_prompt(suite, task["task_id"], task["init_idx"], camera, delta)
        orig_f, orig_w, orig_places, orig_done = rollout.original(
            policy, suite, task["task_id"], task["init_idx"], camera, delta, prompt, frames
        )
        photos_f, photos_w, places, phases, mains, done_at = rollout.rollout(
            policy,
            model,
            run,
            state,
            stacked,
            sp,
            suite,
            task["task_id"],
            task["init_idx"],
            camera,
            delta,
            prompt,
            frames,
        )
        video.write_video(
            path,
            task["id"],
            task["phrase"],
            task["note"],
            frames,
            orig_f,
            orig_w,
            orig_places,
            photos_f,
            photos_w,
            places,
            phases,
            mains,
        )
        print("saved", path, "orig", orig_done, "speed", done_at, flush=True)


def scene_prompt(suite, task_id, init_idx, camera, delta):
    env, _obs, prompt = scene.open_env(suite, task_id, init_idx, camera, delta)
    env.close()
    print("TASK", prompt, flush=True)
    return prompt
