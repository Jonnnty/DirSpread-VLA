import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import gemma
from openpi.models import pi0

RADIUS = 8
BLOCK = gemma.Block(configs=(gemma.get_config("gemma_2b"), gemma.get_config("gemma_300m")))


def kernel_for(width):
    axis = np.arange(-RADIUS, RADIUS + 1)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    if width < 0.05:
        kernel = np.zeros((axis.size, axis.size), np.float32)
        kernel[RADIUS, RADIUS] = 1.0
        return kernel
    kernel = np.exp(-(yy ** 2 + xx ** 2) / (2 * float(width) ** 2)).astype(np.float32)
    kernel /= kernel.sum()
    return kernel


def blur_grid(grid, kernel):
    channels = grid.shape[-1]
    x = jnp.transpose(grid, (0, 3, 1, 2))
    weight = jnp.tile(kernel[None, None, :, :], (channels, 1, 1, 1))
    y = jax.lax.conv_general_dilated(
        x,
        weight,
        window_strides=(1, 1),
        padding=[(RADIUS, RADIUS), (RADIUS, RADIUS)],
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
        feature_group_count=channels,
    )
    return jnp.transpose(y, (0, 2, 3, 1))


def spread_range(x, start, kernel):
    tok = x[:, start : start + 256, :].astype(jnp.float32)
    energy = jnp.linalg.norm(tok, axis=-1, keepdims=True)
    mixed = blur_grid(tok.reshape(tok.shape[0], 16, 16, tok.shape[-1]), kernel)
    mixed = mixed.reshape(tok.shape[0], 256, tok.shape[-1])
    mixed = mixed * (energy / jnp.maximum(jnp.linalg.norm(mixed, axis=-1, keepdims=True), 1e-6))
    return x.at[:, start : start + 256, :].set(mixed.astype(x.dtype))


def rmsnorm(x, scale):
    var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    normed = x * jnp.reciprocal(jnp.sqrt(var + 1e-6))
    return (normed * (1 + scale)).astype(x.dtype)


def attn_probs(x, layer, positions, attn):
    h = rmsnorm(x, layer["pre_attention_norm"]["scale"])
    q = jnp.einsum("BTD,NDH->BTNH", h, layer["attn"]["q_einsum"]["w"].astype(h.dtype))
    kv = jnp.einsum("BSD,2KDH->2BSKH", h, layer["attn"]["kv_einsum"]["w"].astype(h.dtype))
    k = kv[0]
    q = gemma._apply_rope(q, positions=positions)
    k = gemma._apply_rope(k, positions=positions)
    q = q * (q.shape[-1] ** -0.5)
    q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=1)
    logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)
    logits = jnp.where(attn[:, :, None, :, :], logits, jnp.float32(-2.3819763e38))
    return jax.nn.softmax(logits, axis=-1)


def expert0_params(layers, index):
    def take(tree):
        if isinstance(tree, dict):
            return {key: take(value) for key, value in tree.items() if not key.endswith("_1")}
        return tree[index]

    return take(layers)


def pure_params(model):
    import flax.traverse_util as traverse_util

    flat = traverse_util.flatten_dict(nnx.state(model).to_pure_dict())
    if flat and all(path[-1] == "value" for path in flat):
        flat = {path[:-1]: value for path, value in flat.items()}
    return traverse_util.unflatten_dict(flat)


def camera_share(peak, conc):
    peak = np.asarray(peak, np.float64)
    conc = np.asarray(conc, np.float64)
    by_peak = peak / max(float(peak.sum()), 1e-8)
    by_conc = conc / max(float(conc.sum()), 1e-8)
    return 0.5 * (by_peak + by_conc)


def widths_for(share):
    share = np.asarray(share, np.float64)
    raw = np.clip(0.5 - 2.5 * (share - 0.5), 0.0, 1.0)
    return float(raw[0]), float(raw[1])


def word_indexes(token_ids, sp):
    bowl, plate = [], []
    for index, token_id in enumerate(np.asarray(token_ids).tolist()):
        if int(token_id) == 0:
            continue
        text = sp.id_to_piece(int(token_id)).replace("▁", " ").lower()
        if "bowl" in text:
            bowl.append(768 + index)
        if "plate" in text:
            plate.append(768 + index)
    if not bowl or not plate:
        raise RuntimeError(f"missing bowl/plate tokens bowl={bowl} plate={plate}")
    return bowl, [plate[-1]]


def shape_on(probs, queries):
    table = np.asarray(probs[0, 0].mean(axis=0))
    peaks, concs = [], []
    for sl in (slice(0, 256), slice(256, 512)):
        peak_acc = conc_acc = 0.0
        for query in queries:
            row = np.asarray(table[int(query), sl], np.float64)
            mass = float(row.sum())
            peak_acc += float(row.max())
            conc_acc += float(np.sort(row)[-5:].sum()) / max(mass, 1e-8)
        n = len(queries)
        peaks.append(peak_acc / n)
        concs.append(conc_acc / n)
    return peaks, concs


@jax.jit
def layer_step(x, layer, positions, attn):
    probs = attn_probs(x, layer, positions, attn)
    xs, _ = BLOCK.apply({"params": layer}, [x, None], None, positions, attn, [None, None], True)
    return xs[0], probs


def read_words(tokens, attn_mask, positions, stacked, bowl, plate):
    x = tokens
    bowl_peak, bowl_conc, plate_peak, plate_conc = [], [], [], []
    for i in range(18):
        layer = expert0_params(stacked, i)
        x, probs = layer_step(x, layer, positions, attn_mask)
        peak, conc = shape_on(probs, bowl)
        bowl_peak.append(peak)
        bowl_conc.append(conc)
        peak, conc = shape_on(probs, plate)
        plate_peak.append(peak)
        plate_conc.append(conc)
    x.block_until_ready()
    return {
        "bowl_peak": np.mean(bowl_peak, axis=0),
        "bowl_conc": np.mean(bowl_conc, axis=0),
        "plate_peak": np.mean(plate_peak, axis=0),
        "plate_conc": np.mean(plate_conc, axis=0),
    }


def make_runner(model):
    graphdef, state = nnx.split(model)

    def run(state, obs, noise, stacked, k_fixed, k_wrist):
        module = nnx.merge(graphdef, state)
        tokens, mask, ar = module.embed_prefix(obs)
        attn_mask = pi0.make_attn_mask(mask, ar)[:, None, :, :]
        positions = jnp.cumsum(mask, axis=1) - 1

        def body(x, params):
            x = spread_range(x, 0, k_fixed)
            x = spread_range(x, 256, k_wrist)
            xs, kv = BLOCK.apply({"params": params}, [x, None], None, positions, attn_mask, [None, None], True)
            return xs[0], (kv[0], kv[1])

        _, (keys, values) = jax.lax.scan(body, tokens.astype(jnp.bfloat16), stacked)
        cache = (keys, values)
        num_steps = 10
        dt = -1.0 / num_steps
        batch = obs.state.shape[0]

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar, adarms = module.embed_suffix(obs, x_t, jnp.broadcast_to(time, batch))
            suffix_attn = pi0.make_attn_mask(suffix_mask, suffix_ar)
            prefix_attn = einops.repeat(mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full = jnp.concatenate([prefix_attn, suffix_attn], axis=-1)
            pos = jnp.sum(mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (_, suffix_out), _ = module.PaliGemma.llm(
                [None, suffix_tokens], mask=full, positions=pos, kv_cache=cache, adarms_cond=[None, adarms]
            )
            v_t = module.action_out_proj(suffix_out[:, -module.action_horizon :])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, jnp.asarray(1.0, dtype=jnp.float32)))
        return x_0

    return jax.jit(run), state
