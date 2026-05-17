# pandabomber-ai

Self-play RL training for PandaBomber bots, plus a model loader for the gameserver.

## Architecture

| File | Purpose |
|---|---|
| `src/sim.ts` | Pure-TS port of `backend-gameserver/src/game.ts` (deterministic, MU-only, time-stepped at 10ms — matches production's 100Hz tickRate exactly). |
| `src/observation.ts` | Encodes game state → `Float32Array` (multi-channel grid + scalar features). Shared between trainer and bot2. |
| `src/env.ts` | Wraps `Sim` as an RL environment (one obs per seat, action committed at cell-aligned ticks). |
| `src/model.ts` | TF.js CNN (Conv 3×3/32 → Conv 3×3/32 → Dense 64 → 6) + shape-aware JSON weight export so inference needs no TF.js. |
| `src/train.ts` | Self-play Double-DQN with replay buffer, target net, ε-greedy, and a frozen-opponent pool. |
| `src/parity_test.ts` | Confirms bot2's pure-JS forward pass matches tfjs's `predict()` to <1e-6 on shared weights. |
| `../backend-gameserver/src/bot2.ts` | Drop-in bot that loads the JSON weights and runs a pure-JS conv+dense forward pass. |

## Network

Small CNN tuned to be cheap enough that bot2.ts can run the forward pass in pure JS:

```
spatial input: [13, 19, 11]       scalars: [6]
        │                              │
Conv 3×3 same, 32 filters, ReLU        │
        │                              │
Conv 3×3 same, 32 filters, ReLU        │
        │                              │
     Flatten ────────── Concat ────────┘
                          │
                     Dense 64, ReLU
                          │
                     Dense 6, linear (Q)
```

Conv layers see the 13×19 grid directly, which is far more sample-efficient than the
previous flat MLP (translation-invariant features come for free). The flat
observation buffer is laid out in NHWC order so `tf.tensor4d(flat, [N, H, W, C])` and
bot2's `conv2dRelu()` can both consume it without a transpose.

## Inputs (observation)

For each agent, egocentric view:

**Spatial channels** (11 × 13 × 19 = 2717 floats):
- 0: stone walls
- 1: wood walls
- 2: bomb presence
- 3: bomb fuse remaining (0–1)
- 4: time-to-blast per cell (1.0 = exploding now, decays to 0 at 1500ms lookahead)
- 5–7: powerups (NUM / SPE / STR)
- 8: self position
- 9: enemy positions (alive, not knocked)
- 10: knocked enemies (vulnerable — easy kill)

**Scalar features** (6 floats): bombPower, maxBombs, moveSpeed tier, placedBombs ratio, woodLeft / 100, game-phase.

## Action space

6 discrete actions, decided per cell-arrival: `stay`, `up`, `down`, `left`, `right`, `bomb`.

## Reward function

| Event | Reward |
|---|---|
| Destroy wood block | +1.0 |
| Collect powerup | +1.5 |
| Knock an enemy | +5.0 |
| Kill an enemy | +10.0 |
| Get knocked | −10.0 |
| Die | −30.0 |
| Win the game | +20.0 |
| Per simulation step | −0.005 |

Wood and powerup rewards bootstrap the early-game block-clearing phase; knock/kill rewards drive the combat phase; survival pressure shapes endgame play.

## Usage

```bash
cd pandabomber-ai
npm install
# quick smoke test (few episodes, no GPU needed):
npx ts-node src/train.ts --episodes=20 --batchSize=64 --saveEvery=5
# full training run:
npx ts-node src/train.ts --episodes=5000 --save=checkpoints/latest.json
# GPU run (CUDA 11.8 + cuDNN 8.6 required):
TFJS_GPU=1 npx ts-node src/train.ts --episodes=5000 --save=checkpoints/latest.json
# or the bundled shortcut:
npm run train:gpu -- --episodes=5000
# benchmark the per-tick hot path (sim + forward + learn) on CPU or GPU:
npm run profile
npm run profile:gpu
# type-check the whole project (no emit; nothing consumes dist/):
npm run typecheck
```

### GPU notes

`@tensorflow/tfjs-node-gpu` is listed as an `optionalDependency` so `npm install`
won't fail on machines without CUDA. When `TFJS_GPU=1` is set, `src/tfBackend.ts`
hijacks Node's module cache so every `import * as tf from '@tensorflow/tfjs-node'`
in this project actually resolves to the GPU package — no other code change needed.
If the GPU package fails to load (missing CUDA libs etc.) the backend logs a warning
and silently falls back to CPU; check stdout for `[tf] using GPU backend`.

Inference on the gameserver (`bot2.ts`) is pure-JS and does not depend on tfjs at
all, so the gameserver hosts don't need CUDA — only the training/eval machines do.

### Per-tick inference batching

Cells take ~120–220ms to traverse at 10ms/tick, so on the vast majority of ticks
fewer than 4 seats are cell-aligned. The training loop only runs a forward pass
for seats that actually need a fresh decision (cell-aligned + alive + !knocked):
one batch-of-≤1 for the learner and one batch-of-≤3 for the frozen opponents.
Saves the majority of forwards vs. always doing batch-of-1 + batch-of-3 per tick.

All training is 4-player MU self-play from episode 0 — there's no solo-warmup curriculum
because the bot is never used in time-trial games.

Checkpoints are written to `checkpoints/latest.json` as plain JSON.

## Opponent pool

To kill the classic self-play failure mode (online policy drifting against its own
moving target), seats 1–3 don't play the live online network. Instead, every episode:

1. Freeze a copy of the opponent weights at episode start. With probability `pPool`
   (default 0.5) sample from a rolling pool of `poolSize` (default 10) historical
   snapshots; otherwise mirror the current online weights.
2. Seat 0 (learner) plays `online` with ε-greedy exploration.
3. Seats 1–3 share the frozen `opponent` policy, played greedy.
4. All four seats record transitions to the shared replay buffer (DQN is off-policy,
   so frozen-policy data is valid).
5. Every `snapshotEvery` episodes (default 25), push the current online weights into
   the pool, evicting the oldest if over `poolSize`.

Knobs: `--pPool=0.5 --poolSize=10 --snapshotEvery=25`.

**Warm-start / resuming:** if `--save` already points at an existing checkpoint, the
trainer loads those weights and resumes — the saved `globalStep` and `episode` counters
are restored so ε-decay continues from the right place. Override with `--resume=off`
to start fresh and overwrite, or `--resume=on` to require a checkpoint (errors if none
found).

Adam optimizer momentum / variance state is NOT persisted; it warms back up in a few
hundred gradient steps. Replay buffer also doesn't persist — it refills as you train.

## Evaluating a checkpoint

`src/eval.ts` plays deterministic games (ε=0, fixed seeds) between the trained model and
a configurable set of opponents, then reports win rate + behavioral metrics:

```bash
npx ts-node src/eval.ts --model=checkpoints/latest.json --vs=noop,random --games=100
# Compare against an older snapshot to catch regressions:
npx ts-node src/eval.ts --model=checkpoints/latest.json \
    --vs=noop,random,model:checkpoints/ep2000.json --games=100
```

Seats are swapped every other game so corner-spawn / first-mover bias cancels. Each
matchup also reports:
- `avg_dur_s` — average game length in seconds (long games against weak opponents = bot wasted time)
- `avg_wood`, `avg_pups`, `avg_kills` — what the primary actually accomplished per game
- `suicide%` — fraction of games where the primary died from its own bomb

Pass `--json=path` to dump the full result for charting over training time.

Reasonable benchmarks once training stabilizes: >95% vs `noop`, >80% vs `random`. Lower
than that and either training hasn't converged or there's a bug in the reward function.

## Deploying the trained model

Build and restart the gameserver. To swap bot.ts for bot2.ts as the spawned process,
edit `backend-gameserver/src/botManager.ts` and change `BOT_ENTRY` from `./bot.js` to
`./bot2.js`. (Or set `BOT_MODEL_PATH` in env for a custom checkpoint location.)

Before deploying, run `npm run test:parity` — this verifies bot2's pure-JS conv+dense
forward pass matches tfjs's `predict()` on shared weights to within 1e-6. If parity
drifts (e.g. someone reorders an obs channel or tweaks the conv layout), the trained
model would silently produce garbage in production; the test catches that.

## Simulator fidelity

**Exactly modeled** (verified by `src/sim_test.ts`):
- Bomb fuse (3000ms), invulnerability (200ms), knock duration (6000ms), 3-strikes-same-cell ⇒ outright death
- 25ms-per-cell explosion travel; per-cell death window = `[arrival, arrival+50ms]` (matches production's two `checkPlayerDeaths` calls per cell)
- Chain detonation: ray hitting a bomb chains it synchronously, ray PASSES THROUGH the bomb's cell. Each chained bomb's ray gets its own timing
- Wood destroyed when the ray actually arrives (not at fuse-expiry); powerup spawn rolls happen at that moment
- Powerup caps (13 per type), regular spawn odds, 4-powerup death drop in a 5×5 area
- Corner spawn clearing (3 cells per corner are wood-free)

**Movement passability** uses `Math.round(targetCell)` — the same check the authoritative server validation uses (`backend-gameserver/src/gameplayer.ts:48-60 updatePos`). The frontend's `ceil/floor` + `bombWalkingTolerance` + `movementOffsetTolerance` logic is client-side prediction for smooth rendering of *human* input; it doesn't affect what the server accepts. Since the agent decides at cell-aligned positions and moves to adjacent integer cells, sim and server agree on every passability check.

**Kill checks** fire exactly twice per cell — once at the first sim tick where `now >= arrivalMs`, once at the first tick where `now >= arrivalMs + 50ms`. After that the cell is permanently safe (matches production: the two `checkPlayerDeaths` calls per cell in `executeExplosions` / `recurseExecute`, no more pending timers afterward).

**Tick rate**: `SIM_DT_MS = 10ms`, matching production's `tickRate = 100` in `game.ts`. Each "first tick past T" event can be at most 10ms late vs production's exact `setTimeout` firing — well under the 50ms kill window and the 25ms-per-cell ray travel, so timing-sensitive dodge skills should transfer cleanly to production.

Run the smoke tests: `npx ts-node src/sim_test.ts`

## Limitations / TODOs

- Training is single-process / single-thread. Naive `worker_threads`-based parallel envs would give only ~1.2–1.3× wall-clock speedup — the GPU learn step (~75ms at batch=128) dominates per-tick cost, so parallelizing sim alone is bottlenecked downstream. A real speedup requires decoupling actors from the learner (Ape-X / IMPALA-style async actor-learner with a shared replay buffer), which is a substantial refactor; its primary win is sample diversity per gradient update, not raw wall-clock.
- No ELO-league mode in eval (round-robin across checkpoints w/ K=32 ELO updates) — would be a natural follow-up to `eval.ts`.
- No `bot.ts` (heuristic search) baseline — that bot is wired through the gameserver protocol, not a clean function. Future work: extract its decide-loop as a pluggable Agent.
