# pandabomber-ai

Self-play RL training for PandaBomber bots, plus a model loader for the gameserver.

## Architecture

| File | Purpose |
|---|---|
| `src/sim.ts` | Pure-TS port of `backend-gameserver/src/game.ts` (deterministic, MU-only, time-stepped at 10ms — matches production's 100Hz tickRate exactly). |
| `src/observation.ts` | Encodes game state → `Float32Array` (multi-channel grid + scalar features). Shared between trainer and bot2. |
| `src/env.ts` | Wraps `Sim` as an RL environment (one obs per seat, action committed at cell-aligned ticks). |
| `src/model.ts` | TF.js dueling CNN (Conv 3×3/32 → Conv 3×3/32 → Dense 64 → V(1) + A(6) → Q) + shape-aware JSON weight export so inference needs no TF.js. Also serializes optimizer state. |
| `src/train.ts` | Self-play Double-Dueling-DQN: PER buffer, n-step returns, soft target net, per-seat frozen-opponent pool, 2→4-player curriculum, random warmup. |
| `src/per.ts` | Sum-tree prioritized replay buffer with stratified sampling + IS-weight normalization. |
| `src/parity_test.ts` | Confirms bot2's pure-JS forward pass (including the dueling combine) matches tfjs's `predict()` to <1e-6 on shared weights. |
| `../backend-gameserver/src/bot2.ts` | Drop-in bot that loads the JSON weights and runs a pure-JS conv+dense+dueling forward pass. |

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
                       ┌──┴──┐
              Dense 1  │     │  Dense 6
              (value)  │     │  (advantage)
                       └──┬──┘
                 Q = V + (A − mean(A))   ← DuelingCombine layer
```

Conv layers see the 13×19 grid directly, which is far more sample-efficient than the
previous flat MLP (translation-invariant features come for free). The flat
observation buffer is laid out in NHWC order so `tf.tensor4d(flat, [N, H, W, C])` and
bot2's `conv2dRelu()` can both consume it without a transpose.

The dueling head decouples "how good is this state" (V) from "how much better is action
a than the average" (A). In Bomberman most states are dominated by survive/die, so V
absorbs that and A learns the much smaller per-action differences — empirically more
sample-efficient than a single Q head of the same total size.

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
| Collect powerup | +0.5 |
| Knock an enemy | +5.0 |
| Kill an enemy | +15.0 |
| Get knocked | −10.0 |
| Die | −30.0 |
| Win the game | +30.0 |
| Per simulation step | −0.003 |
| Alive at timeout (stalemate) | −15.0 |

Wood reward is +1.0 — a step up from the original +0.2 that left wood as a weak
bootstrap signal and let the policy settle into a mutual-avoidance equilibrium where
neither side bothered destroying anything before the timeout. We don't go higher
because (a) a single well-placed bomb can clear multiple wood blocks at once, and
(b) destroying wood also spawns powerups (+0.5 each) — so the effective per-bomb
wood-clear reward is already a multiple of the per-block number. Pumping wood up
forces early activity, which exposes more enemies to bomb placement and breaks the
stalemate trap without dwarfing the +15 kill / +30 win signals.

Knock/kill rewards still drive combat (kill = 3× knock so chasing the kill always beats
trading knocks). The per-step penalty was relaxed from −0.008 to −0.003 in tandem with
the wood-reward bump — combined, the −111 reward floor of pure stalemate (which was
exactly `-0.008 × 12000 + -15`) is no longer an attractor.

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

Checkpoints are written to `checkpoints/latest.json` as plain JSON.

## Opponent pool

To kill the classic self-play failure mode (online policy drifting against its own
moving target), opponent seats don't play the live online network. Instead, every episode:

1. For **each opponent seat independently**, with probability `pPool` (default 0.5)
   sample weights from a rolling pool of historical snapshots; otherwise mirror the
   current online weights. The learner thus faces a mini-tournament of mixed opponents
   in 4-player games rather than three copies of the same policy.
2. Seat 0 (learner) plays `online` with ε-greedy exploration (ε annealed from
   `epsStart=1.0` to `epsEnd=0.05` over `epsDecaySteps=3M` env-steps).
3. Opponent seats play their frozen weights with a small fixed `epsOpponent=0.05`
   exploration so they're not perfectly predictable against the learner.
4. All seats record transitions to the shared PER buffer (DQN is off-policy, so
   frozen-policy data is valid training signal for the learner).
5. Every `snapshotEverySteps` env-steps (default 250k), push the current online
   weights into the pool, evicting the oldest if over `poolSize` (default 10).
   Step-based rather than episode-based so snapshot cadence is invariant to episode
   length changes from the curriculum.

Knobs: `--pPool=0.5 --poolSize=10 --snapshotEverySteps=250000 --epsOpponent=0.05`.

**Warm-start / resuming:** if `--save` already points at an existing checkpoint, the
trainer loads those weights and resumes — the saved `globalStep` and `episode` counters
are restored so ε-decay continues from the right place. Adam momentum / variance state
is also serialized and re-applied after the first learn step (slot variables don't
exist until then). Override with `--resume=off` to start fresh and overwrite, or
`--resume=on` to require a checkpoint (errors if none found).

Under the default `--resume=auto`, an incompatible checkpoint (e.g., arch tag mismatch
after a network change) is renamed to `<save>.incompat-<timestamp>.bak` and training
starts fresh, so an unattended run can't be killed by a stale file. `--resume=on`
still throws so callers that insist on resume are never silently downgraded.

The replay buffer is NOT persisted across runs — it refills during warmup.

## Training machinery

### Curriculum

Episodes 0–500 (configurable via `--curriculumEpisodes`) are 2-player games (`--curriculumPlayers=2`),
which makes early credit assignment much easier — there's exactly one opponent to model
and roughly half the bomb chaos. From episode 500 onward training switches to full
4-player MU games (`--fullPlayers=4`), the format the bot actually ships in.

### Per-tick inference

Cells take ~120–220ms to traverse at 10ms/tick, so on the vast majority of ticks
fewer than 4 seats are cell-aligned. The training loop only runs a forward pass
for seats that actually need a fresh decision (cell-aligned + alive + !knocked).
Each opponent seat carries its own independently-sampled weights (see Opponent pool
above), so opponent forwards are per-seat batch-1 rather than a shared batch — the
trade is more forwards per tick in exchange for a more diverse opponent mix.

### Prioritized Experience Replay (PER)

`src/per.ts` is a sum-tree buffer with proportional prioritization. New transitions
get `maxPriority` so each is sampled at least once before its priority is updated by
its first TD error. Sampling is stratified: `[0, total)` is sliced into `batchSize`
equal segments, one uniform draw per segment, sum-tree walk to the leaf — O(log N)
per sample, naturally diverse mini-batches.

Loss is IS-weighted Huber: `w_i = (N · P(i))^(−β)` normalized by the batch max so
the heaviest sample has weight 1.0 (standard PER trick). β anneals 0.4 → 1.0 over
`perBetaSteps` env-steps (default 3M) — early-training bias is OK in exchange for
faster learning, full correction by end.

Knobs: `--bufferSize=100000 --perAlpha=0.6 --perBetaStart=0.4 --perBetaEnd=1.0 --perBetaSteps=3000000`.

### n-step returns

`nStep=5` (default): each transition stored is `(s_t, a_t, Σ γ^k·r_{t+k} for k<n, s_{t+n}, done, n)`.
A per-seat sliding window holds the last `nStep` decisions; on each new decision we
emit the head transition with bootstrap from `s_{t+n}`. At episode end the trailing
window flushes with `done=true` and monotonically decreasing `n` (bootstrap is
zeroed by `done`, so the truncated returns are exact).

Why n=5: a single tick is ~10ms but cell-aligned decisions happen every 12–22 ticks,
so 5 inter-decision steps span roughly the bomb fuse (~3000ms) — enough to credit a
"place bomb → walk away → enemy dies" sequence without n-step DQN's variance blowup.

### Random warmup

The first `warmupSteps=10000` env-steps run with `ε=1.0` and no learning, so the
buffer fills with a baseline of uniformly-random transitions before training starts.
Avoids the first thousand gradient updates training on a tiny biased buffer.

### Soft target network

`target ← τ·online + (1-τ)·target` after every learn step (`targetTau=0.005`).
Smoother bootstrap targets than hard periodic copies, no need to pick a copy period.

### Logging

Per-episode `[train]` line and a slower `[summary]` line every `--logEvery=10`
episodes that averages the last N episodes: win rate, stalemate rate, learner survival
rate, wood destroyed, kills scored, knocks landed, deaths, own-bomb suicides, bombs
placed, powerups collected, mean learner/opponent reward, mean steps per episode.
Stalemate rate falling toward 0 + win rate climbing well above 25% (random baseline
for 4-player) is the signal training is working; meanR alone is not enough.

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
