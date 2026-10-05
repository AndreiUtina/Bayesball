# Bayesball — Project Plan

A public website that ranks players using Bayesian statistics. Every player's skill
is a probability distribution, not a single number. Each game result (including the
score margin) updates those distributions, and the site uses them to show the current
ranking and to predict the winner and the likely score of upcoming games.

**Requirements (confirmed):**
- **First sport: foosball** (table football), mainly **2v2** (one player on **attack**, one on
  **defence** per team), with 1v1 also possible. Games are always **first to 10**.
  The platform stays universal, so other sports and team sizes can be added later as new *game types*.
- The **roles** (attack/defence) are recorded for every 2v2 game, and each player gets a separate
  attack and defence rating.
- The platform supports **draws** for game types that can have them. Foosball can't (first to 10).
- The **score margin** counts (a 10–2 win says more than a 10–8 win).
- Sides are **neutral**: there is no home advantage.
- Runs **online** for free, deployed **from git** (push to GitHub, and the site updates).
- **Admins:** the owner (you) can invite other admins and remove them.

---

## 1. Goals

| # | Feature | Description |
|---|---------|-------------|
| G1 | **Leaderboard** | Current ranking of all players (overall, attack, defence tabs), with skill estimate, uncertainty, games played and W-L. |
| G2 | **Add / edit players** | A form to create a player and set a starting belief about them (the *prior*). |
| G3 | **Record games** | Pick the attacker and defender of each team (or the two players in 1v1) and the losing team's score (0–9); the winner has 10. Ratings update automatically. |
| G4 | **Predictions** | Pick two sides and see P(win) and the expected score. A **team balancer** suggests the fairest teams *and roles* from 4+ players. |
| G5 | **History** | Past games, and each player's rating over time. |
| G6 | **Online + admins** | Anyone can view the site. Admins can add and edit players and games. The owner manages the admins. |
| G7 | **Multiple game types** | Foosball first. Other sports can be added later with their own ratings and settings. |

Out of scope for v1: public sign-up, a mobile app, a paid custom domain.

---

## 2. The Bayesian model

### 2.1 Skill representation: one skill per role
In foosball 2v2, one player plays **attack** (the front rods) and one plays **defence** (the goalie
and defence rods). Many players are much better in one role, so each player *i* gets **two linked
skills**:

```
s_i = (attack_i, defence_i) ~ Normal( μ_i, Σ_i )        μ_i: 2 means,  Σ_i: 2×2 covariance
```

- **μ (mu)**: the best guesses, measured in **goals of score margin** compared with an average
  player (0 = average; +2 = "wins by ~2 more goals than average").
- **Σ (sigma)**: how uncertain we are, plus how the two skills are related. New players start with
  large uncertainty and a positive **correlation ρ** between attack and defence: a player who is
  good at attack is probably at least decent at defence. Because of this link, playing in one role
  also teaches the model a little about the other role.
- **Overall skill** = (attack + defence) / 2. This is the main leaderboard number.

The engine is generic: a game type defines its list of roles. Foosball has `[attack, defence]`.
A sport without roles has one skill per player, and everything below still works.

### 2.2 Observation model: score margin
For a 2v2 game, team A's goals depend on A's attacker against B's defender, and the reverse for B.
Taking the difference gives:

```
margin = score_A − score_B
margin = ½(attack_A,att + defence_A,def) − ½(attack_B,att + defence_B,def) + ε,   ε ~ Normal(0, β²)
```

- **1v1**: each player handles all the rods, so their strength is their overall skill ½(attack + defence).
  This puts 1v1 and 2v2 games **on the same scale**, and both feed one leaderboard.
- **β** is the game's natural randomness: how much the margin varies between two equal teams.
- **Neutral sides**: there is no home-advantage term.
- **Draws**: the platform supports them (margin = 0). Foosball is first to 10, so a foosball game
  can't end in a draw. The winner always has 10 points, and the margin is 1–10.
- **Score cap**: because of the race to 10, the margin is at most 10. The Gaussian model is a good
  approximation for normal games. Phase 1 validation checks this for very unequal teams.

In general, every game is a linear observation `margin = hᵀs + ε`, where *s* stacks the skills of the
players involved and *h* holds ±½ for the skill each player used.

### 2.3 Updating after a game (exact Bayesian update)
Both the prior and the likelihood are Gaussian, so the posterior is Gaussian too and is computed
**exactly in closed form** (a Kalman filter update, no sampling needed):

```
predicted mean   m = hᵀμ
predicted var    v = hᵀ P h + β²          P = covariance of the involved skills
surprise         r = observed_margin − m
gain             K = P h / v

μ ← μ + K · r
P ← P − K Kᵀ · v
```

How to read this:
- An unexpected result (large r) moves ratings more than an expected one.
- Skills we know little about move more than well-established ones.
- The skills each player **used** in the game move most. Their other role moves a little, through the correlation ρ.
- Every game reduces uncertainty.

Before each game, a drift term is added (Σ_i += τ²·I) so skills can change over time.
Extra drift is added for players who have been inactive for a long time.

The posterior after each game becomes the prior for the next. (We keep each player's own 2×2
covariance, not the correlations *between* different players. This is the standard
"assumed-density filtering" approximation, which Phase 6 checks against a full model.)

### 2.4 Games without a score
For other sports, a game type can be **win/draw/loss only**. These games use a probit likelihood
with a draw margin (the TrueSkill update) on the same skill model.

### 2.5 Priors (input at the start)
When a player is added, the form converts simple answers into a prior:

| Input on the form | Maps to |
|-------------------|---------|
| Experience: Beginner / Average / Good / Very good | starting overall μ = −1 / 0 / +1 / +2 × *skill scale* |
| Stronger at: Attack / Defence / Both equal | attack and defence μ shifted by ±½ × *skill scale* |
| "How sure are you?": Low / Medium / High | starting σ = 1.0 / 0.7 / 0.4 × default σ |
| (Advanced) raw μ and σ for each role | used directly |

With no input, the player gets μ = 0 for both roles and the default σ.

### 2.6 Ranking
Sorting by μ alone would let a lucky new player top the table, so the leaderboard sorts by a
**conservative score**:

```
ranking_score = μ − k·σ      (k = 2 by default, configurable)
```

The leaderboard has three tabs: **Overall**, **Attack** and **Defence**. Each shows μ ± σ, and
players with fewer than 5 games are labelled *provisional*.

### 2.7 Predictions
For any two teams, with known roles, the predicted margin is Normal(m, v) from the formulas in 2.3:

- **Expected score**: shown as a scoreline, e.g. "10 – 7 for A"
- **P(A wins)** = P(margin > 0). For game types with draws, P(draw) = P(−0.5 < margin < 0.5).
- **Team & role balancer**: given 4 players, it checks all 12 options (3 ways to split the players
  into teams × 2 role choices per team). It shows the fairest options (P(win) closest to 50%), and
  for fixed teams it shows which role assignment gives each team its best chance.

### 2.8 Settings per game type
| Setting | Meaning | Foosball (starting guess) |
|---------|---------|----------|
| `roles` | list of roles | attack, defence |
| `scoring` | `margin` or `win_draw_loss` | margin |
| `points_to_win` | race-to target (form check + scoreline) | 10 |
| `allow_draws` | can the game end level? | no |
| `team_strength` | `sum` or `mean` of the players' skills | mean |
| `beta` | game randomness (goals) | 2.5 |
| `default_sigma` | prior uncertainty per skill | 2.5 |
| `role_correlation` | prior ρ between a player's roles | 0.7 |
| `skill_scale` | size of one experience step in the prior | 1.5 |
| `tau` | skill drift per game | 0.1 |

The starting values are guesses. Once there are ~50+ games, **β, τ and ρ are tuned from the data**
by picking the values that best predict the games so far (empirical Bayes / maximum likelihood).

---

## 3. Tech stack

| Layer | Choice | Why |
|-------|--------|-----|
| Backend | **Python 3.12 + FastAPI** | Python has the best statistics libraries; FastAPI is simple and gives auto API docs. |
| Rating engine | `numpy`, `scipy` (own code, ~200 lines) | The model's update is a few lines of linear algebra. |
| Database | **SQLModel**: SQLite locally, **PostgreSQL (Neon)** online | Same code for both. |
| Frontend | **Jinja2 templates + HTMX + Pico CSS** | No JS build step; small, fast pages that work on phones (useful next to the foosball table). |
| Charts | Chart.js | Rating over time, win-probability bars. |
| Auth | Session-cookie login, hashed passwords (argon2) | Roles: owner / admin. |
| Tests / CI | pytest + **GitHub Actions** | Tests run on every push. |
| Tooling | `uv`, `ruff`, Docker | |

---

## 4. Hosting: free, deployed from git

```
  your laptop ──git push──▶ GitHub repo ──▶ GitHub Actions (run tests)
                                   │
                                   └──auto-deploy──▶ Render (free web service)
                                                       https://bayesball.onrender.com
                                                            │
                                                            └──▶ Neon (free Postgres)
```

| Piece | Service | Free tier | Notes |
|-------|---------|-----------|-------|
| Code | **GitHub** | free (public or private repo) | The single source of truth. |
| Web app | **Render** free web service | 750 instance-hours/month (enough for one app running all month) | Deploys automatically on every push to `main`. Free address: `bayesball.onrender.com` (if the name is free). **Sleeps after 15 min without visitors; the first visit after that takes ~1 min to wake.** |
| Database | **Neon** free Postgres | 0.5 GB per project (enough for millions of games) | Render's own free Postgres is **deleted after 30 days**, so we use Neon instead. |
| Domain | Render subdomain | free | A custom `.com` costs ~€10/year and can be added later without code changes. |

The settings live in a `render.yaml` file in the repo ("infrastructure as code"). The secrets
(`DATABASE_URL`, `SECRET_KEY`) are entered once in the Render dashboard and never committed.

If the ~1 min wake-up becomes annoying, we can upgrade to Render's cheapest paid instance,
or move to another host. The app is a plain Docker container, so moving is easy.

---

## 5. Admins & permissions

| Role | Can do |
|------|--------|
| **Visitor** (no login) | View the leaderboard, players, history and predictions. |
| **Admin** | Everything a visitor can, plus add/edit players, record/edit/delete games, import CSV. |
| **Owner** (you) | Everything an admin can, plus **invite and remove admins**, change game-type settings, recompute ratings, transfer ownership. |

- **First start**: while there is no owner, the app writes a one-time setup link to the server log (valid for 1 day); opening it creates the owner account.
- **Inviting an admin**: the owner creates an **invite link** (valid for 7 days, single use) and sends it, e.g. via WhatsApp. The person opens it and picks a username and password. No email server is needed.
- **Removing an admin**: one click on `/admin/users`; their sessions end immediately.
- **Audit trail**: every game and player stores who created and last edited it, so mistakes can be traced and fixed.

---

## 6. Data model

```
GameType
  id, name, roles, scoring, points_to_win, allow_draws, team_strength,
  beta, default_sigma, role_correlation, skill_scale, tau

Player
  id, name, created_at, active

PlayerRating                      # one per (player, game type)
  player_id, game_type_id
  prior_mu, prior_cov              # what was entered at creation (per role)
  mu, cov                          # current posterior (cached): e.g. 2 means + 2×2 covariance
  games_played, last_played_at

Game
  id, game_type_id, played_at, notes
  score_a, score_b                 # foosball: winner = 10, loser 0–9; null for win_draw_loss games
  outcome                          # A / B / DRAW (derived from the score)
  created_by, updated_by

GameParticipant                   # one row per player per game
  game_id, player_id, side (A|B)
  role (attack|defence|null)       # required for foosball 2v2; null in 1v1
  mu_before, cov_before, mu_after, cov_after       # for history charts

User
  id, username, password_hash, role (owner|admin), active, created_at

Invite
  token_hash, created_by, expires_at, used_by, used_at
```

Because all results are stored, ratings can always be **recomputed from scratch** by replaying
games in date order. This happens automatically when a game is edited or deleted, a prior is
changed, or game-type settings change.

---

## 7. Pages & API

### Pages
| Route | Page | Access |
|-------|------|--------|
| `/` | Leaderboard (game-type selector; Overall / Attack / Defence tabs): rank, name, score, μ ± σ, games, W-L, trend | public |
| `/players/{id}` | Profile: attack & defence rating charts, recent games, best teammates, head-to-head | public |
| `/games` | Game history | public |
| `/predict` | Choose teams and roles and see P(win) and the expected score. Team & role balancer. | public |
| `/players/new` | Add a player with prior inputs | admin |
| `/games/new` | Record a game: 2v2 (attacker + defender per team) or 1v1, the winner, and the loser's score (0–9) | admin |
| `/admin` | Edit/delete games, CSV import/export | admin |
| `/admin/users` | Invite / remove admins | owner |
| `/admin/game-types` | Game-type settings, recompute | owner |
| `/login`, `/invite/{token}` | Login, accept invite | — |

### JSON API
```
GET    /api/game-types                      PATCH /api/game-types/{id}       (owner)
GET    /api/players?game_type=…             POST  /api/players               (admin)
GET    /api/players/{id}                    PATCH /api/players/{id}          (admin)
GET    /api/games?game_type=…               POST  /api/games                 (admin)
                                            PATCH/DELETE /api/games/{id}     (admin)
POST   /api/predict   { game_type, side_a: [{player, role}], side_b: [...] }
                       -> { p_a, p_draw, p_b, expected_margin, interval }
POST   /api/balance   { game_type, player_ids: [...] } -> fairest teams + roles
POST   /api/invites                                                          (owner)
DELETE /api/users/{id}                                                       (owner)
POST   /api/recompute { game_type }                                          (owner)
```

---

## 8. Project structure

```
Bayesball/
├── PLAN.md
├── README.md
├── pyproject.toml
├── Dockerfile
├── render.yaml                  # Render deploy config
├── .github/workflows/ci.yml     # tests + lint on every push
├── app/
│   ├── main.py                  # FastAPI app & routes
│   ├── config.py                # env settings (DATABASE_URL, SECRET_KEY, ...)
│   ├── db.py                    # engine / session
│   ├── models.py                # SQLModel tables
│   ├── auth.py                  # login, roles, invites
│   ├── rating/
│   │   ├── margin.py            # Gaussian margin model with roles (Kalman update)
│   │   ├── outcome.py           # win/draw/loss model (TrueSkill-style)
│   │   ├── predict.py           # probabilities, team & role balancer
│   │   └── replay.py            # recompute from history, parameter tuning
│   ├── templates/
│   └── static/
├── tests/
│   ├── test_margin.py
│   ├── test_predict.py
│   ├── test_auth.py
│   └── test_simulation.py       # recovers known skills from simulated games
└── data/                        # local SQLite (gitignored)
```

---

## 9. Milestones

| Phase | Deliverable | Done when |
|-------|-------------|-----------|
| **0. Setup** | git, GitHub repo, `uv` project, FastAPI "hello", ruff, pytest, GitHub Actions | A push to GitHub runs CI successfully |
| **1. Rating engine** | Margin model with attack/defence skills, drift, priors, predictions, team & role balancer, simulation test | Tests pass (see validation below) |
| **2. Data + API** | Models, CRUD, recompute on change, foosball game type | Players and games can be created through `/docs` |
| **3. Website** | Leaderboard, add-player form with priors, record-game form (2v2 with roles, and 1v1) | You can add players, enter games and see the ranking change |
| **4. Predictions & history** | `/predict`, team balancer, player profile with chart | Probabilities and the expected score are shown |
| **5. Go online** | Login, owner/admin roles, invite links, Docker, Neon, Render auto-deploy | The site is live at `*.onrender.com`; writes need a login; you can invite an admin |
| **6. Extras** | β/τ/ρ auto-tuning, CSV import/export, other sports, PyMC full model to cross-check | Optional |

### Validation (Phase 1)
Simulate 20 foosball players with known "true" attack and defence skills, and generate 500
random race-to-10 games (mostly 2v2 with random roles, some 1v1). Then run the engine and check that:
1. **Ranking**: the rank correlation between true and estimated overall skills is high (Spearman > 0.9),
   and the attack and defence skills are each recovered well (Spearman > 0.8).
2. **Calibration**: games predicted at 70% are won about 70% of the time.
3. **Margin**: the expected-margin error beats a "predict 0" baseline.
4. **Uncertainty**: about 90% of true skills fall inside the 90% intervals.

---

## 10. Decisions log

| Question | Answer |
|----------|--------|
| Game format | Foosball, mainly 2v2 (attack + defence per team), 1v1 possible; always first to 10, so no draws |
| Roles | Recorded for every 2v2 game; separate attack/defence ratings from day one |
| Home advantage | None: sides are neutral |
| Admins | The owner (Andrei) invites and removes other admins |
| Hosting | GitHub → Render (free `*.onrender.com` address) + Neon Postgres |
| Past games | None to import, so the leaderboard starts from scratch |


## 11. Roadmap for improvements.

1. Take into account the number of goals for predictions and ratings.

2. Improve the display panel for players: ex showing the total nb of goals or the total attack vs defense games.