# Paper arena and bot registry

The market watch, Alpaca paper broker, and internal mock venue share one set of
account-owned strategy seats. They have separate order and accounting tables.
No endpoint accepts a live trading host or live credential.

## Authority and credentials

`permission_groups.authority` resolves the highest direct human permission
group and implies every lower group. User-created content-sharing groups do not
enter this calculation. `member`, `dev`, and `operator` use their own Alpaca
paper key. `admin` and service accounts use the one key holder configured by an
admin at `/api/paper/shared-account`. The credential remains in the restricted
state secrets tree and is never returned by the paper APIs.

## Venue and accounting

Alpaca paper buys are capped per strategy: cautious $5 per order and $50 gross
buys; standard $25/$250; explore $100/$1000. An intent and unique
`client_order_id` are committed before the broker request. A repeated
`request_id` returns the same intent. An uncertain response stays uncertain
until `/api/paper/orders/{id}/reconcile` reads that client order ID from Alpaca.
The scoreboard uses recorded broker fill quantities and prices, then marks
holdings at the latest available bar; it reports returns before fees. Alpaca
strategy sells and a full execution-fee ledger are not implemented.

The internal mock venue is explicitly `internal_mock`. It starts each strategy
with its risk-tier gross-buy budget as cash, uses cached latest bars, and fills
immediately with assumptions recorded on the receipt: 10 bps spread, 5 bps
markup, 5 bps fee. It supports buys and sells against its own holdings, tracks
cash and realized gains, and never changes Alpaca positions. It currently
assumes the capped order is fully liquid at its modeled execution price; it
does not replay the order book or simulate partial fills. Mock receipts must
not be used as Alpaca order-success training targets.

## Bots and trainers

`/api/paper/bots` registers a source reference and owner. Python references
name an owned Divider job (`divider:<12-hex>`); symbolic references name an
owned Math Lab job (`mathlab:<16-hex>`); compiled references name a catalogued
product (`product:<64-hex>`); remote API registrations contain an HTTPS URL;
`agent` registrations need no source. Registration does not execute code,
compile jobs, or call remote URLs. A runner and remote API invocation protocol
remain to be built before registered sources can autonomously trade.

A bot uses its returned ID in `X-Meowdoku-Bot` on authenticated paper or market
data requests. `/api/paper/presence` shows bots seen in the last five minutes;
the Markets page fades their robot cards by last activity. REST and DM paths
are known from the route. The MCP bridge supplies an `mcp_reported` marker;
this is descriptive only and can be supplied by another HTTP client. The
server never uses the marker for authorization. A future server-side kernel
runner will produce the `kernel` channel.

`/api/paper/trainer-events?after=<cursor>` is a paged, sanitized event stream
for service/admin network trainers (`paper_train` capability). Multiple
trainers keep independent cursors. Events distinguish `internal_mock` fills
from Alpaca paper submissions and fill deltas. An Alpaca paper fill event is
emitted once for the initially returned filled quantity or for a later
increase on reconciliation. Trainer processes can register as bots and use
the same strategy/order APIs; no autonomous tensor trainer is started by this
feature.

## Historical admission gate

Automated mock and Alpaca paper order routes require a historical qualification
for the strategy's assigned bot and risk tier. Qualification is bound to the
registered source hash, so changing a registration invalidates the proof. The
internal judge stores the historical dataset digest, window net P&L, and sell
fill counts. It requires at least three windows, positive net P&L and at least
one sell fill in each, a passing validation verdict, and average net P&L of at least 1% of the risk tier's
maximum per-order capital. Registration and read-only market access remain
possible without qualification. No public route accepts claimed backtest
returns.

The judge entry trusts the in-process replay caller's P&L and fill counts; a
deployed admission service must bind those to immutable mock receipts. The
`X-Meowdoku-Bot` header is client supplied, and an unassigned manual strategy
can order without it. Thus this gate covers registered automated strategies,
not every possible authenticated HTTP client. No autonomous runner or
scheduled historical judge is deployed yet.

## Qualification spectators across server instances

`qualification_runs` stores server-owned worker progress in the same
`paper_arena.sqlite` as the arena. Any Meowdoku instance using that state path
serves the same bounded, sanitized run list at the public HTTPS GET
`/api/paper/qualification/live`. The Markets page polls it every two seconds,
and signed-in users can also use `look qualification` in the paper DM. Rows
contain worker label, training phase, epoch count, validation checkpoints and
final historical verdict; they contain no weights, credentials, source paths or account
activity. A run with no update for 90 seconds is displayed as stalled. The
runner publishes every five epochs and at validation/compile/holdout phases.

The current Nodus historical trainer accepts `--publish-db` and `--worker` to
publish to that shared arena file. A local scratch run completed 480 epochs in
about 19 seconds and appeared as unqualified. This is cross-instance
spectating, not distributed gradient training or a running server scheduler.
`passed_historical` is a spectator result, not an automatic registration or
permission to order. Training is bounded to three seeds and 160 epochs each; it does not continue
until it finds a profit. The already-inspected AAPL holdout cannot be reused
for a new admission decision.

## Modular strategy experiments

Each strategy stores a policy module and a nonempty selection of outcome
modules. The initial policies are `manual`, `momentum`, `mean_reversion`,
`tensor`, and `external`. The first two are explicit dry-run baselines;
`tensor` and `external` report that they need trained weights or an external
runner. `/api/paper/proposals` evaluates a page of strategies against one
cached history snapshot and places no orders. This keeps signal comparison
separate from broker execution.

The independent outcome modules are `fixed_horizon`, `closed_trade`, and
`fill_quality`. They require the relevant future mark, close fill, or execution
reference before producing a label. `TensorOutcomeNet` uses the existing
`AbstractTensor` autograd to fit all three heads in one batch, with an observed
mask so an immature or unselected outcome contributes no loss. No one reward
rule is fixed for every strategy. A future trainer must persist reference
marks, mature labels from Alpaca paper fill events, and checkpoints before its
network can issue proposals. The current tensor policy deliberately returns
`ready=false`.

## Interfaces

The SpeakToMe REST catalog declares all paper routes and generates signed-in
MCP tools from the same entries. `/api/paper/dm` relays `look`, `examine`,
`make`, and `use` commands to these operations. `/api/paper/map` projects
monitored symbols, strategies, and order receipts into the existing
`abstract-ui-div-map-v0` Living Data graph; `/api/paper/map/query` uses its
`GraphView` query contract. The Markets page has the human DM form and bot
presence container.
