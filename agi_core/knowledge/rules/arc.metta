; ARC symbolic policy scaffold for the executive agent prototype.
; The Python bridge currently mirrors these rules into a shadow fact store so
; the runtime can behave deterministically even when the Hyperon package is not
; installed. The expressions here are the long-term source of truth for the
; symbolic policy.

(= (reset-required NOT_PLAYED) True)
(= (reset-required GAME_OVER) True)
(= (goal advance-level) True)
(= (goal avoid-reset-loop) True)
(= (goal finish-episode) True)
(= (goal-hint WIN) hold-win)
(= (goal-hint NOT_PLAYED) re-enter-level)
(= (goal-hint GAME_OVER) re-enter-level)
(= (goal-hint ACTIVE) advance-level)
(= (goal-hint NOT_FINISHED) advance-level)
(= (stagnation-hint True) break-stall)
(= (stagnation-hint False) advance-level)

(= (symbolic-prior ACTION1) 0.72)
(= (symbolic-prior ACTION2) 0.68)
(= (symbolic-prior ACTION3) 0.64)
(= (symbolic-prior ACTION4) 0.60)
(= (symbolic-prior ACTION5) 0.56)
(= (symbolic-prior ACTION6) 0.52)

; Board-type-specific action-prior overrides.
(= (board-prior puzzle ACTION1) 0.30)
(= (board-prior puzzle ACTION2) 0.35)
(= (board-prior puzzle ACTION3) 0.30)
(= (board-prior puzzle ACTION4) 0.30)
(= (board-prior puzzle ACTION5) 0.80)
(= (board-prior puzzle ACTION6) 0.70)

(= (board-prior platformer ACTION1) 0.60)
(= (board-prior platformer ACTION2) 0.65)
(= (board-prior platformer ACTION3) 0.60)
(= (board-prior platformer ACTION4) 0.65)
(= (board-prior platformer ACTION5) 0.40)
(= (board-prior platformer ACTION6) 0.35)

; Board-type-specific goal hints — augment the state-based hints above.
(= (board-goal-hint puzzle) explore-board)
(= (board-goal-hint platformer) navigate-to-goal)
(= (board-goal-hint unknown) advance-level)

; Board-type-specific stagnation thresholds — how many retries before
; stagnation is declared.  Puzzle boards give up sooner (switch entity),
; platformers try harder.
(= (board-stagnation-threshold puzzle) 2)
(= (board-stagnation-threshold platformer) 5)
(= (board-stagnation-threshold unknown) 3)

; Board-type reset thresholds — how many stagnation steps before RESET.
(= (board-reset-threshold puzzle) 8)
(= (board-reset-threshold platformer) 12)
(= (board-reset-threshold unknown) 6)

; Board-type explore bias — which axis to prioritize when exploring.
; platformer: horizontal movement covers more ground.
; puzzle: vertical movement explores layers.
(= (board-explore-bias platformer) horizontal)
(= (board-explore-bias puzzle) vertical)
(= (board-explore-bias unknown) horizontal)

; Board-type interaction range — max distance for clicking on interactive
; objects.  Puzzles need wider range, platformers are tighter.
(= (board-interact-range puzzle) 4)
(= (board-interact-range platformer) 2)
(= (board-interact-range unknown) 3)

; Board-type movement step preference — which direction to try first.
; platformer: right/down (scrolling direction)
; puzzle: down/up (vertical exploration)
(= (board-move-bias platformer) right)
(= (board-move-bias puzzle) down)
(= (board-move-bias unknown) right)

; Whether to prefer beam navigation over direct movement per board type.
; high = always try beam nav first; low = skip beam nav for movement.
(= (board-beam-priority puzzle) high)
(= (board-beam-priority platformer) low)
(= (board-beam-priority unknown) medium)

; How urgently to switch controllable entities when stuck per board type.
; high = switch after few retries; low = exhaust all options first.
(= (board-switch-urgency puzzle) high)
(= (board-switch-urgency platformer) low)
(= (board-switch-urgency unknown) medium)

; Path preference for A* navigation per board type.
; direct = prefer shortest path; safe = avoid interactive obstacles.
(= (board-path-preference puzzle) direct)
(= (board-path-preference platformer) safe)
(= (board-path-preference unknown) balanced)

; Per-action retry tolerance per board type — how many times to retry
; a failed action before giving up.  puzzle = fewer (switch sooner).
(= (board-retry-tolerance puzzle) 2)
(= (board-retry-tolerance platformer) 5)
(= (board-retry-tolerance unknown) 3)

; Goal proximity threshold per board type — max Manhattan distance to
; consider a goal as "reached".  puzzle needs tight precision.
(= (board-goal-proximity puzzle) 1)
(= (board-goal-proximity platformer) 3)
(= (board-goal-proximity unknown) 2)

; Expected controllable entity count per board type.
; puzzle = multiple entities to switch between; platformer = single.
(= (board-entity-count puzzle) multiple)
(= (board-entity-count platformer) single)
(= (board-entity-count unknown) single)

; Whether gravity affects movement per board type.
; platformer has gravity pulling down; puzzle has no gravity.
(= (board-gravity-enabled puzzle) false)
(= (board-gravity-enabled platformer) true)
(= (board-gravity-enabled unknown) false)

; Whether the board scrolls horizontally per board type.
; platformer often scrolls; puzzle typically stays fixed.
(= (board-horizontal-scroll puzzle) false)
(= (board-horizontal-scroll platformer) true)
(= (board-horizontal-scroll unknown) false)

; Collision behavior per board type.
; solid = hard walls/obstacles; overlap = soft boundaries.
(= (board-collision-type puzzle) solid)
(= (board-collision-type platformer) solid)
(= (board-collision-type unknown) solid)

; How level exit is triggered per board type.
; position = reach a specific tile; action = perform an action.
(= (board-exit-type puzzle) action)
(= (board-exit-type platformer) position)
(= (board-exit-type unknown) position)

; Probe strategy per board type — how to interact with the environment.
; click = probe/inspect cells; swipe = directional gesture.
(= (board-probe-strategy puzzle) click)
(= (board-probe-strategy platformer) swipe)
(= (board-probe-strategy unknown) click)

; Focus mode per board type — single vs multi-entity attention.
; single = focus on one entity; multi = distribute across entities.
(= (board-focus-mode puzzle) multi)
(= (board-focus-mode platformer) single)
(= (board-focus-mode unknown) single)

; Navigation style per board type — how movement works.
; grid = snap to tiles; free = pixel-based movement.
(= (board-nav-style puzzle) grid)
(= (board-nav-style platformer) free)
(= (board-nav-style unknown) free)

; Danger detection per board type — whether to avoid hazards.
; avoid = pathfind around hazards; ignore = treat as normal terrain.
(= (board-danger-detection puzzle) ignore)
(= (board-danger-detection platformer) avoid)
(= (board-danger-detection unknown) ignore)

; Viewport style per board type — how the board is viewed.
; fixed = full board visible; follow = viewport follows player.
(= (board-viewport-style puzzle) fixed)
(= (board-viewport-style platformer) follow)
(= (board-viewport-style unknown) fixed)

; Solver selection rules — map dynamics categories to solver names.
(= (solver-for static) noop)
(= (solver-for navigation) navigation)
(= (solver-for sliding) sliding)
(= (solver-for gravity) gravity)
(= (solver-for push-pull) push-pull)
(= (solver-for beam) beam)
(= (solver-for signal) signal)
(= (solver-for cellular-automata) observe)
(= (solver-for pattern-transform) observe)
(= (solver-for multi-agent) multi-agent)
(= (solver-for inventory) inventory)
(= (solver-for hybrid) hybrid)
(= (solver-for puzzle) mirror)
(= (solver-for mirror) mirror)

; ---------------------------------------------------------------------------
; Dynamics-based board policy rules (12-category taxonomy).
; These augment the board-type rules above and are preferred when the primary
; dynamics category is known.
; ---------------------------------------------------------------------------
; static — no player interaction possible
(= (board-goal-hint static) none)
(= (board-stagnation-threshold static) 10)
(= (board-reset-threshold static) 20)
(= (board-explore-bias static) none)
(= (board-interact-range static) 0)
(= (board-move-bias static) none)
(= (board-beam-priority static) low)
(= (board-switch-urgency static) low)
(= (board-path-preference static) balanced)
(= (board-retry-tolerance static) 10)
(= (board-goal-proximity static) 5)
(= (board-entity-count static) single)
(= (board-gravity-enabled static) false)
(= (board-horizontal-scroll static) false)
(= (board-collision-type static) solid)
(= (board-exit-type static) position)
(= (board-probe-strategy static) click)
(= (board-focus-mode static) single)
(= (board-nav-style static) free)
(= (board-danger-detection static) ignore)
(= (board-viewport-style static) fixed)

; navigation — standard agent movement with pathfinding
(= (board-goal-hint navigation) navigate-to-goal)
(= (board-stagnation-threshold navigation) 5)
(= (board-reset-threshold navigation) 10)
(= (board-explore-bias navigation) horizontal)
(= (board-interact-range navigation) 2)
(= (board-move-bias navigation) right)
(= (board-beam-priority navigation) low)
(= (board-switch-urgency navigation) low)
(= (board-path-preference navigation) direct)
(= (board-retry-tolerance navigation) 5)
(= (board-goal-proximity navigation) 2)
(= (board-entity-count navigation) single)
(= (board-gravity-enabled navigation) false)
(= (board-horizontal-scroll navigation) false)
(= (board-collision-type navigation) solid)
(= (board-exit-type navigation) position)
(= (board-probe-strategy navigation) click)
(= (board-focus-mode navigation) single)
(= (board-nav-style navigation) free)
(= (board-danger-detection navigation) ignore)
(= (board-viewport-style navigation) fixed)

; sliding — momentum / ice physics, objects continue until blocked
(= (board-goal-hint sliding) navigate-to-goal)
(= (board-stagnation-threshold sliding) 6)
(= (board-reset-threshold sliding) 12)
(= (board-explore-bias sliding) horizontal)
(= (board-interact-range sliding) 2)
(= (board-move-bias sliding) right)
(= (board-beam-priority sliding) low)
(= (board-switch-urgency sliding) low)
(= (board-path-preference sliding) safe)
(= (board-retry-tolerance sliding) 6)
(= (board-goal-proximity sliding) 2)
(= (board-entity-count sliding) single)
(= (board-gravity-enabled sliding) false)
(= (board-horizontal-scroll sliding) true)
(= (board-collision-type sliding) solid)
(= (board-exit-type sliding) position)
(= (board-probe-strategy sliding) click)
(= (board-focus-mode sliding) single)
(= (board-nav-style sliding) free)
(= (board-danger-detection sliding) avoid)
(= (board-viewport-style sliding) follow)

; gravity — platformer-style falling mechanics
(= (board-goal-hint gravity) navigate-to-goal)
(= (board-stagnation-threshold gravity) 5)
(= (board-reset-threshold gravity) 12)
(= (board-explore-bias gravity) horizontal)
(= (board-interact-range gravity) 2)
(= (board-move-bias gravity) right)
(= (board-beam-priority gravity) low)
(= (board-switch-urgency gravity) low)
(= (board-path-preference gravity) safe)
(= (board-retry-tolerance gravity) 5)
(= (board-goal-proximity gravity) 3)
(= (board-entity-count gravity) single)
(= (board-gravity-enabled gravity) true)
(= (board-horizontal-scroll gravity) true)
(= (board-collision-type gravity) solid)
(= (board-exit-type gravity) position)
(= (board-probe-strategy gravity) swipe)
(= (board-focus-mode gravity) single)
(= (board-nav-style gravity) free)
(= (board-danger-detection gravity) avoid)
(= (board-viewport-style gravity) follow)

; push-pull — Sokoban-like block manipulation
(= (board-goal-hint push-pull) explore-board)
(= (board-stagnation-threshold push-pull) 3)
(= (board-reset-threshold push-pull) 10)
(= (board-explore-bias push-pull) vertical)
(= (board-interact-range push-pull) 1)
(= (board-move-bias push-pull) down)
(= (board-beam-priority push-pull) medium)
(= (board-switch-urgency push-pull) medium)
(= (board-path-preference push-pull) direct)
(= (board-retry-tolerance push-pull) 3)
(= (board-goal-proximity push-pull) 1)
(= (board-entity-count push-pull) single)
(= (board-gravity-enabled push-pull) false)
(= (board-horizontal-scroll push-pull) false)
(= (board-collision-type push-pull) solid)
(= (board-exit-type push-pull) position)
(= (board-probe-strategy push-pull) click)
(= (board-focus-mode push-pull) single)
(= (board-nav-style push-pull) grid)
(= (board-danger-detection push-pull) ignore)
(= (board-viewport-style push-pull) fixed)

; beam — ray / mirror / light-propagation puzzles
(= (board-goal-hint beam) explore-board)
(= (board-stagnation-threshold beam) 2)
(= (board-reset-threshold beam) 8)
(= (board-explore-bias beam) vertical)
(= (board-interact-range beam) 4)
(= (board-move-bias beam) down)
(= (board-beam-priority beam) high)
(= (board-switch-urgency beam) high)
(= (board-path-preference beam) direct)
(= (board-retry-tolerance beam) 2)
(= (board-goal-proximity beam) 1)
(= (board-entity-count beam) multiple)
(= (board-gravity-enabled beam) false)
(= (board-horizontal-scroll beam) false)
(= (board-collision-type beam) solid)
(= (board-exit-type beam) action)
(= (board-probe-strategy beam) click)
(= (board-focus-mode beam) multi)
(= (board-nav-style beam) grid)
(= (board-danger-detection beam) ignore)
(= (board-viewport-style beam) fixed)

; signal — circuit / switch / logic puzzles
(= (board-goal-hint signal) explore-board)
(= (board-stagnation-threshold signal) 3)
(= (board-reset-threshold signal) 8)
(= (board-explore-bias signal) vertical)
(= (board-interact-range signal) 4)
(= (board-move-bias signal) down)
(= (board-beam-priority signal) medium)
(= (board-switch-urgency signal) high)
(= (board-path-preference signal) direct)
(= (board-retry-tolerance signal) 3)
(= (board-goal-proximity signal) 1)
(= (board-entity-count signal) multiple)
(= (board-gravity-enabled signal) false)
(= (board-horizontal-scroll signal) false)
(= (board-collision-type signal) solid)
(= (board-exit-type signal) action)
(= (board-probe-strategy signal) click)
(= (board-focus-mode signal) multi)
(= (board-nav-style signal) grid)
(= (board-danger-detection signal) ignore)
(= (board-viewport-style signal) fixed)

; cellular-automata — board evolves without player input
(= (board-goal-hint cellular-automata) observe)
(= (board-stagnation-threshold cellular-automata) 8)
(= (board-reset-threshold cellular-automata) 15)
(= (board-explore-bias cellular-automata) horizontal)
(= (board-interact-range cellular-automata) 1)
(= (board-move-bias cellular-automata) right)
(= (board-beam-priority cellular-automata) low)
(= (board-switch-urgency cellular-automata) low)
(= (board-path-preference cellular-automata) balanced)
(= (board-retry-tolerance cellular-automata) 8)
(= (board-goal-proximity cellular-automata) 3)
(= (board-entity-count cellular-automata) single)
(= (board-gravity-enabled cellular-automata) false)
(= (board-horizontal-scroll cellular-automata) false)
(= (board-collision-type cellular-automata) solid)
(= (board-exit-type cellular-automata) position)
(= (board-probe-strategy cellular-automata) click)
(= (board-focus-mode cellular-automata) single)
(= (board-nav-style cellular-automata) free)
(= (board-danger-detection cellular-automata) ignore)
(= (board-viewport-style cellular-automata) fixed)

; pattern-transform — board changes according to inferred rules
(= (board-goal-hint pattern-transform) observe)
(= (board-stagnation-threshold pattern-transform) 6)
(= (board-reset-threshold pattern-transform) 12)
(= (board-explore-bias pattern-transform) vertical)
(= (board-interact-range pattern-transform) 3)
(= (board-move-bias pattern-transform) down)
(= (board-beam-priority pattern-transform) medium)
(= (board-switch-urgency pattern-transform) medium)
(= (board-path-preference pattern-transform) direct)
(= (board-retry-tolerance pattern-transform) 6)
(= (board-goal-proximity pattern-transform) 2)
(= (board-entity-count pattern-transform) single)
(= (board-gravity-enabled pattern-transform) false)
(= (board-horizontal-scroll pattern-transform) false)
(= (board-collision-type pattern-transform) solid)
(= (board-exit-type pattern-transform) action)
(= (board-probe-strategy pattern-transform) click)
(= (board-focus-mode pattern-transform) multi)
(= (board-nav-style pattern-transform) grid)
(= (board-danger-detection pattern-transform) ignore)
(= (board-viewport-style pattern-transform) fixed)

; multi-agent — switch between multiple controllable entities
(= (board-goal-hint multi-agent) explore-board)
(= (board-stagnation-threshold multi-agent) 3)
(= (board-reset-threshold multi-agent) 8)
(= (board-explore-bias multi-agent) vertical)
(= (board-interact-range multi-agent) 3)
(= (board-move-bias multi-agent) down)
(= (board-beam-priority multi-agent) medium)
(= (board-switch-urgency multi-agent) high)
(= (board-path-preference multi-agent) direct)
(= (board-retry-tolerance multi-agent) 3)
(= (board-goal-proximity multi-agent) 1)
(= (board-entity-count multi-agent) multiple)
(= (board-gravity-enabled multi-agent) false)
(= (board-horizontal-scroll multi-agent) false)
(= (board-collision-type multi-agent) solid)
(= (board-exit-type multi-agent) action)
(= (board-probe-strategy multi-agent) click)
(= (board-focus-mode multi-agent) multi)
(= (board-nav-style multi-agent) grid)
(= (board-danger-detection multi-agent) ignore)
(= (board-viewport-style multi-agent) fixed)

; inventory — collect items before reaching an exit
(= (board-goal-hint inventory) navigate-to-goal)
(= (board-stagnation-threshold inventory) 4)
(= (board-reset-threshold inventory) 10)
(= (board-explore-bias inventory) horizontal)
(= (board-interact-range inventory) 1)
(= (board-move-bias inventory) right)
(= (board-beam-priority inventory) low)
(= (board-switch-urgency inventory) low)
(= (board-path-preference inventory) direct)
(= (board-retry-tolerance inventory) 4)
(= (board-goal-proximity inventory) 2)
(= (board-entity-count inventory) single)
(= (board-gravity-enabled inventory) false)
(= (board-horizontal-scroll inventory) false)
(= (board-collision-type inventory) solid)
(= (board-exit-type inventory) position)
(= (board-probe-strategy inventory) click)
(= (board-focus-mode inventory) single)
(= (board-nav-style inventory) free)
(= (board-danger-detection inventory) ignore)
(= (board-viewport-style inventory) fixed)

; hybrid — multiple dynamics detected simultaneously
(= (board-goal-hint hybrid) explore-board)
(= (board-stagnation-threshold hybrid) 3)
(= (board-reset-threshold hybrid) 8)
(= (board-explore-bias hybrid) vertical)
(= (board-interact-range hybrid) 3)
(= (board-move-bias hybrid) down)
(= (board-beam-priority hybrid) medium)
(= (board-switch-urgency hybrid) medium)
(= (board-path-preference hybrid) direct)
(= (board-retry-tolerance hybrid) 3)
(= (board-goal-proximity hybrid) 1)
(= (board-entity-count hybrid) multiple)
(= (board-gravity-enabled hybrid) false)
(= (board-horizontal-scroll hybrid) false)
(= (board-collision-type hybrid) solid)
(= (board-exit-type hybrid) action)
(= (board-probe-strategy hybrid) click)
(= (board-focus-mode hybrid) multi)
(= (board-nav-style hybrid) grid)
(= (board-danger-detection hybrid) ignore)
(= (board-viewport-style hybrid) fixed)

(= (prefer-novel-action stalled) True)
(= (prefer-successful-pattern repeated-progress) True)
