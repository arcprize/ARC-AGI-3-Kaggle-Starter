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

; Game-specific action-prior overrides.
(= (game-prior ar25 ACTION1) 0.55)
(= (game-prior ar25 ACTION2) 0.78)
(= (game-prior ar25 ACTION3) 0.60)
(= (game-prior ar25 ACTION4) 0.72)
(= (game-prior ar25 ACTION5) 0.45)
(= (game-prior ar25 ACTION6) 0.30)

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

(= (prefer-novel-action stalled) True)
(= (prefer-successful-pattern repeated-progress) True)
