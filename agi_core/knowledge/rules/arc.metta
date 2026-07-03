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

(= (prefer-novel-action stalled) True)
(= (prefer-successful-pattern repeated-progress) True)
