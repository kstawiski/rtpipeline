# Gotchas & Pitfalls

Things to watch out for in this codebase.

## [2026-01-05 22:58]
Lines 2150-2255 in organize.py had critical indentation bug causing entire fraction processing block to be unreachable (code after continue statement was indented inside the continue block)

_Context: Treatment fraction data collection in organize_and_merge function - now fixed_
