# Mid-project deck

- `Team10_MidProject.pdf`: submission file (14 pages: 10 talk + 4 Project Card appendix).
- `Team10_MidProject.html`: self-contained fixed-stage web deck (works offline after fonts cache).

## Source

The deck is built from a separate `midproject/` project that sits beside this
repository, not inside it. Its `frontend/` directory holds:

- `content.mjs`: all slide copy and layouts; numbers are read from
  `midproject/results.json`, `finetune.json`, `zero_shot_diagnostics.json`,
  `timing_from_logs.json` and `error_audit.json`.
- `build.mjs`: fixed 1920×1080 stage, typography, charts and controls. It
  writes `frontend/Team10_MidProject_v2.html`, which is copied here as
  `Team10_MidProject.html`.
- `render.mjs`: screenshots every slide for visual checks.
- `export-pdf.mjs`: clean PDF export (chrome hidden).

Rebuild and export from the directory that contains `midproject/`:

```bash
node midproject/frontend/build.mjs
node midproject/frontend/render.mjs
node midproject/frontend/export-pdf.mjs
```

## Controls

| Key | Action |
|---|---|
| → / Space / PageDown | Next slide |
| ← / PageUp | Previous slide |
| Home / End | First / last slide |
| F | Full screen |
| E | Edit mode (text saved to localStorage) |
| N | Open the speaker-notes window with talk timer |
| Ctrl+S | Save edits while in edit mode |
| Wheel / swipe | Previous / next |

The stage is authored at 1920×1080 and scaled uniformly; it never reflows slide
content for phones. Reduced-motion preferences are respected.

Speaker notes are embedded in every slide. Press **N** (or use the Notes button)
to open a separate window with the current note, the 10:00 talk countdown, the
current-slide timer and the per-slide target. The talk limit is 10 minutes;
questions come after the talk. The same notes are written to
`SPEAKER_NOTES.md`, which is gitignored.
