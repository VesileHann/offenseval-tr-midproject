# Mid-project deck

- `Team10_MidProject.pdf` — submission file (14 pages: 10 talk + 4 Project Card appendix).
- `Team10_MidProject.html` — redesigned, self-contained fixed-stage web deck (works offline after fonts cache).
- `Team10_MidProject_legacy.*` — previous deck, kept for reference only.

## Source

The content is data-driven from `midproject/frontend/`:

- `content.mjs` — all slide copy and layouts; numbers are read from
  `midproject/results.json`, `finetune.json`, `zero_shot_diagnostics.json`,
  `timing_from_logs.json` and `error_audit.json`.
- `build.mjs` — fixed 1920×1080 stage, typography, charts and controls.
- `render.mjs` — screenshots every slide for visual checks.
- `export-pdf.mjs` — clean PDF export (chrome hidden).

Rebuild and export:

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
| Ctrl+S | Save edits while in edit mode |
| Wheel / swipe | Previous / next |

The stage is authored at 1920×1080 and scaled uniformly; it never reflows slide
content for phones. Reduced-motion preferences are respected.
