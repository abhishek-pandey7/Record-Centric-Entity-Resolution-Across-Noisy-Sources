# One Business, Many Records

**Record-Centric Entity Resolution Across Noisy Sources** — Amazon ML Challenge 2026.

This project resolves business records from three noisy sources into real-world entities. Each record retrieves its likely owners via multi-key blocking, a LightGBM matcher scores the pairs, and the final decisions optimize F0.5. A transliteration dictionary learned from the training data handles Indic scripts, and country-agnostic features extend to France.

## Layout

| Path | Contents |
|---|---|
| `code/business_entity_resolution/` | Pipeline source (`src/`), tests, run instructions |
| `tasks/plan.md`, `tasks/todo.md` | Design decisions, data findings, task list |
| `submissions/log.md` | Leaderboard submission history |

The competition dataset and documents are not included. See `code/business_entity_resolution/README.md` to reproduce the results.
