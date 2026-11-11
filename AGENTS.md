# Business Pulse engineering workflow

This Git repository is the source of truth. Do not edit the older extracted
`business_pulse_ml` workspace copy or publish its environment/ZIP files.

- Read README.md, CONTRIBUTING.md, VERSION, CHANGELOG.md and the task's relevant
  forecasting / alert contract before changing implementation.
- Keep existing src/scripts interfaces compatible unless a breaking change is
  explicitly required. Do not add infrastructure without a present requirement.
- Forecasts may use only information available at forecast_at. Preserve temporal
  validation, freeze selection before final evaluation, never tune on final TEST.
- Synthetic incident labels and hypothetical costs must remain explicitly marked.
  Forecast error is not financial loss; alert benefit requires separate evidence.
- Keep labels out of detection rules and deduplication state.
- Do not send Telegram messages or integrate external delivery without the user's
  explicit authorization. Tests use fake transports; new alert CLI has no send flag.
- Work on a focused feature/fix branch for new work. Make meaningful commits;
  update CHANGELOG's Unreleased section. Use PRs into main after tests pass.
- Before finishing: `python -m pytest -q`, `python -m scripts.check_repository`
  after staging, plus the relevant reproducible experiment. Report actual checks.
- VERSION identifies the application release. Report schema/protocol versions
  (currently 0.5 / 0.6) need not equal the semantic release version.
- Release only a tested commit, record its provenance, and never move an existing
  published version tag. Follow CONTRIBUTING.md for release commands.
- Never commit credentials, real transactions, local_state/, environments or
  unreviewed bot archives. Synthetic, immutable reports live in reports/vX.Y.Z/;
  new working runs go into ignored outputs/.
