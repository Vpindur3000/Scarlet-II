# Third-party components

Scarlet II contains original engine code and a small number of clearly separated third-party GPL components.

## Pyrrhic Syzygy probing

Directory: `third_party/pyrrhic/`

These files originate from the Pyrrhic tablebase probing code as distributed in Berserk 14. Original copyright and license notices are preserved. Scarlet II uses them only for local Syzygy probing.

Upstream: Berserk by Jay Honnold and contributors. License: GNU GPL version 3.

## Lc0 policy map

Directory: `third_party/lc0-policy-map/`

Scarlet II includes one unmodified Lc0 table header, `neural/tables/policy_map.h`, used by regression tests to verify policy encoding. Its original copyright header is preserved and the corresponding GPL license is included.

Upstream: Leela Chess Zero and contributors. License: GNU GPL version 3 or later.

Scarlet II does not include Lc0 search code, inference backends, executables, or neural-network weights.

No upstream project endorses or is affiliated with Scarlet II.
