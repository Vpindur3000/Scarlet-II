# Syzygy до 5–6 фигур

Scarlet использует локальные Syzygy-файлы через встроенный MIT-лицензированный
Pyrrhic/Fathom-совместимый probe. В дистрибутив таблицы не включены: укажите
каталог, где лежат пары WDL (`.rtbw`) и DTZ (`.rtbz`). На Windows несколько
каталогов разделяются `;`, на Linux — `:`.

```text
setoption name UseSyzygy value true
setoption name SyzygyPath value D:/chess/syzygy/6men
setoption name SyzygyProbeLimit value 6
isready
backend
```

`backend` покажет `largest=N`; это реальное максимальное число фигур среди
найденных файлов, а не заявка движка. Если там `largest=5`, Scarlet честно
использует только 3–5-men позиции даже при `SyzygyProbeLimit=6`.

## Семантика поиска

- В позиции без рокировки и с не более чем `min(largest, SyzygyProbeLimit)`
  фигурами Syzygy WDL даёт точную границу для внутреннего узла.
- В корне используется `tb_probe_root` с DTZ и текущим `halfmoveClock`; он
  возвращает конкретный легальный ход, сохраняющий WDL. Команда `syzygy`
  показывает `wdl`, `dtz` и этот ход.
- WDL сам по себе не получает текущий счётчик правила 50 ходов. Поэтому
  Scarlet консервативно применяет WDL shortcut внутри поиска только при
  `halfmoveClock=0`. При ненулевом счётчике DTZ решает root точно, а поиск
  безопасно продолжает обычным путём.
- Обычный WDL win/loss в UCI выводится как `cp ±20000`, не как `mate`: Syzygy
  доказывает результат, но не сообщает корректную длину мата.
- Cursed win и blessed loss не рекламируются как forced win/loss; они считаются
  draw в WDL shortcut. Это исключает ложное «доказательство» около 50 ходов.

Рокировка исключает probe по определению Syzygy. En passant передаётся в
backend. `go infinite` не получает преждевременный `bestmove`: даже полностью
решённая root-позиция ждёт `stop` как требует UCI.

## Проверка установки

```text
position fen 7k/8/8/8/8/8/PPPP4/K7 w - - 0 1
syzygy
go depth 2
```

При наличии `KPPPPvK.rtbw` и `KPPPPvK.rtbz` диагностическая команда должна
выдать `syzygy root ... largest 6`, а конечная строка поиска — `Syzygy exact
root ...`. Для полной проверки на реальном наборе нужны также естественные
таблицы-предшественники выбранного эндшпиля.

Автоматизированная проверка этого же 6-фигурного случая:

```text
python scripts/syzygy_smoke.py build-windows-avx2/scarlet.exe D:/chess/syzygy/6men
```

Исходный probe находится в `third_party/pyrrhic/` и сохраняет
его MIT-лицензию в `LICENSE`.
