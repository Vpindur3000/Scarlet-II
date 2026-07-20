# Scarlet II v0.7.1 — изменения и границы релиза

v0.7.1 — релиз корректности, воспроизводимости и UCI-совместимости поверх
поиска Modern B*. Он не объявляет коридоры NNUE/LD2 математическими
доказательствами игры.

## Что изменилось

- `position` теперь транзакционен: некорректный FEN или ход не меняет текущую
  позицию движка частично.
- Реализован `go searchmoves` в Modern B* и ClassicAB.
- Modern B* сохраняет и читает TT по полному контексту позиции (доска,
  rule-50, история повторений и восемь кадров LD2). Запись используется
  только для порядка ходов; её corridor не может вызвать отсечение.
- По умолчанию `HeuristicEarlyStop=false`. Даже `heuristic_strict` не
  завершает timed search досрочно: движок расходует время, выделенное GUI.
- LD2 cache ограничен `LD2CacheEntries` (по умолчанию 16384). При заполнении
  удаляются старые записи FIFO, а не очищается весь cache.
- Root-кандидаты LD2 обрабатываются реальным параллельным batch path через
  постоянный пул worker-потоков. В финальной строке Modern B* видны
  `ld2_batch_calls`, `ld2_batch_positions` и `ld2_batch_workers`.
- Syzygy подключён локально через MIT-лицензированный Pyrrhic probe: до шести
  фигур, WDL внутри поиска и DTZ-выбор лучшего хода в корне.
- Диагностика Modern B* сообщает `tt_order_hits` и состояние LD2 cache через
  команду `backend`.

## Новые или уточнённые UCI-элементы

```text
go ... searchmoves e2e4 d2d4
setoption name HeuristicEarlyStop value false
setoption name LD2CacheEntries value 16384
setoption name LD2BatchSize value 4
setoption name LD2BatchWorkers value 0
setoption name SyzygyPath value D:/tablebases/syzygy
setoption name SyzygyProbeLimit value 6
```

`LD2BatchSize` задаёт размер root-пакета (1..32); `LD2BatchWorkers` задаёт
размер постоянного пула (0 — auto, до четырёх потоков из `LeelaThreads`).
Это параллельное исполнение независимых позиций, а не векторизация tensor-N
внутри одной свёртки.

`SyzygyPath` принимает каталог (или список каталогов через системный разделитель)
с `.rtbw` и `.rtbz`. Tablebase-файлы не входят в исходный архив. `SyzygyProbeLimit`
не позволяет этой версии обращаться к 7-men файлам даже если они лежат в каталоге.

## Проверка релиза

```bash
./scripts/regression_v071.sh ./build/scarlet
```

Гейт включает exact perft, parity инкрементального NNUE, LD2 policy/cache,
Modern B*/ClassicAB, `go infinite`, transactional `position`, `go searchmoves`
и фактический LD2 root batch path.

## Что сознательно не обещает v0.7.1

- Сила не измерена матчами или SPRT и не выводится из технической регрессии.
- LD2 value/policy остаются эвристическими источниками, а не точными
  минимакс-границами.
- Tensor-N batch/пакетные AVX2-ядра ещё не реализованы; текущий batch
  распараллеливает независимые root-позиции между постоянными worker-потоками.
- Внутренний WDL shortcut намеренно применяется только при `halfmoveClock=0`;
  позиции с ненулевым счётчиком корректно решаются DTZ только в корне. Это
  консервативнее, чем выдавать ложную exact-границу по одному WDL.

Внешний воспроизводимый матч/SPSA-гейт описан в
[`StrengthValidation_v0.7.1.md`](StrengthValidation_v0.7.1.md). Он остаётся
эмпирической проверкой и не заменяется этой технической регрессией.
