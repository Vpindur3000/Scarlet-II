# Scarlet II v0.6 BETA: как устроен Modern B*

## Что здесь действительно является основным поиском

В v0.6 основной поиск хранит явное дерево `ProofNode`. У каждого узла есть:

```text
позиция, входящий ход, parent/children
[lower; upper]
Berserk value, Leela value, fused center
Leela/heuristic prior
danger, confidence, visits, proof depth
состояния evaluated / tactical / expanded / terminal
```

После корневого развёртывания движок не запускает полный alpha-beta отдельно
для каждого хода. На каждой proof-итерации выбирается одна важная ветвь и один
frontier node. Выбор идёт рекурсивно до неразвёрнутого узла, после чего узел
уточняется или разворачивается. Интервалы обновляются назад до корня.

## Построение рабочего коридора

Быстрая основная оценка — Berserk 14 NNUE. Если для узла вызвана LD2, её WDL
преобразуется в независимую pseudo-cp оценку:

```text
leela_cp = 400 * atanh(W - L), clamp [-2200; +2200]
```

Она специально не привязывается к score Berserk. Иначе «второе мнение»
наследовало бы ошибку первого оценщика.

Базовый коридор:

```text
lower = min(berserk, leela, tactical) - danger_margin
upper = max(berserk, leela, tactical) + danger_margin
```

Если Leela ещё не вызывалась, Berserk-only узел получает широкий fallback
margin. Это Reduced Proof Tier: ход не отбрасывается, но остаётся неопределённым.

Когда узел развёрнут, его children дают новый negamax-коридор:

```text
parent.lower = max(-child.upper)
parent.upper = max(-child.lower)
```

Полученный коридор пересекается с локальным dual-evaluator коридором, если они
согласуются. Если доказательство из потомков и локальные оценщики не
пересекаются, сохраняется полный hull обоих источников — конфликт расширяет
неопределённость, а не замалчивается.

Важно: это эвристические рабочие интервалы, не теоретические bounds идеальной
игры. Поэтому метка `heuristic_strict` означает строгое разделение именно этих текущих
интервалов.

## B-PUCT allocator

У корня allocator сравнивает текущего proof-лидера и challengers. Основные
сигналы:

```text
proof pressure: challenger.upper против best.lower
ширина коридора
Leela policy prior * sqrt(total visits) / (1 + visits)
danger
отсутствующая Leela value
штраф за уже потраченные visits
```

Внутри дерева цель чередуется по negamax:

- для лидера сжимается его pessimistic сторона;
- у challenger проверяется optimistic сторона;
- после перехода к ребёнку `lower`/`upper` меняются ролями из-за отрицания.

Это не MCTS: число visits не является финальным критерием выбора. Оно только
распределяет proof-бюджет. Финальный объект остаётся `[lower; upper]`.

## Насколько сильно влияет Leela

По умолчанию влияние намеренно заметное:

```text
LeelaPolicyWeight = 55%
LeelaValueWeight  = 45%
```

Policy смешивается с обычным move-ordering prior для каждого развёрнутого
узла. Поэтому странный ход с высоким prior не объявляется хорошим, но получает
право на раннее исследование.

Value используется как второе независимое мнение и остаётся частью локального
коридора после углубления. Вызовы LD2 происходят:

- у корня, если в лимите остаётся время;
- для верхних quiet root moves;
- на выбранных frontier nodes;
- для широких и proof-critical ветвей;
- для high-policy кандидатов.

LD2 не вызывается внутри alpha-beta/qsearch. Backend кэширует WDL и policy.
Энкодер содержит настоящие последние восемь кадров позиции; отсутствующая
история из FEN остаётся нулевой, как в legacy 112-plane encoding.

В release AVX2/OpenMP-сборке один LD2 probe обычно занимает миллисекунды, но
движок измеряет реальную среднюю стоимость и прекращает новые probes заранее,
если они уже не помещаются в `movetime`.

## HCE — санитария, не третий равный голос

Перед LD2 шумный frontier (`capture`, `promotion`, `check`, высокий danger)
проходит полный короткий HCE alpha-beta на 3–4 полухода с qsearch. Неполный
результат, прерванный временем или `stop`, не попадает ни в узел, ни в TT.

Тихие узлы не обязаны проходить AB: они сразу могут получить LD2 и/или быть
развёрнуты в Modern B* tree. Поэтому обычная тихая позиция не превращается
обратно в root-wrapper над alpha-beta.

## Три способа завершить выбор

```text
heuristic_strict:
    lower(proof_leader) > upper(all challengers)

heuristic_practical:
    lower(proof_leader) + PracticalMargin > challenger.upper
    + достаточный effort, устойчивость и узкий corridor

budget:
    время закончилось; lower-heavy soft score
    + center/upper + policy - width - danger
```

В debug выводятся и `decision best`, и `proof_leader`: при budget decision они
могут отличаться, и это нормально.

## Как читать `DebugModernBStar`

Пример строки:

```text
move=e2e4 L=10 U=57 C=33 W=47 b=32 l=35 tv=na fused=33
pol=0.2371 risk=0 conf=58 visits=5 depth=3 leela=1 tact=0 expanded=1
```

| Поле | Значение |
|---|---|
| `L/U` | коридор хода с точки зрения корня |
| `C/W` | центр и ширина |
| `b` | локальная Berserk NNUE оценка хода |
| `l` | локальная LD2 value или `na` |
| `tv` | результат tactical sanitizer или `na` |
| `fused` | локальный взвешенный центр |
| `pol` | смешанный Leela/heuristic prior |
| `risk/conf` | HCE danger и доверие к локальному коридору |
| `visits/depth` | proof effort и глубина subtree |

Строка `mbstar config` показывает обе корневые оценки и фактические веса
Leela. Финальная строка содержит размер дерева, число frontier expansions,
LD2 probes и HCE tactical refinements.

## Что уже есть и что оставлено на следующие beta

| Идея Revolution | v0.6 |
|---|---|
| Явное best-first interval tree | реализовано |
| Berserk NNUE как массовый evaluator | реализовано; L1 layout исправлен |
| LD2 value + policy | реализовано in-process |
| B-PUCT proof pressure | реализовано |
| Reduced Proof Tiering | реализовано как Berserk-only → LD2/tactical/full expansion |
| HCE tactical sanitation | реализовано, 3–4 ply |
| Interval TT | реализовано |
| Repetition / 50 moves / mate distance | реализовано |
| Full history/continuation/capture history | пока нет |
| SEE/ProbCut/singular proof extensions | пока нет |
| Correction history по структурам | пока нет |
| LD2 batch inference | root-пакеты через постоянный worker pool; tensor-N batch пока нет |
| Syzygy | v0.7.1: локальные WDL/DTZ tablebase до 6 фигур; tablebase-файлы не входят в архив |
| Формально доказуемые игровые bounds | нет; коридоры эвристические |

Цель этой beta — проверить жизнеспособность другого профиля поиска, а не
заявить готовую силу 3500 Elo. Силу и коэффициенты нужно измерять матчами,
SPRT и набором тактических/позиционных регрессий.
