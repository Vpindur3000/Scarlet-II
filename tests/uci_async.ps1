param([Parameter(Mandatory = $true)][string]$Engine)
$ErrorActionPreference = 'Stop'

function Invoke-EngineSteps([array]$Steps) {
    $process = [Diagnostics.Process]::new()
    $process.StartInfo.FileName = $Engine
    $process.StartInfo.WorkingDirectory = Split-Path -Parent $Engine
    $process.StartInfo.UseShellExecute = $false
    $process.StartInfo.RedirectStandardInput = $true
    $process.StartInfo.RedirectStandardOutput = $true
    $process.StartInfo.CreateNoWindow = $true
    if (-not $process.Start()) { throw 'failed to start engine' }
    # Complete synchronous network load before latency measurements. Without
    # this handshake a stop timestamp taken 40 ms after Process.Start would
    # mostly measure cold executable/network startup, not search cancellation.
    $process.StandardInput.Write("isready`n")
    $process.StandardInput.Flush()
    $prefix = ''
    do {
        $line = $process.StandardOutput.ReadLine()
        if ($null -eq $line) { throw 'engine exited before readyok' }
        $prefix += $line + "`n"
    } while ($line -ne 'readyok')
    $stopWatch = $null
    foreach ($step in $Steps) {
        if ($step.Text -eq "stop`n") { $stopWatch = [Diagnostics.Stopwatch]::StartNew() }
        $process.StandardInput.Write($step.Text)
        $process.StandardInput.Flush()
        if ($step.Delay -gt 0) { Start-Sleep -Milliseconds $step.Delay }
    }
    $process.StandardInput.Close()
    $output = $prefix + $process.StandardOutput.ReadToEnd()
    if (-not $process.WaitForExit(3000)) {
        $process.Kill()
        throw 'engine did not exit after stop/quit'
    }
    if ($null -ne $stopWatch) { $stopWatch.Stop() }
    return @{ Output = $output; StopMilliseconds = if ($null -eq $stopWatch) { 0 } else { $stopWatch.ElapsedMilliseconds } }
}

function Count-Bestmoves([string]$Output) {
    return @($Output -split "`r?`n" | Where-Object { $_ -match '^bestmove ' }).Count
}

# Let enabled LD2 enter the search before cancellation. The gate measures from
# the UCI stop write through join/process exit, not engine startup.
$stopped = Invoke-EngineSteps @(
    @{ Text = "position startpos`n"; Delay = 0 },
    @{ Text = "go infinite`n"; Delay = 40 },
    @{ Text = "stop`n"; Delay = 0 },
    @{ Text = "quit`n"; Delay = 0 })
if ((Count-Bestmoves $stopped.Output) -ne 1) { throw 'stop did not produce exactly one bestmove' }
if ($stopped.StopMilliseconds -gt 250) { throw "stop latency exceeded gate: $($stopped.StopMilliseconds) ms" }

$quit = Invoke-EngineSteps @(
    @{ Text = "position startpos`n"; Delay = 0 },
    @{ Text = "go infinite`n"; Delay = 30 },
    @{ Text = "quit`n"; Delay = 0 })
if ((Count-Bestmoves $quit.Output) -ne 0) { throw 'quit emitted bestmove' }

$ponder = Invoke-EngineSteps @(
    @{ Text = "position startpos`n"; Delay = 0 },
    @{ Text = "go ponder wtime 1000 btime 1000`n"; Delay = 20 },
    @{ Text = "ponderhit`n"; Delay = 20 },
    @{ Text = "stop`n"; Delay = 0 },
    @{ Text = "quit`n"; Delay = 0 })
if ((Count-Bestmoves $ponder.Output) -ne 1) { throw 'ponderhit/stop did not produce exactly one bestmove' }

$nodes = Invoke-EngineSteps @(
    @{ Text = "position startpos`n"; Delay = 0 },
    @{ Text = "go nodes 5000`n"; Delay = 400 },
    @{ Text = "quit`n"; Delay = 0 })
if ((Count-Bestmoves $nodes.Output) -ne 1) { throw 'node-limited go did not finish exactly once' }

$timed = Invoke-EngineSteps @(
    @{ Text = "position startpos`n"; Delay = 0 },
    @{ Text = "go movetime 50`n"; Delay = 300 },
    @{ Text = "quit`n"; Delay = 0 })
if ((Count-Bestmoves $timed.Output) -ne 1) { throw 'time-limited go did not finish exactly once' }
if ($timed.Output -notmatch 'info string bounds \[') { throw 'Modern result omitted diagnostic bounds' }

$restarted = Invoke-EngineSteps @(
    @{ Text = "position startpos`n"; Delay = 0 },
    @{ Text = "go infinite`n"; Delay = 20 },
    @{ Text = "go nodes 5000`n"; Delay = 20 },
    @{ Text = "setoption name Hash value 32`n"; Delay = 0 },
    @{ Text = "go movetime 30`n"; Delay = 200 },
    @{ Text = "stop`n"; Delay = 0 },
    @{ Text = "quit`n"; Delay = 0 })
$restartBestmoves = Count-Bestmoves $restarted.Output
if ($restartBestmoves -ne 3) { throw "new go/setoption lifecycle produced $restartBestmoves bestmoves:`n$($restarted.Output)" }

$missingNetwork = Join-Path ([IO.Path]::GetTempPath()) 'scarlet-definitely-missing-policy_value.pb'
$degraded = Invoke-EngineSteps @(
    @{ Text = "setoption name LeelaWeightsFile value $missingNetwork`n"; Delay = 0 },
    @{ Text = "position startpos`n"; Delay = 0 },
    @{ Text = "go movetime 30`n"; Delay = 200 },
    @{ Text = "quit`n"; Delay = 0 })
if ($degraded.Output -notmatch 'info string error network LD2') { throw 'missing LD2 error was silent' }
if ($degraded.Output -notmatch 'info string warning degraded backend') { throw 'missing LD2 degraded warning was silent' }
if ((Count-Bestmoves $degraded.Output) -ne 1) { throw 'degraded search did not return a move' }

Write-Output "ok: async stop=$($stopped.StopMilliseconds)ms quit ponderhit limits restart setoption degraded"
