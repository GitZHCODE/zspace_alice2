param(
    [string]$Model = "llama3.2-vision:latest",
    [int]$Iteration = 0,
    [double]$TargetAverageScore = 8.5,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$AliceDir = Join-Path $RepoRoot "alice2"
$ExePath = Join-Path $AliceDir "build_zspace\bin\Release\alice2.exe"
$ScreenshotsDir = Join-Path $AliceDir "src\screenshots"
$AssetDir = Join-Path $AliceDir "userSrc\zspace\SDF\UrbanDesing_VLM_Critique_A_assets"
$LogPath = Join-Path $AliceDir "userSrc\zspace\SDF\UrbanDesing_VLM_Critique_A.md"
$PromptPath = Join-Path $AliceDir "userSrc\zspace\SDF\UrbanDesing_VLM_Critique_A_prompt.txt"

New-Item -ItemType Directory -Force -Path $ScreenshotsDir, $AssetDir | Out-Null

if (-not $SkipBuild) {
    Push-Location $AliceDir
    try {
        & ".\build_with_zspace.bat"
        if ($LASTEXITCODE -ne 0) {
            throw "alice2 build failed. If alice2.exe is open, close it and rerun this script."
        }
    }
    finally {
        Pop-Location
    }
}

if (-not (Test-Path -LiteralPath $ExePath)) {
    throw "alice2.exe not found at $ExePath"
}

function Invoke-Capture {
    param(
        [string]$Mode,
        [string]$OutputPath
    )

    $start = Get-Date
    $env:URBAN_CODEX_AUTOCAPTURE = "1"
    $env:URBAN_CODEX_CAPTURE_MODE = $Mode

    Push-Location $AliceDir
    try {
        $process = Start-Process -FilePath $ExePath -WorkingDirectory $AliceDir -Wait -PassThru
        if ($process.ExitCode -ne 0) {
            throw "alice2 capture failed for mode '$Mode' with exit code $($process.ExitCode)"
        }
    }
    finally {
        Pop-Location
        Remove-Item Env:\URBAN_CODEX_AUTOCAPTURE -ErrorAction SilentlyContinue
        Remove-Item Env:\URBAN_CODEX_CAPTURE_MODE -ErrorAction SilentlyContinue
    }

    $shot = Get-ChildItem -LiteralPath $ScreenshotsDir -Filter "*.png" |
        Where-Object { $_.LastWriteTime -ge $start.AddSeconds(-2) } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if (-not $shot) {
        throw "No screenshot was written for mode '$Mode' in $ScreenshotsDir"
    }

    Copy-Item -LiteralPath $shot.FullName -Destination $OutputPath -Force
    return $OutputPath
}

function Convert-ImageToBase64 {
    param([string]$Path)
    return [Convert]::ToBase64String([IO.File]::ReadAllBytes($Path))
}

function Invoke-OllamaVisionCritique {
    param(
        [string]$FigurePath,
        [string]$GradientPath
    )

    $prompt = Get-Content -LiteralPath $PromptPath -Raw
    $body = @{
        model = $Model
        prompt = $prompt
        stream = $false
        images = @(
            (Convert-ImageToBase64 $FigurePath),
            (Convert-ImageToBase64 $GradientPath)
        )
    } | ConvertTo-Json -Depth 6

    $response = Invoke-RestMethod -Method Post -Uri "http://localhost:11434/api/generate" -ContentType "application/json" -Body $body
    return [string]$response.response
}

function Get-AverageScore {
    param([string]$Critique)
    $matches = [regex]::Matches($Critique, "(?i)(urban planner|architect)[^0-9]{0,40}([0-9]+(?:\.[0-9]+)?)\s*/\s*10")
    if ($matches.Count -lt 2) { return $null }
    $sum = 0.0
    foreach ($match in $matches) {
        $sum += [double]$match.Groups[2].Value
    }
    return $sum / $matches.Count
}

$iterationLabel = "iteration_{0:D3}" -f $Iteration
$figurePath = Join-Path $AssetDir "$iterationLabel`_figure.png"
$gradientPath = Join-Path $AssetDir "$iterationLabel`_gradient.png"

$figurePath = Invoke-Capture -Mode "figure" -OutputPath $figurePath
$gradientPath = Invoke-Capture -Mode "gradient" -OutputPath $gradientPath
$critique = Invoke-OllamaVisionCritique -FigurePath $figurePath -GradientPath $gradientPath
$averageScore = Get-AverageScore -Critique $critique
$date = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

$relativeFigure = "UrbanDesing_VLM_Critique_A_assets/$iterationLabel`_figure.png"
$relativeGradient = "UrbanDesing_VLM_Critique_A_assets/$iterationLabel`_gradient.png"
$scoreText = if ($null -eq $averageScore) { "not parsed" } else { "{0:N2}" -f $averageScore }

$entry = @"

## Iteration $Iteration

- Date: $date
- Ollama model: $Model
- Average score parsed by script: $scoreText

### Figure Screenshot

![Iteration $Iteration figure]($relativeFigure)

### Gradient Screenshot

![Iteration $Iteration gradient]($relativeGradient)

### VLM Critique

```text
$critique
```

### Codex Update Notes

- Pending Codex interpretation and parameter/code update for the next iteration.
"@

Add-Content -LiteralPath $LogPath -Value $entry

Write-Host "[URBAN VLM LOOP] Wrote $LogPath"
Write-Host "[URBAN VLM LOOP] Figure: $figurePath"
Write-Host "[URBAN VLM LOOP] Gradient: $gradientPath"
Write-Host "[URBAN VLM LOOP] Average score: $scoreText"

if ($null -ne $averageScore -and $averageScore -ge $TargetAverageScore) {
    Write-Host "[URBAN VLM LOOP] Target score achieved."
    exit 0
}

Write-Host "[URBAN VLM LOOP] Target not reached. Codex should update parameters/assignments before the next iteration."
exit 2
