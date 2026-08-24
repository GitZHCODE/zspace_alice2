param(
    [Parameter(Mandatory = $true)] [string] $CMakeExe,
    [Parameter(Mandatory = $true)] [string] $BuildDir,
    [Parameter(Mandatory = $true)] [string] $Config,
    [Parameter(Mandatory = $true)] [string] $ZspaceCoreDir,
    [Parameter(Mandatory = $true)] [string] $ZspaceToolsetsDir,
    [string] $BuildTarget = "alice2",
    [string] $BuildToolArgs = "/m /nr:false /clp:ErrorsOnly"
)

$ErrorActionPreference = "Stop"

function ConvertTo-ProcessArgument {
    param([Parameter(Mandatory = $true)] [string] $Argument)

    if ($Argument -notmatch '\s|"') { return $Argument }
    return '"' + ($Argument -replace '"', '\"') + '"'
}

function Start-CleanEnvironmentProcess {
    param(
        [Parameter(Mandatory = $true)] [string] $FileName,
        [Parameter(Mandatory = $true)] [string[]] $Arguments
    )

    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $FileName
    $psi.UseShellExecute = $false
    $psi.WorkingDirectory = (Get-Location).Path
    $psi.Arguments = ($Arguments | ForEach-Object { ConvertTo-ProcessArgument $_ }) -join " "

    $pathValue = $env:Path
    $psi.EnvironmentVariables.Clear()

    $sourceEnvironment = [Environment]::GetEnvironmentVariables("Process")
    foreach ($key in $sourceEnvironment.Keys) {
        if ($key -ieq "path") { continue }
        if ($psi.EnvironmentVariables.ContainsKey($key)) { continue }
        $psi.EnvironmentVariables[$key] = [string] $sourceEnvironment[$key]
    }
    $psi.EnvironmentVariables["Path"] = $pathValue

$process = [System.Diagnostics.Process]::Start($psi)
    $process.WaitForExit()
    return $process.ExitCode
}

$configureArgs = @(
    "-S", ".",
    "-B", $BuildDir,
    "-G", "Visual Studio 17 2022",
    "-A", "x64",
    "-DCMAKE_BUILD_TYPE=$Config",
    "-DCMAKE_SUPPRESS_REGENERATION=ON",
    "-DALICE2_WITH_ZSPACE_CORE=ON",
    "-DALICE2_USE_ZSPACE_SOURCE=ON",
    "-DZSPACE_CORE_DIR=$ZspaceCoreDir",
    "-DALICE2_WITH_ZSPACE_TOOLSETS=ON",
    "-DZSPACE_TOOLSETS_DIR=$ZspaceToolsetsDir"
)

$exitCode = Start-CleanEnvironmentProcess -FileName $CMakeExe -Arguments $configureArgs
if ($exitCode -ne 0) { exit $exitCode }

$buildArgs = @("--build", $BuildDir, "--config", $Config, "--target", $BuildTarget, "--")
$buildArgs += ($BuildToolArgs -split "\s+" | Where-Object { $_ })

$exitCode = Start-CleanEnvironmentProcess -FileName $CMakeExe -Arguments $buildArgs
exit $exitCode
