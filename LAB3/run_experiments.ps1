param(
    [ValidateRange(1, 10000)] [int]$Epochs = 30,
    [string]$DataDir = "../data",
    [string]$Python = "python",
    [ValidateSet("auto", "cpu", "cuda")] [string]$Device = "auto",
    [ValidateRange(1, 4096)] [int]$BatchSize = 128,
    [ValidateRange(0, 32)] [int]$NumWorkers = 0,
    [ValidateRange(1, 64)] [int]$Threads = 4
)

$ErrorActionPreference = "Stop"
Push-Location -LiteralPath $PSScriptRoot
try {
    $experiments = @(
        @{ Name = "baseline"; Model = "baseline"; Augment = "none" },
        @{ Name = "ablation"; Model = "baseline"; Augment = "standard" },
        @{ Name = "improved"; Model = "improved"; Augment = "standard" }
    )
    foreach ($experiment in $experiments) {
        $resultDir = Join-Path "results" $experiment.Name
        Write-Host ("Training {0}: model={1}, augmentation={2}" -f $experiment.Name, $experiment.Model, $experiment.Augment)
        $trainArguments = @(
            "-m", "src.train", "--model", $experiment.Model,
            "--augment", $experiment.Augment, "--data-dir", $DataDir,
            "--output-dir", $resultDir, "--epochs", $Epochs,
            "--batch-size", $BatchSize, "--num-workers", $NumWorkers,
            "--device", $Device, "--threads", $Threads, "--channels-last"
        )
        & $Python @trainArguments
        if ($LASTEXITCODE -ne 0) { throw "Training failed: $($experiment.Name)" }
    }

    # All configurations are fixed before looking at any test-set results.
    foreach ($experiment in $experiments) {
        $resultDir = Join-Path "results" $experiment.Name
        Write-Host ("Evaluating validation-selected checkpoint: {0}" -f $experiment.Name)
        $evaluateArguments = @(
            "-m", "src.evaluate", "--checkpoint", (Join-Path $resultDir "best.pt"),
            "--data-dir", $DataDir, "--output-dir", $resultDir,
            "--batch-size", $BatchSize, "--num-workers", $NumWorkers,
            "--device", $Device, "--threads", $Threads
        )
        & $Python @evaluateArguments
        if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $($experiment.Name)" }
    }
}
finally {
    Pop-Location
}
