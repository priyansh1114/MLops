$ErrorActionPreference = "Stop"

$image = "heart-disease-api:latest"
$context = (& kubectl config current-context).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Unable to read the current Kubernetes context."
}
if ($context -ne "docker-desktop") {
    throw "This loader requires the docker-desktop Kubernetes context; current context is '$context'."
}

& docker image inspect $image *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Docker image '$image' is not available. Build it first with docker build -t $image ."
}

$containerdProcesses = & wsl.exe -d docker-desktop -- sh -lc "ps -ef"
if ($LASTEXITCODE -ne 0) {
    throw "Unable to inspect the Docker Desktop WSL processes."
}

$runtimeLine = $containerdProcesses |
    Select-String -Pattern "^\s*(\d+)\s+.*\s/usr/local/bin/containerd$" |
    Select-Object -First 1
if (-not $runtimeLine) {
    throw "Unable to locate the Docker Desktop Kubernetes containerd runtime."
}
$runtimePid = [regex]::Match($runtimeLine.Line, "^\s*(\d+)\s").Groups[1].Value

$importCommand = "docker save $image | wsl.exe -d docker-desktop -- chroot /proc/$runtimePid/root /usr/local/bin/ctr --address /run/containerd/containerd.sock -n k8s.io images import --platform linux/amd64 -"
& $env:ComSpec /c $importCommand
if ($LASTEXITCODE -ne 0) {
    throw "Failed to import '$image' into Docker Desktop Kubernetes."
}

Write-Output "Imported $image into Docker Desktop Kubernetes containerd."
