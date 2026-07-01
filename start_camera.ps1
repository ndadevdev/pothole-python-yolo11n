$local = (tailscale ip 2>$null).Trim()
$peers = tailscale status 2>$null |
    Select-String -Pattern '^\d+\.\d+\.\d+\.\d+' |
    ForEach-Object { $_.ToString().Split(' ', [StringSplitOptions]::RemoveEmptyEntries)[0] } |
    Where-Object { $_ -ne $local }
if (-not $peers) {
    Write-Host "ERROR: Tidak ada device Tailscale lain ditemukan." -ForegroundColor Red
    Write-Host "Pastikan HP sudah install Tailscale dan login akun yang sama." -ForegroundColor Yellow
    Read-Host "Press Enter"
    exit 1
}
$camIp = $peers[0]
Write-Host "Connecting to $camIp :8080 ..." -ForegroundColor Green
& "$PSScriptRoot\.venv\Scripts\python.exe" main.py --camera "http://${camIp}:8080/video"
Read-Host "Press Enter"
