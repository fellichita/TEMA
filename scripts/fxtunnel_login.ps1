# Один раз сохраняет токен fxTunnel для публичной ссылки сайта.
# Токен берётся из буфера обмена: переносы строк и пробелы из него убираются,
# поэтому вставка токена, разбитого на две строки, больше ничего не ломает.
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$client = Join-Path $env:LOCALAPPDATA "fxTunnel\fxtunnel.exe"

function Finish([string] $message) {
    Write-Host ""
    Write-Host $message
    Write-Host ""
    $Host.UI.RawUI.FlushInputBuffer()
    Read-Host "Нажмите Enter, чтобы закрыть окно" | Out-Null
}

if (-not (Test-Path $client)) {
    Finish "Клиент fxTunnel не найден. Установите его: irm https://fxtun.ru/install.ps1 | iex"
    exit 1
}
Write-Host "============================================================"
Write-Host " Вход в fxTunnel"
Write-Host "============================================================"
Write-Host ""
Write-Host "1. Скопируйте API-токен (начинается с sk_fxtunnel_) из личного"
Write-Host "   кабинета https://fxtun.ru - обычным Ctrl+C."
Write-Host "2. Вернитесь в это окно и просто нажмите Enter."
Write-Host "   Вставлять ничего не нужно: токен возьмётся из буфера обмена."
Write-Host ""
Read-Host "Скопировали? Нажмите Enter" | Out-Null
$Host.UI.RawUI.FlushInputBuffer()

$token = ((Get-Clipboard -Raw) -replace '\s', '')
if ($token -notmatch '^sk_fxtunnel_[A-Za-z0-9]+$') {
    Finish "В буфере обмена нет токена fxTunnel. Скопируйте его целиком и запустите этот файл ещё раз."
    exit 1
}
& $client login -t $token | Out-Null
& $client domains list *> $null
if ($LASTEXITCODE -eq 0) {
    Finish "Готово: токен сохранён и принят сервером fxTunnel. Можно вернуться в Claude."
    exit 0
}
Finish "Сервер fxTunnel не принял токен. Проверьте его в личном кабинете и запустите файл ещё раз."
exit 1
