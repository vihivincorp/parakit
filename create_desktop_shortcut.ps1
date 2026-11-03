$WshShell = New-Object -ComObject WScript.Shell
$DesktopPath = [System.Environment]::GetFolderPath([System.Environment+SpecialFolder]::Desktop)
$ShortcutPath = Join-Path -Path $DesktopPath -ChildPath "Multimodal AI Assistant.lnk"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = "wscript.exe"
$Shortcut.Arguments = "`"$ScriptDir\start_background.vbs`""
$Shortcut.WorkingDirectory = $ScriptDir
$Shortcut.Description = "Launch Multimodal AI Desktop Assistant in Background"
$Shortcut.Save()

Write-Host "Desktop shortcut created at: $ShortcutPath"

