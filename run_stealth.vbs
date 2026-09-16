' Stealth launcher for Multimodal AI Assistant
' Runs with pythonw.exe: No console window, No taskbar button, No Alt+Tab entry
Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
WshShell.Run "pythonw.exe main.py", 0, False
