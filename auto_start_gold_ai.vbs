Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "C:\Users\OWNER\Downloads\pembuatan aplikasi"
WshShell.Run "python -B app.py", 0, False
