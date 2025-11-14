.PHONY: clean

go:
	python main.py

clean:
	@echo I am cleaning raw ppg signals...
	@powershell -Command "if (Test-Path './data/rawsignal') { Remove-Item -Path './data/rawsignal' -Recurse -Force }"
	@powershell -Command "New-Item -ItemType Directory -Path './data/rawsignal' -Force | Out-Null"
	@echo Clean completed!

analyze:
	python ppgprocessor.py