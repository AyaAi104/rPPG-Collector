.PHONY: clean

go:
	python main.py

clean:
	@echo I am cleaning raw ppg signals...
	@powershell -Command "if (Test-Path './data/rawsignal') { Remove-Item -Path './data/rawsignal' -Recurse -Force }"
	@powershell -Command "New-Item -ItemType Directory -Path './data/rawsignal' -Force | Out-Null"
	@echo Clean completed!

cleanvideos:
	@echo I am cleaning videos contents...
	@powershell -Command "Remove-Item -Path './data/video/*' -Recurse -Force"
	@echo Clean completed!

empty:
	@echo I am cleaning all datas...
	@powershell -Command "Remove-Item -Path './data/video/*' -Recurse -Force"
	@powershell -Command "Remove-Item -Path './data/rawsignal/*' -Recurse -Force"
	@echo Clean completed!

analyze:
	python ppg_processor.py