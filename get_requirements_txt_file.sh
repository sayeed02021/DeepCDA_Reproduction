#!/usr/bin/env bash
pip install pipreqs
pip freeze > requirements_all.txt
pipreqs . --print | sed 's/==.*//' > used.txt
grep -Ff used.txt requirements_all.txt > requirements.txt
rm requirements_all.txt
rm used.txt