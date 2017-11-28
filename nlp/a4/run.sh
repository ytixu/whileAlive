#!/bin/bash

python sumbasic.py orig docs/doc1-*.txt > result.txt
python sumbasic.py simplified docs/doc1-*.txt >> result.txt
python sumbasic.py leading docs/doc1-*.txt >> result.txt

python sumbasic.py orig docs/doc2-*.txt >> result.txt
python sumbasic.py simplified docs/doc2-*.txt >> result.txt
python sumbasic.py leading docs/doc2-*.txt >> result.txt

python sumbasic.py orig docs/doc3-*.txt >> result.txt
python sumbasic.py simplified docs/doc3-*.txt >> result.txt
python sumbasic.py leading docs/doc3-*.txt >> result.txt

python sumbasic.py orig docs/doc4-*.txt >> result.txt
python sumbasic.py simplified docs/doc4-*.txt >> result.txt
python sumbasic.py leading docs/doc4-*.txt >> result.txt

