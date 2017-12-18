#!/bin/bash

python sumbasic.py orig docs/doc1-*.txt > orig-1.txt
python sumbasic.py simplified docs/doc1-*.txt > simplified-1.txt
python sumbasic.py leading docs/doc1-*.txt > leading-1.txt

python sumbasic.py orig docs/doc2-*.txt > orig-2.txt
python sumbasic.py simplified docs/doc2-*.txt > simplified-2.txt
python sumbasic.py leading docs/doc2-*.txt > leading-2.txt

python sumbasic.py orig docs/doc3-*.txt > orig-3.txt
python sumbasic.py simplified docs/doc3-*.txt > simplified-3.txt
python sumbasic.py leading docs/doc3-*.txt > leading-3.txt

python sumbasic.py orig docs/doc4-*.txt > orig-4.txt
python sumbasic.py simplified docs/doc4-*.txt > simplified-4.txt
python sumbasic.py leading docs/doc4-*.txt > leading-4.txt

