LATEXMK ?= latexmk

.PHONY: all clean

all: writeup.pdf

writeup.pdf: main.tex
	$(LATEXMK) -xelatex -interaction=nonstopmode -halt-on-error -jobname=writeup main.tex

clean:
	$(LATEXMK) -C -jobname=writeup main.tex
