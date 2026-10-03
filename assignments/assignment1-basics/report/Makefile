LATEXMK ?= latexmk

.PHONY: all clean

all: main.pdf

main.pdf: main.tex problem_walkthrough.tex
	$(LATEXMK) -xelatex -interaction=nonstopmode -halt-on-error main.tex

clean:
	$(LATEXMK) -C main.tex
