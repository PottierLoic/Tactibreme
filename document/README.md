## Local latex !!

### Linux
> install package `pdflatex` (for compiling latex) and `evince`(for displaying pdfs)
> go to the folder with the `.tex` file and compile it with `pdflatex -shell-escape XXX.tex --halt-on-error`.
> open your compiled `.pdf` file with `evince main.pdf &`.

### Mac

Install MacTex using homebrew

```bash
brew cask install mactex
```

Then simply compile the `.tex` file using `pdflatex` and open the `.pdf` file using `open`

```bash
pdflatex -shell-escape XXX.tex --halt-on-error
open XXX.pdf
```

### Windows

Simply install [MiKTeX](https://miktex.org/download) from the website or using winget

```bash
winget install MiKTeX.MiKTeX
```

Then compile the `.tex` file using `pdflatex` and open the `.pdf` file using `start`

```bash
pdflatex -shell-escape XXX.tex --halt-on-error
start XXX.pdf
```