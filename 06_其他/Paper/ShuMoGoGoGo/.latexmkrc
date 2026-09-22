# This template requires XeLaTeX; latexmk repeats passes for cross-references.
$pdf_mode = 5;
$out_dir = 'build';
$xelatex = 'xelatex -synctex=1 -interaction=nonstopmode -file-line-error -halt-on-error %O %S';

# On this Linux host, /usr/local/lib/libz produces truncated TeX format files.
# Prefer the distribution libraries for this build and its child processes.
if ($^O eq 'linux' && -f '/usr/lib/x86_64-linux-gnu/libz.so.1') {
    $ENV{'LD_LIBRARY_PATH'} = '/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu'
        . (defined $ENV{'LD_LIBRARY_PATH'} ? ':' . $ENV{'LD_LIBRARY_PATH'} : '');
}
