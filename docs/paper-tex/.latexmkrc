# latexmk configuration for Korean document compilation
# Optimized for XeLaTeX with Korean font support

# Use XeLaTeX as the PDF engine
$pdf_mode = 5;  # 5 = xelatex

# XeLaTeX command with shell-escape for minted and other packages
$xelatex = 'xelatex -shell-escape -synctex=1 -interaction=nonstopmode -file-line-error %O %S';

# Output directory
$out_dir = 'output';

# Biber for bibliography (if using biblatex)
$biber = 'biber --input-directory=output %O %S';
$bibtex_use = 2;  # Use biber instead of bibtex

# Files to clean
$clean_ext = 'bbl nav snm vrb synctex.gz run.xml bcf fls';

# Additional extensions for distclean
$clean_full_ext = 'pdf';

# Automatically continue on errors
$force_mode = 1;

# Preview mode settings (for -pvc option)
$preview_mode = 1;
$pdf_previewer = 'start %O %S';  # Windows default PDF viewer

# Generate PDF with maximum optimization
$postscript_mode = 0;
$dvi_mode = 0;

# Enable recorder mode for better dependency tracking
$recorder = 1;

# Ensure output directory exists
system("mkdir -p output") if ! -d "output";
