%# Copyright (c) 2023-2026 Christoph P. Neumann

% !TeX spellcheck = en-US
% LTeX: language=en-US
% !TeX encoding = utf8
% !TeX program = pdflatex
% !BIB program = biblatex / biber

% DISCLAIMER: Only for latex/pdflatex in combination with biblatex using a
% biber backend. (And neither for XeLaTeX or LuaTeX nor for bibtex or natbib.)
% Because to set the required bibliography configuration, the biblatex
% package needs to be loaded.
% I also need to adjust formatting of ONLINE entries, thus, I have to make
% some assumptions to rely on. And inputenc must be loaded before biblatex.
% Thus, inputenc, fontenc and biblatex with backend=biber are hard coded part
% of this template.
% I could remove these from the cls and make this part of the example.tex,
% but then it would be less foolproof in my opinion.
% I apologize for this opinionated design decission to anyone who is
% jeopardised by it.
% 
% I also had to decide whether to hardcode required usage of IEEEtran
% "conference" option into my class. I decided against it.
% Thus, it is required to use IARIA class like this:
% \documentclass[conference]{iaria}.
% The iaria class passes all class options (like "conference") to IEEEtran.
% The reasoning: There might be a future use case (speculatively) to pass
% IEEEtran options like "a4paper", for IARIA purposes (?)
% For now, IARIA requires IEEE "conference" option, only.

\NeedsTeXFormat{LaTeX2e}

\ProvidesClass{iaria}[2026/02/24 IARIA template v0.9.1]

% Define marker counter for options (counter are global in latex):
\newcounter{iaria@subfigtracker}
\newcounter{iaria@subcaptiontracker}
\newcounter{iaria@pbalancetracker}
\newcounter{iaria@flushendtracker}

% Declare options:

\DeclareOption{subfig}{
  \setcounter{iaria@subfigtracker}{1}
  \typeout{iaria class: subfig option is set (use subfloat command).}
}
\DeclareOption{subcaption}{
  \setcounter{iaria@subcaptiontracker}{1}
  \typeout{iaria class: subcaption option is set (use subfigure command).}
}

% IEEE option onecolumn will be ignored
\DeclareOption{onecolumn}{\OptionNotUsed}

\DeclareOption{pbalance}{
  \setcounter{iaria@pbalancetracker}{1}
  \typeout{iaria class: pbalance option is set.}
}
\DeclareOption{flushend}{
  \setcounter{iaria@flushendtracker}{1}
  \typeout{iaria class: flushend option is set.}
}

% We'll pass any document class options along to the underlying class
% (required IEEE options: conference! For IARIA purposes.)
\DeclareOption*{%
  \PassOptionsToClass{\CurrentOption}{IEEEtran}
}

% Now we'll execute any options passed in
\ProcessOptions\relax

% Instead of defining each and every little detail
% required to create a new document class,
% you can base your class on an existing document class.
\LoadClass{IEEEtran}

% Now paste your code from the preamble here.
% Replace \usepackage with \RequirePackage.
% (The syntax for both commands is the same.)

\RequirePackage{times} % Times New Roman

% inputenc must be loaded before biblatex
\RequirePackage[utf8]{inputenc}

% https://tex.stackexchange.com/questions/664
% /why-should-i-use-RequirePackaget1fontenc
% Has to be loaded AFTER any font packages.
% See https://tex.stackexchange.com/a/2869/9075.
\RequirePackage[T1]{fontenc}

\RequirePackage{graphicx}
\RequirePackage{xcolor}

\RequirePackage{url}
\RequirePackage{orcidlink}
\RequirePackage{hyperref}
\hypersetup{hidelinks,unicode}

% For cite compression: activated by citestyle, which
% must be given at package load time, and is NOT supported
% later on by \ExecuteBibliographyOptions
\RequirePackage[
  style=ieee,
  citestyle=ieee-comp, % by default, activate cite compression
  backend=biber
]{biblatex}
% TODO: numeric-comp results in "[7-9]" and ieee-comp in "[7]-[9]". IARIA requires IEEE-style cite compression, i.e., ieee-comp!
% However, ieee-comp is relatively new! From the perspective of Overleaf, it is only available with Overlead TeXlive 2025 as compiler setting. If you need a lower texlive version, for whichever reason, downgrade to numeric-comp, and usually you will still get it accepted (but then, also use \multiciterangedelim, below)

% Diferring from IEEE, IARIA requires 
%   in bibliography: 6+ authors as first author et al.
%   in \citeauthor: any case of et al. with just first author et al.
% as preamble options of biblatex that may be given after load-time:
\ExecuteBibliographyOptions{
  defernumbers=true,
  hyperref,
  maxnames=5,
  minnames=1,
  maxcitenames=2,
  mincitenames=1,
  %%% IARIA requires format for "Accessed: " dates in YYYY-MM-DD
  %urldate=iso8601, % DEPRECATED, use iso
  urldate=iso, % iso requires also seconds=true
  seconds=true
}

% Due to applied option defernumbers=true:
% Prevent citations in the references are being numbered as '0' in ArXiv
% https://tex.stackexchange.com/questions/463556
% /all-the-citations-in-the-references-are-being-numbered-as-0-in-arxiv-while-upl
\makeatletter
\let\blx@rerun@biber\relax
\makeatother

% When using multiple citation keys in the same \cite command,
% change IEEEtran "[1], [5], [7]" to IARIA-required "[1][5][7]":
%\renewcommand*{\multicitedelim}{][} % for numeric-comp
\renewcommand*{\multicitedelim}{}    % for ieee-comp
%\renewcommand*{\multiciterangedelim}{]--[} % only required for numeric-comp, not for ieee-comp

\renewcommand*{\bibfont}{\small}

% Zotero generates wrong language fields with wrong "en"
% instead of correct "english" option, thus, despite
% default biblatex clearlang, the "en" will incorrectly
% be printed out, even in english papers.
% Make "en" equivalent to "english" and/or its derivatives:
\DeclareRedundantLanguages{en}{english,american,british,
  canadian,australian,newzealand,USenglish,UKenglish}

% Diferring from IEEE, IARIA requires, for ONLINE entries,
% no point but a comma between author and title:
% For style=ieee
\DeclareBibliographyDriver{online}{%
  \usebibmacro{bibindex}%
  \usebibmacro{begentry}%
  \usebibmacro{author/editor+others/translator+others}
  \setunit{\labelnamepunct}\newblock
  \usebibmacro{title}%
  \newunit
  \printlist{language}%
  \newunit\newblock
  \usebibmacro{byauthor}%
  \newunit\newblock
  \usebibmacro{byeditor+others}%
  \newunit\newblock
  \printfield{version}%
  \newunit
  \printfield{note}%
  \newunit\newblock
  \printlist{organization}
  \newunit\newblock
  \usebibmacro{date}%
  \newunit\newblock
  \iftoggle{bbx:eprint}
  {\usebibmacro{eprint}}
  {}%
  \newunit\newblock
  \usebibmacro{url+urldate}%
  \newunit\newblock
  \usebibmacro{addendum+pubstate}%
  \setunit{\bibpagerefpunct}\newblock
  \usebibmacro{pageref}%
  \usebibmacro{finentry}}

\makeatletter
% Diferring from IEEE, IARIA requires non-abbreviated references:
\renewcommand{\fnum@figure}{Figure~\thefigure}
% Diferring from IEEE, IARIA requires 14 point bold Times for the title
\renewcommand{\title}[1]{\renewcommand{\@title}{\bfseries\Large #1}}
% Diferring from IEEE, IARIA requires "Keywords" instead of "Index Terms":
\renewcommand\IEEEkeywordsname{Keywords}
\makeatother

\RequirePackage{xpatch}
% Diferring from IEEE, IARIA requires a hyphen after "Keywords"
% instead of an  emdash:
\xpatchcmd\IEEEkeywords{---}{-}{}{}

% Diferring from IEEE, IARIA requires also the keywords in Italic
% (and Bold)(and lower case!):
\let\oldIEEEkeywords\IEEEkeywords
\def\IEEEkeywords{\oldIEEEkeywords\itshape\ignorespaces}

\makeatletter
\renewcommand{\IEEEauthorblockN}[1]{%
% copied from IEEEtran.cls:
    \relax\@IEEEauthorblockNstyle% set the default text style
    \gdef\@IEEEauthorblockXinterlinespace{0pt}% disable strut for spacer row
    \if@IEEEprevauthorblockincol\expandafter\@IEEEauthorblockNtopspaceline\fi
    \global\@IEEEprevauthorblockincoltrue% we now have a block in this column
    \gdef\@IEEEauthorblockXinterlinespace{\@IEEEauthorblockNinterlinespace}%
    \large
    #1%
    \crcr}
\makeatother

% IARIA requires to bring the table caption
% in the same line as label "TABLE I"
\usepackage{etoolbox}
\makeatletter
\patchcmd{\@makecaption}
  {\\}
  {.\ }
  {}
  {}
\makeatother

% caption is compatible with babel that intervenes with tablename
\RequirePackage{caption}
% IARIA requires table caption to be
% above, centered, with upper case label ('TABLE'), and small caps text.
% (note that the caption package is only controlling the look & feel of the captions,
%  it does not control the placement of the captions.)
\captionsetup[table]{
  style=base,
  name=TABLE,
  position=auto, % position option is NOT about forcing the label above/below, but ONLY about spacing!
  justification=centering,
  labelsep=period,
  textfont={footnotesize, sc},
  labelfont=footnotesize}
% IARIA requires figure caption to be
% below and centered 
\captionsetup[figure]{
  style=base,
  position=auto, % position option is NOT about forcing the label above/below, but ONLY about spacing!
  justification=centering,
  labelsep=period,
  font={footnotesize,rm}}
% For controlling the placement of the captions
% (table: top | figure: bottom), the package caption
% recommends to use package floatrow:
% (Without foatrom, the positioning is determined by
%  the textual (!) position of command \label in relation to
%  includegraphics/begin-tabular; a well-known latex aspect.)
\RequirePackage{floatrow}
\floatsetup[table]{capposition=top,captionskip=5pt}
\floatsetup[figure]{capposition=bottom,captionskip=8pt}

% IARIA subfigure captions require parentheses and footnotesize:
\ifnum\value{iaria@subfigtracker}=1
  \typeout{iaria class: load subfig package (use subfloat command).}
  \RequirePackage[caption=false,font=footnotesize]{subfig}
\fi
\ifnum\value{iaria@subcaptiontracker}=1
  \typeout{iaria class: load subcaption package (use subfigure command).}
  \RequirePackage{subcaption}
  % IARIA requires subfigure caption to number them with
  % parentheses around latin chars (like "(a)" and "(b)")
  \captionsetup[subfigure]{labelformat=parens,font=footnotesize}
\fi

% Balance/level columns at the last page / in bibliography
% DISCLAIMER: No solution is a 100%/perfect one!
% Mainstream option: pbalance (see https://ctan.org/pkg/pbalance)
% BUT: incompatible with arxiv automated process (!)
\ifnum\value{iaria@pbalancetracker}=1
  \typeout{iaria class: load pbalance package.}
  \RequirePackage{pbalance}
\fi
% Flushend => works with arxiv, but major incampatibility with lineno
% (and, thus, also with mindflow et al.)
\ifnum\value{iaria@flushendtracker}=1
  \typeout{iaria class: load flushend package.}
  \RequirePackage{flushend}
\fi

% ========================================================
% Packages with recommendation to load them last:

% Allows to hyphenate a word that contains a dash:
% https://stackoverflow.com/questions/2193307
% /how-do-i-get-latex-to-hyphenate-a-word-that-contains-a-dash
% Use \-/ for a breakable dash that does not prevent the
% remainer of the word to be hyphenated.
\RequirePackage[shortcuts]{extdash}
% ========================================================

% Finally, we'll use \endinput to indicate that LaTeX can stop reading
% this file. LaTeX will ignore anything after this line.
\endinput
