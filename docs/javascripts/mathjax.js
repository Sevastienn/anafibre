window.MathJax = {
  loader: {
    load: ["[tex]/physics"]
  },
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["$$", "$$"], ["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    packages: {
      "[+]": ["physics"]
    },
    macros: {
      ii: "\\mathrm{i}",
      ee: "\\mathrm{e}",
      vc: "\\vb*",
      uv: "\\vu*",
      bessel: ["\\operatorname{J}_{#1}", 1],
      neumann: ["\\operatorname{Y}_{#1}", 1],
      macdonald: ["\\operatorname{K}_{#1}", 1],
      hankel: ["\\operatorname{H}^{#1}_{#2}", 2, "(1)"],
      harmol: ["\\operatorname{Z}_{#1}", 1]
    }
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

if (window.document$ && window.document$.subscribe) {
  window.document$.subscribe(() => {
    if (window.MathJax && window.MathJax.startup) {
      window.MathJax.startup.output.clearCache();
      window.MathJax.typesetClear();
      window.MathJax.texReset();
      window.MathJax.typesetPromise();
    }
  });
}
