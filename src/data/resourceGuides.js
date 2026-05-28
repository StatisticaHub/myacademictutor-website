export const guidePages = {
  "p-values-confidence-intervals-effect-sizes": {
    title: "Understanding p-values, confidence intervals and effect sizes",
    tag: "Statistics",
    level: "Intermediate",
    description:
      "A clearer explanation of statistical significance, practical importance and how to report results responsibly.",
    intro:
      "Students often report results as simply significant or not significant. A stronger interpretation explains the p-value, the effect estimate, the confidence interval and whether the result matters in context.",
    keyTerms: [
      {
        term: "p-value",
        meaning:
          "Helps assess how incompatible the observed data are with a null model or hypothesis.",
      },
      {
        term: "Confidence interval",
        meaning:
          "Shows a range of plausible values for the effect estimate and gives a sense of precision.",
      },
      {
        term: "Effect size",
        meaning:
          "Describes how large or meaningful a difference, association or effect is.",
      },
    ],
    sections: [
      {
        heading: "Do not stop at the p-value",
        body:
          "A small p-value may suggest evidence against a null hypothesis, but it does not show whether the effect is large, important or useful. A large p-value also does not prove that there is no effect.",
      },
      {
        heading: "Use confidence intervals to discuss uncertainty",
        body:
          "A confidence interval helps you discuss precision. A narrow interval suggests a more precise estimate, while a wide interval suggests more uncertainty.",
      },
      {
        heading: "Effect size answers a different question",
        body:
          "Effect size helps answer whether the result is meaningful. Depending on the analysis, this could be a mean difference, odds ratio, risk ratio, hazard ratio, correlation or standardised effect.",
      },
      {
        heading: "Better reporting",
        body:
          "A strong result sentence includes the estimate, confidence interval, p-value and plain-language interpretation. Avoid reporting only whether p < 0.05.",
      },
    ],
    checklist: [
      "Report the estimate, not only the p-value.",
      "Include a confidence interval where appropriate.",
      "Explain the direction and size of the effect.",
      "Discuss practical, clinical or academic importance.",
      "Avoid saying p > 0.05 proves no effect.",
      "Avoid saying p < 0.05 proves importance.",
    ],
  },
};