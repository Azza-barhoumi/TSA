// Presentation deck for the Box--Jenkins CPI project.
// Compiled by src/08_build_slides.py, which substitutes every __KEY__ placeholder.
#set document(title: "Forecasting US Inflation — Box–Jenkins")
// A fixed page height of 11.2 cm is the design constraint that makes the deck a
// deck: no slide can spill onto a second page, because anything that overflows
// is clipped by the page rather than flowing.  Width stays at 19.86 cm so the
// aspect ratio is 16:9.
#set page(
  paper: "presentation-16-9",
  height: 11.2cm,
  margin: (x: 0.78cm, top: 0.5cm, bottom: 1.0cm),
  numbering: none,
)
#set text(font: "New Computer Modern", size: 13.5pt, lang: "en")
#set par(justify: false, leading: 0.58em)

// ---- palette: identical to the report's matplotlib style -------------------
#let ink = rgb("#1f3864")      // data / headings
#let red = rgb("#c00000")      // model / emphasis
#let blue = rgb("#2e75b6")     // accent
#let grey = rgb("#7f7f7f")     // benchmarks
#let pale = rgb("#eef1f6")     // panels
#show figure.caption: set text(size: 8.5pt, fill: grey)

// ---- slide furniture -------------------------------------------------------
#let slide(title, body, tag: none) = {
  pagebreak(weak: true)
  set page(header: none)
  set text(size: 13pt)
  block(width: 100%, height: 100%)[
    #set align(left + top)
    #body
  ]
}

#let kicker(t, right: none) = {
  set text(size: 10.5pt, fill: red, weight: "bold")
  grid(
    columns: (1fr, auto),
    upper(t),
    if right != none { text(size: 6.8pt, fill: grey, weight: "regular")[upper(right)] },
  )
  v(0.05em)
}

#let h1(t) = {
  set text(size: 18pt, fill: ink, weight: "bold")
  t
  v(0.22em)
  line(length: 100%, stroke: 1.2pt + red)
  v(0.40em)
}

#let h2(t) = {
  set text(size: 12pt, fill: ink, weight: "bold")
  t
  v(0.45em)
}

#let bullets(..items) = {
  set text(size: 12.5pt)
  for it in items.pos() {
    grid(
      columns: (0.62em, 1fr),
      gutter: 0.5em,
      align: (left + top, left + top),
      text(fill: red, weight: "bold")[–],
      it,
    )
    v(0.42em)
  }
}

#let stat(big, label, col: red, size: 15.0pt) = block(
  fill: pale, inset: (x: 0.5em, y: 0.26em), radius: 3pt, width: 100%,
)[
  #set align(center)
  #text(size: size, weight: "bold", fill: col)[#big]
  #v(0.02em)
  #text(size: 8.5pt, fill: grey)[#label]
]

#let note(t) = {
  set text(size: 8.5pt, fill: grey, style: "italic")
  v(0.5em)
  t
}

#let fig(path, h: 6.4cm, cap: none) = {
  figure(
    image(path, height: h),
    caption: cap,
    kind: image,
    gap: 0.25em,
  )
}

// ===========================================================================
// 1 — title
// ===========================================================================
// The title slide is a full-bleed panel: page geometry is overridden for this
// one page so the block cannot be broken across a page boundary.
#[
  #set page(fill: ink, margin: 0cm, header: none)
  #set text(fill: white, size: 8.2pt)
  #block(width: 100%, height: 100%)[
  #align(center + top)[
    #v(1.05em)
    #text(size: 10.5pt, fill: rgb("#9fb4d6"))[TUNIS BUSINESS SCHOOL — TIME SERIES ANALYSIS AND FORECASTING]
    #v(0.6em)
    #line(length: 58%, stroke: 1pt + red)
    #v(0.55em)
    #text(size: 19.2pt, weight: "bold")[Forecasting US Inflation]
    #v(0.30em)
    #text(size: 13.5pt, fill: rgb("#d6deea"))[A Box–Jenkins analysis of the Kaggle CPI-U dataset]
    #v(0.7em)
    #line(length: 58%, stroke: 1pt + red)
    #v(0.85em)
    #block(width: 78%)[
    #text(size: 8.2pt)[
      #grid(
        columns: (auto, 1fr),
        gutter: 1.0em,
        align: (right, left),
        [*Question*], [One- to five-year ahead inflation, and how wide the band],
        [*Data*], [1,296 monthly CPI-U observations, 1913–2020],
        [*Model*], [ARIMA(1,1,3)(1,0,1)[12] + constant],
        [*Result*], [71\% lower forecast error than a random walk],
      )
    ]
    ]
  ]
  ]
]

// ===========================================================================
// 2 — the decision
// ===========================================================================
#slide("The decision this answers")[
  #kicker("Why an inflation forecast matters")
  #h1[One number a manager has to commit to]

  #grid(
    columns: (1.15fr, 1fr),
    gutter: 1.4em,
    [
      #text(size: 12.6pt, style: "italic", fill: ink)[
        "What US consumer price inflation should a Tunisian importer, treasurer or
        bank budget for over the next one to five years — and how wide is the band
        within which the realised rate should fall?"
      ]
      #v(1.0em)
      #bullets(
        [A five-year dollar loan needs a *nominal* compensation for expected US inflation.],
        [An indexation clause needs a forecast *and* a tolerance.],
        [Both are decision-relevant only with a stated uncertainty.],
      )
    ],
    [
      #v(1.1em)
#stat("2.9%", "long-run annual inflation implied by the model")
      #v(0.5em)
#stat("±1.8 ⇒ ±8.6", "95% band, in points, at 1 year ⇒ at 5 years", size: 13.2pt)
      #v(0.5em)
#stat("16.5", "months: half-life of an inflation shock", col: blue)
    ],
  )
]

// ===========================================================================
// 3 — data
// ===========================================================================
#slide("Data and sample")[
  #kicker("Answering the brief: source, frequency, period, suitability")
  #h1[A real series, 1,296 monthly observations]

  #grid(
    columns: (1fr, 1fr),
    gutter: 1.6em,
    [
      #bullets(
        [*Source.* Kaggle "US Inflation Data" — CPI-U, US city average, all items.],
        [*Frequency.* Monthly. *Coverage.* 1913-01 to 2020-12, no gaps, no duplicates.],
        [*Integrity.* Raw CSV verified against its SHA-256 digest on every run.],
      )
    ],
    [
      #bullets(
        [*Estimation sample.* 1957-01 to 2020-12, #text(fill: red, weight: "bold")[n = 768].],
        [*Hold-out.* 2016-01 to 2020-12, #text(fill: red, weight: "bold")[h = 60] — never seen in fitting.],
        [*Why 1957?* The pre-war deflations are a different regime; a contract written today will not meet them.],
      )
    ],
  )

  #v(0.6em)
  #fig("figures/f01_series_components.png", cap: [Level, log level, and inflation. The level trends; inflation does not.])
]

// ===========================================================================
// 4 — regimes
// ===========================================================================
#slide("The series is not stationary")[
  #kicker("Chapter 1 — components")
  #h1[A sequence of regimes, not a fixed mean]

  #grid(
    columns: (1.25fr, 1fr),
    gutter: 1.5em,
    [
      #fig("figures/f17_decades.png", cap: [Annual inflation by decade. The dispersion changes too.])
    ],
    [
      #v(0.4em)
      #text(size: 8.8pt)[Annual inflation averaged]
      #v(0.5em)
      #grid(
        columns: (auto, auto),
        gutter: (0.9em, 0.25em),
        align: (left, right),
        [1960s], [2.3%],
        [1970s], [6.8%],
        [1980s], [5.4%],
        [1990s], [2.95%],
        [2000s], [2.5%],
        [2010s], [1.75%],
      )
      #v(0.8em)
      #note[Volcker: 13.8% (Mar 1980) → 3.8% (end 1982). That single episode dominates the sample mean — and the residuals.]
    ],
  )
]

// ===========================================================================
// 5 — Box-Cox + differencing
// ===========================================================================
#slide("Transformation and differencing")[
  #kicker("Chapters 2–3 — variance stabilisation, unit roots")
  #h1[Log, then one difference]

  #grid(
    columns: (1fr, 1fr),
    gutter: 1.4em,
    [
      #fig("figures/f18_boxcox_slide.png", h: 6.0cm)
      #text(size: 8.2pt)[λ̂ = *0.42*, 95#text(size: 6.8pt)[%] interval [0.31, 0.54].]
    ],
    [
      #v(0.2em)
      #bullets(
        [λ = 0 lies *outside* the interval — stated honestly, not hidden.],
        [The likelihood is flat: conclusions are unchanged under log or square root.],
        [The log is kept because it turns the index into a *growth rate* — what a manager budgets for.],
      )
      #v(0.3em)
#stat("d = 1", "ADF: unit root not rejected; KPSS: rejected", size: 12.6pt)
      #v(0.4em)
#stat("D = 0", [a real annual echo, but *not* a seasonal unit root], size: 15.6pt, col: blue)
    ],
  )
]

// ===========================================================================
// 6 — ACF / PACF
// ===========================================================================
#slide("Identification")[
  #kicker("Chapter 6 — reading the correlogram")
  #h1[An annual echo at lags 11–13]

  #fig("figures/f04_acf_pacf.png", h: 6.3cm, cap: [ACF and PACF of the log level and of its first difference.])

  #grid(
    columns: (1fr, 1fr),
    gutter: 1.4em,
    [
      #text(size: 12.5pt)[The log level's ACF decays from 0.997 to 0.965 at lag 12 — a non-stationary signature.]
    ],
    [
      #text(size: 12.5pt)[After differencing: spikes of 0.42 / 0.45 / 0.34 at lags 11 / 12 / 13, far outside the ±0.072 band.]
    ],
  )
  #note[That echo is why a purely non-seasonal ARIMA fails.]
]

// ===========================================================================
// 7 — the grid
// ===========================================================================
#slide("Model selection")[
  #kicker("Chapters 6–7 — multiplicative SARMA")
  #h1[63 specifications, filtered on white noise]

  #grid(
    columns: (1fr, 1.05fr),
    gutter: 1.5em,
    [
      #fig("figures/f06_model_selection.png", cap: [Best models by BIC, in both families.])
    ],
    [
      #v(0.5em)
      #bullets(
        [Grid over (p,q) ∈ {0..3}², (P,Q) ∈ {(0,0),(1,0),(0,1),(1,1)}, d = 1, D = 0.],
        [Estimated by Gaussian ML through the Kalman filter.],
        [*54 of the 63* fail Ljung–Box at 12 lags.],
        [BIC alone would pick a model whose residuals are not white noise.],
      )
      #v(0.4em)
#stat("ARIMA(1,1,3)(1,0,1)[12]", "selected: lowest BIC among models that pass", size: 11.4pt)
    ],
  )
]

// ===========================================================================
// 8 — the seasonal block matters
// ===========================================================================
#slide("Why the seasonal block is indispensable")[
  #kicker("Chapter 7 — like-for-like comparison")
  #h1[Remove Θ(B¹²) and the echo returns]

  #fig("figures/f07_seasonal_block.png", h: 6.3cm, cap: [Residual ACF: same orders, with and without the seasonal MA term.])

  #grid(
    columns: (1fr, 1fr),
    gutter: 1.4em,
    [
      #text(size: 12.5pt)[*Without* the seasonal block: residual ACF at lag 12 is *+0.213*, far outside ±0.071.]
    ],
    [
      #text(size: 12.5pt)[*With* it: *−0.018*. Dropping the seasonal AR instead raises BIC from 163.9 to 233.9.]
    ],
  )
]

// ===========================================================================
// 9 — three estimation methods
// ===========================================================================
#slide("The three estimation methods")[
  #kicker("Chapter 4 — MoM, least squares, maximum likelihood")
  #h1[They agree — except where the method breaks down]

  #grid(
    columns: (1.05fr, 1fr),
    gutter: 1.5em,
    [
      #text(size: 8.8pt)[All three methods are applied to the *same* series, π#sub[t].]
      #v(0.7em)
      #grid(
        columns: (auto, auto, auto),
        gutter: (0.8em, 0.35em),
        align: (left, right, right),
        [], [#text(size: 8.2pt, fill: grey)[MA(1) θ]], [#text(size: 8.2pt, fill: grey)[ARMA(1,1) φ]],
        [Moments], [*fails*], [0.556],
        [Cond. LS], [0.411], [0.719],
        [Hannan–Rissanen], [—], [0.739],
        [Exact ML], [0.411], [0.718],
      )
      #v(0.9em)
      #text(size: 8.2pt)[Conditional and exact likelihood differ only in the first few observations, so for n = 767 they are interchangeable.]
    ],
    [
      #v(0.3em)
      #h2[Where the method of moments fails]
      #text(size: 12.5pt)[For an MA(1), ρ₁ = θ/(1+θ²) inverts only if |ρ₁| ≤ ½.]
      #v(0.6em)
#stat("−0.763", "discriminant of the moment equation (needs ≥ 0)", size: 12.6pt)
      #v(0.5em)
      #text(size: 12.5pt)[Here ρ₁ = *0.556* > ½, so the moment estimator *does not exist*. Not a numerical accident — an infeasibility, and the reason the course moves on to least squares and ML.]
    ],
  )
]

// 10 — parameters
// ===========================================================================
#slide("The fitted model")[
  #kicker("Chapter 6 — estimation")
  #h1[Eight parameters, seven significant at 1%]

  #text(size: 13.5pt)[
    (1 − *0.9588*B)(1 − *0.9747*B¹²) Δy#sub[t] = *0.0003* + (1 − *−0.5823*B − *−0.1058*B² − *−0.1281*B³)(1 − *−0.8676*B¹²) ε#sub[t]
  ]
  #v(0.6em)

  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 0.7em,
stat("−55.36", "log-likelihood", size: 12.6pt),
stat("126.90", "AICc", size: 12.6pt),
stat("163.85", "BIC", size: 12.6pt),
stat("0.2586", "σ̂ (points/month)", size: 12.6pt),
  )
  #v(0.8em)
  #bullets(
    [Stationary and invertible: all inverse roots inside the unit circle.],
    [The constant is *not* individually significant (p = 0.39) but is *retained* — see slide 14.],
    [Residuals pass the portmanteau test: LB(12) p = 0.21, LB(24) p = 0.10.],
  )
]

// ===========================================================================
// 11 — diagnostics that pass
// ===========================================================================
#slide("Diagnostics that pass")[
  #kicker("Chapter 6 — stage 3 of the cycle")
  #h1[No mean bias, no variance drift, no residual autocorrelation]

  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [
      #fig("figures/f15_residual_acf_pacf.png", cap: [Residual ACF and PACF inside the Bartlett bands.])
    ],
    [
      #v(0.3em)
      #grid(
        columns: (1fr, 1fr),
        gutter: 0.7em,
stat("p = 0.985", "zero mean", size: 13.2pt, col: blue),
stat("p = 0.963", "constant variance", size: 13.2pt, col: blue),
      )
      #v(0.6em)
      #grid(
        columns: (1fr, 1fr),
        gutter: 0.7em,
stat("p = 0.21", "Ljung–Box (12)", size: 10.8pt),
stat("p = 0.12", "Box–Pierce (24)", size: 10.8pt),
      )
      #v(0.8em)
      #note[The decisive check passes: the model captures the linear dependence.]
    ],
  )
]

// ===========================================================================
// 12 — diagnostics that fail
// ===========================================================================
#slide("Diagnostics that fail — stated plainly")[
  #kicker("Honest limitations")
  #h1[Two assumptions do not hold]

  #grid(
    columns: (1fr, 1fr),
    gutter: 1.6em,
    [
      #h2[Leptokurtic residuals]
#stat("4.1", "excess kurtosis (0 for a Gaussian)", size: 13.2pt)
      #v(0.5em)
      #text(size: 12.5pt)[Jarque–Bera p ≈ 10⁻¹¹⁹. The distribution has fat tails, not a bell shape.]
      #v(0.6em)
      #text(size: 12.5pt)[*Consequence:* Gaussian intervals are too narrow in 1973–74 and 2008–09.]
    ],
    [
      #h2[Conditional heteroskedasticity]
#stat("49.6", "ARCH-LM(12) statistic", size: 18.0pt, col: blue)
      #v(0.5em)
      #text(size: 12.5pt)[LB on squared residuals = 76.4, p < 10⁻¹⁰. Volatility clusters.]
      #v(0.6em)
      #text(size: 12.5pt)[*Consequence:* point forecasts are unaffected; *interval width* is.]
    ],
  )
  #v(0.9em)
  #note[Both are outside Chapters 1–7. They are reported as limitations, not modelled — a GARCH extension would be the next step.]
]

// ===========================================================================
// 13 — forecast fanchart
// ===========================================================================
#slide("Forecasting")[
  #kicker("Chapter 5 — MMSE forecasts and their error variance")
  #h1[The band widens more slowly than a random walk's]

  #fig("figures/f09_forecast_fanchart.png", h: 6.3cm, cap: [60-month hold-out. The 80% and 95% bands are derived from ψ-weights, not asserted.])

  #grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 0.7em,
stat("0.261", "±1σ at h = 1", size: 11.4pt),
stat("1.8141", "±1σ at h = 12", size: 11.4pt),
stat("8.6154", "±1σ at h = 60", size: 11.4pt),
  )
]

// ===========================================================================
// 14 — accuracy
// ===========================================================================
#slide("Out-of-sample accuracy")[
  #kicker("Chapters 5–6 — hold-out evaluation, 2016–2020")
  #h1[71% lower error than the naive benchmark]

  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [
      #fig("figures/f11_accuracy_comparison.png", cap: [RMSE on the log index against three benchmarks.])
    ],
    [
      #v(0.4em)
      #grid(
        columns: (auto, auto),
        gutter: (1.2em, 0.45em),
        align: (left, right),
        [*This model*], [#text(fill: red, weight: "bold")[#text(size: 12.6pt)[1.79]]],
        [Random walk], [6.23],
        [RW with drift], [4.49],
        [Seasonal RW], [1.86],
      )
      #v(0.9em)
#stat("71%", "skill score versus the random walk", size: 13.2pt)
      #v(0.5em)
      #text(size: 8.2pt)[Rolling one-step RMSE *0.225* against the theoretical σ = 0.261 — the model is correctly specified.]
    ],
  )
]

// ===========================================================================
// 15 — the constant
// ===========================================================================
#slide("A decision worth defending")[
  #kicker("Model choice, not p-value chasing")
  #h1[The constant is insignificant — and stays]

  #grid(
    columns: (1fr, 1fr),
    gutter: 1.4em,
    [
      #fig("figures/f13_annual_inflation_forecast.png", cap: [Annual inflation: forecast against realisation.])
    ],
    [
      #v(0.4em)
      #text(size: 8.8pt)[The constant's t-statistic gives p = 0.39, so BIC prefers the model without it.]
      #v(0.8em)
      #bullets(
        [Without it, the model's long-run drift is *exactly zero* — i.e. 0% inflation forever.],
        [On the hold-out, RMSE rises from *1.79* to *3.40*.],
        [After 60 months the two forecasts differ by *3.2 log points*.],
      )
      #v(0.3em)
#stat("1.79 vs 3.40", "hold-out RMSE, with vs without the constant", size: 10.8pt)
    ],
  )
]

// ===========================================================================
// 16 — interpretation
// ===========================================================================
#slide("What it means in money")[
  #kicker("Plain language for a decision-maker")
  #h1[From RMSE to a price]

  #grid(
    columns: (1fr, 1fr),
    gutter: 1.6em,
    [
      #h2[Pricing a 5-year dollar loan]
      #text(size: 8.8pt)[Budget *2.9%* expected inflation, and price at the *upper* part of the band — not the mean. The band is ±8.6 points at five years; the risk is asymmetric once you are lending.]
      #v(0.8em)
#stat("±1.8 → ±8.6", "95% band, 1 year → 5 years", size: 11.4pt)
    ],
    [
      #h2[Indexation clauses]
      #text(size: 8.8pt)[Annual-rate forecast error is *0.76* points. A tolerance of ±1.5 points covered *58 of 60* hold-out months; ±1.0 covered 47.]
      #v(0.8em)
#stat("58 / 60", "hold-out months inside ±1.5 points", size: 15.6pt, col: blue)
    ],
  )
  #v(0.7em)
  #note[On a \$10m five-year exposure, the RMSE difference (6.23 → 1.79 log points) is the difference between a ~\$640k and a ~\$180k typical pricing error.]
]

// ===========================================================================
// 17 — limitations / next
// ===========================================================================
#slide("Limitations and what comes next")[
  #kicker("Bounding how the result may be used")
  #h1[What this model cannot do]

  #bullets(
    [*No causal content.* It extrapolates the generating process; it cannot anticipate a policy change, a supply shock, or a pandemic.],
    [*Regime instability.* Three regimes in one sample; the largest residuals (1973-08, 2008-11) are exactly the breaks.],
    [*Interval width is too optimistic* in turbulent periods (ARCH), so a risk application should widen the bands.],
    [*Log transform is a convenience*, not a likelihood result — λ̂ = 0.42 excludes 0, but conclusions are unchanged either way.],
  )
  #v(0.4em)
  #note[Natural extensions: GARCH for the variance, Markov switching for the regimes. Both are deliberately outside the Chapters 1–7 scope.]
]

// ===========================================================================
// 18 — takeaway
// ===========================================================================
#slide("Three things to remember")[
  #kicker("Takeaway")
  #h1[The result in three lines]

  #v(0.4em)
  #grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 0.9em,
stat("2.9%", "expected annual US inflation over 1–5 years", size: 15.0pt),
stat("±8.6", "points of 95% uncertainty at five years", size: 20.4pt, col: blue),
stat("71%", "lower forecast error than a random walk", size: 15.0pt),
  )
  #v(1.0em)
  #bullets(
    [A parsimonious *ARIMA(1,1,3)(1,0,1)[12]* captures the persistence, the annual echo and the drift — and its residuals pass the portmanteau tests at 12 and 24 lags.],
    [The seasonal block is *not optional*: without it the lag-12 residual autocorrelation is +0.213 instead of −0.018.],
    [The remaining failure is in the *variance*, not the mean — reported, quantified, and the boundary of the syllabus.],
  )
]

// ===========================================================================
// 19 — closing
// ===========================================================================
#align(center)[
  #v(3.0em)
  #text(size: 20.4pt, weight: "bold", fill: ink)[Thank you]
  #v(1.0em)
  #line(length: 46%, stroke: 1.2pt + red)
  #v(1.2em)
  #text(size: 12.5pt)[
    Full report: #text(fill: red)[report/main.pdf] — 34 pages, 17 figures, 13 tables \
    Code: #text(fill: red)[run_all.py] reproduces everything in ~9 minutes
  ]
  #v(1.4em)
  #text(size: 8.2pt, fill: grey, style: "italic")[
    Cryer & Chan (2008), *Time Series Analysis*, Chapters 1–7.
  ]
]