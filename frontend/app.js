const DEFAULT_URLS = [
  "https://www.ebay.com.sg/itm/206158794969",
  "https://www.ebay.com.sg/itm/236047233833",
];

const WAITING_STAGES = [
  {
    label: "Listing check",
    title: "Reading both listings",
    copy: "Confirming the item titles, seller details, and current asking prices.",
  },
  {
    label: "Listing trust",
    title: "Checking listing trust",
    copy: "Scoring the listing content for deception risk while the review branch is processed separately.",
  },
  {
    label: "Market value",
    title: "Sizing up market value",
    copy: "Comparing each item against similar listings to estimate the price edge.",
  },
  {
    label: "Decision",
    title: "Preparing the recommendation",
    copy: "Pulling the comparison into a cleaner side by side decision view.",
  },
  {
    label: "Decision guide",
    title: "Preparing the final explanation",
    copy: "Turning the result into a fuller buying guide with the trade-offs and checks that matter most.",
  },
];

const elements = {
  form: document.getElementById("compare-form"),
  modePill: document.getElementById("mode-pill"),
  urlA: document.getElementById("url-a"),
  urlB: document.getElementById("url-b"),
  compareButton: document.getElementById("compare-button"),
  swapButton: document.getElementById("swap-button"),
  demoButton: document.getElementById("demo-button"),
  message: document.getElementById("form-message"),
  statusPill: document.getElementById("status-pill"),
  waitingPanel: document.getElementById("waiting-panel"),
  waitingTitle: document.getElementById("waiting-title"),
  waitingCopy: document.getElementById("waiting-copy"),
  waitingSteps: document.getElementById("waiting-steps"),
  resultsShell: document.getElementById("results-shell"),
  resultsCaption: document.getElementById("results-caption"),
  verdictCard: document.getElementById("verdict-card"),
  llmCard: document.getElementById("llm-card"),
  listingGrid: document.getElementById("listing-grid"),
};

let isLoading = false;
let compareReady = true;
let waitingTimerId = null;
let waitingStageIndex = 0;

initialize();

async function initialize() {
  elements.urlA.value = DEFAULT_URLS[0];
  elements.urlB.value = DEFAULT_URLS[1];

  elements.form.addEventListener("submit", handleSubmit);
  elements.swapButton.addEventListener("click", handleSwap);
  elements.demoButton.addEventListener("click", handleDemoPair);

  await refreshHealth();
}

async function refreshHealth() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) {
      throw new Error(`Health request failed with status ${response.status}.`);
    }

    renderHealth(await response.json());
  } catch (error) {
    console.error(error);
    compareReady = false;
    elements.modePill.textContent = "Unavailable";
    setStatus("Offline", "error");
    setMessage("Comparison is temporarily unavailable. Please try again in a moment.");
    syncControls();
  }
}

function renderHealth(health) {
  compareReady = Boolean(health && health.ready);
  elements.modePill.textContent = compareReady ? "Live comparison" : "Unavailable";
  setStatus(compareReady ? "Ready" : "Offline", compareReady ? "success" : "error");

  if (!isLoading) {
    setMessage(compareReady ? buildReadyMessage(health) : "Comparison is temporarily unavailable. Please try again later.");
  }

  syncControls();
}

async function handleSubmit(event) {
  event.preventDefault();

  const urlA = elements.urlA.value.trim();
  const urlB = elements.urlB.value.trim();

  if (!urlA || !urlB) {
    setStatus("Missing links", "error");
    setMessage("Add two eBay listing URLs before comparing.");
    return;
  }

  if (!compareReady) {
    setStatus("Offline", "error");
    setMessage("Comparison is temporarily unavailable. Please try again later.");
    return;
  }

  isLoading = true;
  syncControls();
  setStatus("Comparing", "loading");
  setMessage("This can take a little while while both listings are checked live.");
  startWaitingExperience();

  try {
    const result = await requestComparison(urlA, urlB);
    stopWaitingExperience();
    renderComparison(result, { urlA, urlB });
    setStatus("Done", "success");
    setMessage("Comparison complete. Review the side by side result below.");
  } catch (error) {
    console.error(error);
    stopWaitingExperience();
    if (error.payload && error.payload.runtime && error.payload.runtime.ready === false) {
      compareReady = false;
      elements.modePill.textContent = "Unavailable";
    }
    renderFailure(error.payload);
    setStatus("Could not compare", "error");
    setMessage(resolveErrorMessage(error.payload));
  } finally {
    isLoading = false;
    syncControls();
  }
}

function handleSwap() {
  const originalA = elements.urlA.value;
  elements.urlA.value = elements.urlB.value;
  elements.urlB.value = originalA;
}

function handleDemoPair() {
  elements.urlA.value = DEFAULT_URLS[0];
  elements.urlB.value = DEFAULT_URLS[1];
  setStatus("Ready", "success");
  setMessage("Loaded a sample pair for a quick comparison.");
}

function syncControls() {
  const disabled = isLoading || !compareReady;
  elements.compareButton.disabled = disabled;
  elements.swapButton.disabled = isLoading;
  elements.demoButton.disabled = isLoading;
  elements.compareButton.textContent = isLoading ? "Comparing..." : "Compare listings";
}

async function requestComparison(urlA, urlB) {
  const response = await fetch("/api/compare", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      url_a: urlA,
      url_b: urlB,
    }),
  });

  const contentType = response.headers.get("Content-Type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : { error: { message: await response.text() } };

  if (!response.ok) {
    const error = new Error(resolveErrorMessage(payload));
    error.payload = payload;
    throw error;
  }

  return payload;
}

function startWaitingExperience() {
  waitingStageIndex = 0;
  elements.waitingPanel.hidden = false;
  updateWaitingStage(waitingStageIndex);
  renderLoadingVerdict(WAITING_STAGES[waitingStageIndex]);
  renderLoadingLlm();
  renderLoadingListings();
  elements.resultsShell.scrollIntoView({ behavior: "smooth", block: "start" });

  clearWaitingTimer();
  waitingTimerId = window.setInterval(() => {
    if (waitingStageIndex < WAITING_STAGES.length - 1) {
      waitingStageIndex += 1;
    }
    updateWaitingStage(waitingStageIndex);
    renderLoadingVerdict(WAITING_STAGES[waitingStageIndex]);
  }, 1700);
}

function stopWaitingExperience() {
  clearWaitingTimer();
  elements.waitingPanel.hidden = true;
}

function clearWaitingTimer() {
  if (waitingTimerId !== null) {
    window.clearInterval(waitingTimerId);
    waitingTimerId = null;
  }
}

function updateWaitingStage(stageIndex) {
  const stage = WAITING_STAGES[stageIndex] || WAITING_STAGES[0];
  elements.waitingTitle.textContent = stage.title;
  elements.waitingCopy.textContent = stage.copy;
  elements.waitingSteps.innerHTML = WAITING_STAGES.map((item, index) => {
    const state =
      index < stageIndex ? "is-complete" : index === stageIndex ? "is-current" : "";
    return `<div class="waiting-step ${state}">${escapeHtml(item.label)}</div>`;
  }).join("");
}

function renderLoadingVerdict(stage) {
  elements.resultsCaption.textContent =
    "Comparing both listings now. The board below will fill in as soon as the result is ready.";
  elements.verdictCard.innerHTML = `
    <div class="loading-card">
      <span class="summary-label">Recommendation</span>
      <h3 class="verdict-title">${escapeHtml(stage.title)}</h3>
      <p class="verdict-copy">${escapeHtml(stage.copy)}</p>
      <div class="summary-band">
        ${renderLoadingBlock()}
        ${renderLoadingBlock()}
        ${renderLoadingBlock()}
      </div>
      <div class="loading-copy">
        <div class="placeholder-line"></div>
        <div class="placeholder-line"></div>
        <div class="placeholder-line short"></div>
      </div>
    </div>
  `;
}

function renderLoadingLlm() {
  elements.llmCard.innerHTML = `
    <div class="loading-card">
      <div class="llm-header">
        <div>
          <span class="summary-label">Decision guide</span>
          <h3 class="llm-title">Building the detailed explanation</h3>
          <p class="llm-copy">
            Expanding the recommendation into a clearer read on the trade-offs, the safer choice, and what still deserves a manual check.
          </p>
        </div>
        <span class="story-pill">In progress</span>
      </div>
      <div class="llm-layout">
        <div class="llm-story-grid">
          <div class="llm-paragraph">
            <div class="llm-placeholder-lines">
              <div class="placeholder-line"></div>
              <div class="placeholder-line"></div>
              <div class="placeholder-line"></div>
              <div class="placeholder-line short"></div>
            </div>
          </div>
          <div class="llm-paragraph">
            <div class="llm-placeholder-lines">
              <div class="placeholder-line"></div>
              <div class="placeholder-line"></div>
              <div class="placeholder-line"></div>
              <div class="placeholder-line short"></div>
            </div>
          </div>
        </div>
        <aside class="llm-side-panel">
          <div class="llm-side-card">
            <span class="summary-label">Before you buy</span>
            <div class="llm-placeholder-lines">
              <div class="placeholder-line"></div>
              <div class="placeholder-line"></div>
              <div class="placeholder-line short"></div>
            </div>
          </div>
        </aside>
      </div>
    </div>
  `;
}

function renderLoadingListings() {
  elements.listingGrid.innerHTML = [
    renderLoadingListingCard("Listing A"),
    renderLoadingListingCard("Listing B"),
  ].join("");
}

function renderLoadingListingCard(label) {
  return `
    <article class="listing-card loading-card">
      <div class="listing-header">
        <div>
          <span class="listing-label">${escapeHtml(label)}</span>
          <h3>Building the comparison card</h3>
          <div class="loading-copy">
            <div class="placeholder-line"></div>
            <div class="placeholder-line short"></div>
          </div>
        </div>
        <div class="gauge">
          <div class="gauge-inner">
            <strong>...</strong>
            <span>loading</span>
          </div>
        </div>
      </div>
      <div class="chip-row">
        <span class="chip">Checking price</span>
        <span class="chip">Checking trust</span>
        <span class="chip">Checking market</span>
      </div>
      <div class="metric-grid">
        ${renderLoadingMetric()}
        ${renderLoadingMetric()}
        ${renderLoadingMetric()}
        ${renderLoadingMetric()}
      </div>
    </article>
  `;
}

function renderLoadingMetric() {
  return `
    <div class="metric-card">
      <div class="placeholder-line short"></div>
      <div class="placeholder-line" style="margin-top:0.7rem;"></div>
      <div class="placeholder-line short" style="margin-top:0.55rem;"></div>
    </div>
  `;
}

function renderLoadingBlock() {
  return `
    <div class="summary-block">
      <div class="placeholder-line short"></div>
      <div class="placeholder-line" style="margin-top:0.7rem;"></div>
    </div>
  `;
}

function renderComparison(result, submittedUrls) {
  const listingA = withFallbackUrl(result.listing_a, submittedUrls.urlA);
  const listingB = withFallbackUrl(result.listing_b, submittedUrls.urlB);
  const comparison = result.comparison || {};
  const delta = Math.abs(Number(comparison.good_value_probability_delta || 0));

  elements.resultsCaption.textContent = buildResultsCaption(listingA, listingB, comparison);
  elements.verdictCard.innerHTML = renderVerdictCard(listingA, listingB, comparison, delta);
  elements.llmCard.innerHTML = renderLlmCard(result.llm_explanation, comparison);
  elements.listingGrid.innerHTML = [
    renderListingCard(listingA, "Listing A", comparison.verdict === "better_A"),
    renderListingCard(listingB, "Listing B", comparison.verdict === "better_B"),
  ].join("");

  elements.resultsShell.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderFailure(payload) {
  const message = resolveErrorMessage(payload);
  elements.resultsCaption.textContent =
    "The comparison did not finish this time. Please review the message below and try again.";
  elements.verdictCard.innerHTML = `
    <span class="summary-label">Recommendation</span>
    <h3 class="verdict-title">We could not finish the comparison</h3>
    <p class="verdict-copy">${escapeHtml(message)}</p>
  `;
  elements.llmCard.innerHTML = renderLlmCard(null, {}, true);
  elements.listingGrid.innerHTML = [
    renderFailureListingCard("Listing A"),
    renderFailureListingCard("Listing B"),
  ].join("");
}

function renderFailureListingCard(label) {
  return `
    <article class="placeholder-card">
      <p class="placeholder-label">${escapeHtml(label)}</p>
      <h3>Comparison unavailable</h3>
      <p>Try again once the service is ready and both listing links are valid.</p>
    </article>
  `;
}

function renderVerdictCard(listingA, listingB, comparison, delta) {
  const verdict = describeVerdict(comparison.verdict, listingA, listingB);
  const reasons = Array.isArray(comparison.reasons) ? comparison.reasons : [];
  const winnerLabel = resolveWinnerLabel(comparison.verdict);
  const verdictTone = resolveVerdictTone(comparison.verdict);

  return `
    <div class="verdict-top">
      <div>
        <span class="summary-label">Final recommendation</span>
        <h3 class="verdict-title">${escapeHtml(verdict.title)}</h3>
        <p class="verdict-copy">${escapeHtml(verdict.copy)}</p>
      </div>
      <div class="verdict-mark" data-tone="${escapeAttribute(verdictTone)}">
        <span class="verdict-mark-label">Best read right now</span>
        <strong>${escapeHtml(winnerLabel)}</strong>
      </div>
    </div>
    <div class="summary-band">
      <div class="summary-block">
        <span class="summary-label">Recommended pick</span>
        <strong>${escapeHtml(winnerLabel)}</strong>
      </div>
      <div class="summary-block">
        <span class="summary-label">Head to head edge</span>
        <strong>${escapeHtml(resolveEdgeLabel(comparison.verdict, delta))}</strong>
      </div>
      <div class="summary-block">
        <span class="summary-label">Read on this result</span>
        <strong>${escapeHtml(resolveDecisionFeel(comparison.verdict, delta))}</strong>
      </div>
    </div>
    <div class="reason-stack">
      ${reasons.map((reason) => `<p class="reason-line">${escapeHtml(reason)}</p>`).join("")}
      <p class="reason-line">${escapeHtml(buildSecondaryInsight(listingA, listingB, comparison.verdict))}</p>
    </div>
  `;
}

function renderLlmCard(explanation, comparison = {}, showErrorState = false) {
  const paragraphs = Array.isArray(explanation && explanation.paragraphs)
    ? explanation.paragraphs.filter((paragraph) => String(paragraph || "").trim())
    : [];
  const watchouts = Array.isArray(explanation && explanation.watchouts)
    ? explanation.watchouts.filter((item) => String(item || "").trim())
    : [];
  const status = explanation && explanation.status ? explanation.status : "pending";
  const title =
    explanation && explanation.title
      ? explanation.title
      : showErrorState
        ? "The detailed explanation is unavailable for this run"
        : "The detailed buying guide will appear here";
  const lead =
    explanation && explanation.lead
      ? explanation.lead
      : showErrorState
        ? "Run a successful comparison first and this guide will fill in automatically."
        : "Once the comparison finishes, this section will expand the recommendation into a fuller buyer-facing guide.";
  const note = resolveExplanationNote(status, comparison, showErrorState);
  const storyLabel = resolveExplanationLabel(status, showErrorState);
  const watchoutTitle = comparison && comparison.verdict === "better_A"
    ? "What to double-check on Listing A"
    : comparison && comparison.verdict === "better_B"
      ? "What to double-check on Listing B"
      : "What to double-check";

  if (!paragraphs.length) {
    return `
      <div class="llm-header">
        <div>
          <span class="summary-label">Decision guide</span>
          <h3 class="llm-title">${escapeHtml(title)}</h3>
          <p class="llm-copy">${escapeHtml(lead)}</p>
        </div>
        <span class="story-pill">${escapeHtml(storyLabel)}</span>
      </div>
      <div class="llm-layout">
        <div class="llm-story-grid">
          <div class="llm-paragraph">
            <div class="llm-placeholder-lines">
              <div class="placeholder-line"></div>
              <div class="placeholder-line"></div>
              <div class="placeholder-line"></div>
              <div class="placeholder-line short"></div>
            </div>
          </div>
          <div class="llm-paragraph">
            <div class="llm-placeholder-lines">
              <div class="placeholder-line"></div>
              <div class="placeholder-line"></div>
              <div class="placeholder-line"></div>
              <div class="placeholder-line short"></div>
            </div>
          </div>
        </div>
        <aside class="llm-side-panel">
          <div class="llm-side-card">
            <span class="summary-label">What to expect</span>
            <p class="llm-note">${escapeHtml(lead)}</p>
          </div>
        </aside>
      </div>
    `;
  }

  return `
    <div class="llm-header">
      <div>
        <span class="summary-label">Decision guide</span>
        <h3 class="llm-title">${escapeHtml(title)}</h3>
        <p class="llm-copy">${escapeHtml(lead)}</p>
      </div>
      <span class="story-pill">${escapeHtml(storyLabel)}</span>
    </div>
    <div class="llm-layout">
      <div class="llm-story-grid">
        ${paragraphs.map((paragraph) => `<p class="llm-paragraph">${escapeHtml(paragraph)}</p>`).join("")}
      </div>
      <aside class="llm-side-panel">
        ${
          watchouts.length
            ? `
          <div class="llm-side-card llm-watchouts">
            <span class="summary-label">${escapeHtml(watchoutTitle)}</span>
            <ul class="llm-watchout-list">
              ${watchouts.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
            </ul>
          </div>
        `
            : ""
        }
        ${
          note
            ? `
          <div class="llm-side-card">
            <span class="summary-label">Keep in mind</span>
            <p class="llm-note">${escapeHtml(note)}</p>
          </div>
        `
            : ""
        }
      </aside>
    </div>
  `;
}

function renderListingCard(listing, label, isHighlighted) {
  const valueProbability = clampPercent(Number(listing.good_value_probability || 0) * 100);
  const trustProbability = clampPercent(Number(listing.trust_probability || 0) * 100);
  const ewomScore = clampPercent(Number(listing.ewom_score_0_to_100 || 0));
  const predictionTone = resolvePredictionTone(listing.prediction);
  const recommendationBadge = isHighlighted
    ? `<span class="listing-mark">Recommended</span>`
    : "";

  return `
    <article class="listing-card" data-animate data-highlight="${String(isHighlighted)}">
      <div class="listing-header">
        <div>
          <div class="listing-label-row">
            <span class="listing-label">${escapeHtml(label)}</span>
            ${recommendationBadge}
          </div>
          <h3>${escapeHtml(listing.title || label)}</h3>
          <a class="listing-link" href="${escapeAttribute(listing.source_url || "#")}" target="_blank" rel="noreferrer">${escapeHtml(
            shortenUrl(listing.source_url)
          )}</a>
        </div>
        <div class="gauge" style="--value:${valueProbability}; --tone:${resolveGaugeTone(predictionTone)};">
          <div class="gauge-inner">
            <strong>${escapeHtml(formatProbability(listing.good_value_probability || 0))}</strong>
            <span>value read</span>
          </div>
        </div>
      </div>

      <div class="chip-row">
        <span class="chip" data-tone="${escapeAttribute(predictionTone)}">${escapeHtml(formatPrediction(listing.prediction))}</span>
        <span class="chip" data-tone="${escapeAttribute(resolveRetrievalTone(listing.retrieval_status))}">${escapeHtml(formatRetrieval(listing.retrieval_status))}</span>
        <span class="chip">${escapeHtml(String(listing.retrieved_neighbor_count ?? 0))} market matches</span>
        <span class="chip">${escapeHtml(String(listing.seller_feedback_review_count ?? 0))} reviews analyzed</span>
      </div>

      <div class="metric-grid">
        ${renderMetricCard(
          "Item price",
          formatCurrency(listing.total_price, listing.total_price_currency),
          "Shipping is excluded from this score."
        )}
        ${renderMetricCard(
          "Market price",
          formatCurrency(listing.peer_price, listing.total_price_currency),
          "Estimated from similar listings."
        )}
        ${renderMetricCard(
          "Price edge",
          formatPriceGap(listing.price_gap_vs_peer),
          "Positive means the listing is priced below the market read."
        )}
        ${renderMetricCard(
          "Listing trust",
          formatProbability(listing.trust_probability || 0),
          "Higher means the listing copy reads as more trustworthy.",
          trustProbability
        )}
        ${renderMetricCard(
          "Review signal",
          formatScoreOutOf100(listing.ewom_score_0_to_100),
          "Higher means the review picture looks healthier.",
          ewomScore
        )}
        ${renderMetricCard(
          "Market source",
          formatPeerPriceSource(listing.peer_price_source),
          "Shows where the market benchmark came from."
        )}
      </div>

      <div class="listing-footer">
        <p class="footer-copy">${escapeHtml(buildListingFooter(listing))}</p>
        <span class="badge" data-tone="${escapeAttribute(predictionTone)}">${escapeHtml(
          formatPrediction(listing.prediction)
        )}</span>
      </div>
    </article>
  `;
}

function renderMetricCard(label, value, copy, meterPercent) {
  return `
    <div class="metric-card">
      <span class="metric-label">${escapeHtml(label)}</span>
      <div class="metric-value">${escapeHtml(value)}</div>
      <div class="metric-copy">${escapeHtml(copy)}</div>
      ${
        Number.isFinite(meterPercent)
          ? `<div class="signal-bar"><span style="width:${clampPercent(meterPercent)}%"></span></div>`
          : ""
      }
    </div>
  `;
}

function buildResultsCaption(listingA, listingB, comparison) {
  const verdict = describeVerdict(comparison.verdict, listingA, listingB);
  return `${verdict.title}. Compare the two cards below before deciding which listing to buy.`;
}

function buildSecondaryInsight(listingA, listingB, verdict) {
  if (verdict === "better_A") {
    return `${listingA.title} shows the stronger overall value read right now, especially on price position and supporting trust signals.`;
  }
  if (verdict === "better_B") {
    return `${listingB.title} edges ahead on the current value read and looks like the safer buy from the signals available.`;
  }
  if (verdict === "tie") {
    return "This result is close enough that the better final tie-breakers are condition, warranty, shipping speed, or any seller detail you personally care about.";
  }
  return "The result does not have enough support to confidently choose one listing over the other yet.";
}

function buildListingFooter(listing) {
  const peerStatus = formatRetrieval(listing.retrieval_status);
  const peers = String(listing.retrieved_neighbor_count ?? 0);
  const comments = String(listing.seller_feedback_review_count ?? 0);
  return `${peerStatus}. Based on ${peers} similar listings and ${comments} review texts.`;
}

function buildReadyMessage(health) {
  const listingTrust = health && health.listing_trust ? health.listing_trust : {};
  if (listingTrust.ready) {
    return "Paste two eBay links to compare price, listing trust, and review signals in one cleaner decision view.";
  }
  return "Comparison is unavailable right now. Please try again in a moment.";
}

function resolveExplanationLabel(status, showErrorState) {
  if (showErrorState) {
    return "Unavailable";
  }
  if (status === "fallback") {
    return "Signal-based read";
  }
  if (status === "generated") {
    return "Detailed guide";
  }
  return "Preparing";
}

function resolveExplanationNote(status, comparison, showErrorState) {
  if (showErrorState) {
    return "Run the comparison again once both listing links are valid.";
  }
  if (status === "fallback") {
    return "This guide is based on the comparison signals alone, so use it as direction rather than a final guarantee.";
  }
  if (comparison && comparison.verdict === "insufficient_evidence") {
    return "The result is still thin, so treat the recommendation as tentative and verify the listing details manually.";
  }
  return "";
}

function resolveErrorMessage(payload) {
  if (payload && payload.error && payload.error.message) {
    return "We hit a problem while comparing these listings. Please try again.";
  }
  return "We hit a problem while comparing these listings. Please try again.";
}

function withFallbackUrl(listing, fallbackUrl) {
  return {
    ...(listing || {}),
    source_url: listing && listing.source_url ? listing.source_url : fallbackUrl,
  };
}

function describeVerdict(verdict, listingA, listingB) {
  if (verdict === "better_A") {
    return {
      title: "Listing A comes out ahead",
      copy: `${listingA.title} currently looks like the stronger buy. It holds the better overall value read against the available market and trust signals.`,
    };
  }

  if (verdict === "better_B") {
    return {
      title: "Listing B comes out ahead",
      copy: `${listingB.title} currently looks like the stronger buy. Its overall mix of value and supporting signals reads a bit better right now.`,
    };
  }

  if (verdict === "tie") {
    return {
      title: "This one is too close to call",
      copy: "The two listings land close enough together that the safest takeaway is to treat them as a near tie rather than force a winner.",
    };
  }

  return {
    title: "This result needs more support",
    copy: "There is not enough reliable evidence yet to confidently pick one listing over the other.",
  };
}

function resolveWinnerLabel(verdict) {
  if (verdict === "better_A") {
    return "Listing A";
  }
  if (verdict === "better_B") {
    return "Listing B";
  }
  if (verdict === "tie") {
    return "Too close to call";
  }
  return "Not enough support";
}

function resolveEdgeLabel(verdict, delta) {
  if (verdict === "tie") {
    return "Very small";
  }
  if (verdict === "insufficient_evidence") {
    return "Not reliable yet";
  }
  return formatProbability(delta);
}

function resolveDecisionFeel(verdict, delta) {
  if (verdict === "tie") {
    return "Close call";
  }
  if (verdict === "insufficient_evidence") {
    return "Needs caution";
  }
  return delta >= 0.12 ? "Clear edge" : "Lean, not landslide";
}

function resolveVerdictTone(verdict) {
  if (verdict === "better_A" || verdict === "better_B") {
    return "good";
  }
  if (verdict === "tie") {
    return "warn";
  }
  return "danger";
}

function formatPrediction(prediction) {
  if (prediction === "good_value") {
    return "Looks strong";
  }
  if (prediction === "not_good_value") {
    return "Watch pricing";
  }
  if (prediction === "insufficient_evidence") {
    return "Not enough evidence";
  }
  return "Unknown";
}

function formatRetrieval(status) {
  if (status === "usable") {
    return "Market match found";
  }
  if (!status) {
    return "Market read pending";
  }
  return "Market read limited";
}

function formatPeerPriceSource(source) {
  if (source === "retrieval") {
    return "Similar listings";
  }
  if (!source || source === "none") {
    return "No market match";
  }
  return source.replaceAll("_", " ");
}

function formatPriceGap(gap) {
  if (gap === null || gap === undefined || Number.isNaN(Number(gap))) {
    return "No market gap";
  }

  const value = Number(gap);
  if (Math.abs(value) < 0.001) {
    return "At market price";
  }
  if (value > 0) {
    return `${formatProbability(value)} below`;
  }
  return `${formatProbability(Math.abs(value))} above`;
}

function formatCurrency(value, currency) {
  const amount = Number(value);
  if (!Number.isFinite(amount)) {
    return "-";
  }

  if (currency) {
    try {
      return new Intl.NumberFormat("en-SG", {
        style: "currency",
        currency,
        maximumFractionDigits: 2,
      }).format(amount);
    } catch (error) {
      return `${currency} ${amount.toFixed(2)}`;
    }
  }

  return amount.toFixed(2);
}

function formatProbability(value) {
  const probability = Number(value);
  if (!Number.isFinite(probability)) {
    return "-";
  }
  return `${(probability * 100).toFixed(0)}%`;
}

function formatScoreOutOf100(value) {
  const score = Number(value);
  if (!Number.isFinite(score)) {
    return "-";
  }
  return `${score.toFixed(1)} / 100`;
}

function resolvePredictionTone(prediction) {
  if (prediction === "good_value") {
    return "good";
  }
  if (prediction === "not_good_value") {
    return "warn";
  }
  return "danger";
}

function resolveRetrievalTone(status) {
  if (status === "usable") {
    return "good";
  }
  if (!status) {
    return "warn";
  }
  return "danger";
}

function resolveGaugeTone(tone) {
  if (tone === "warn") {
    return "var(--sun)";
  }
  if (tone === "danger") {
    return "var(--coral)";
  }
  return "var(--accent)";
}

function shortenUrl(url) {
  if (!url) {
    return "No URL provided";
  }

  try {
    const parsed = new URL(url);
    return `${parsed.host}${parsed.pathname}`;
  } catch (error) {
    return url;
  }
}

function setMessage(message) {
  elements.message.textContent = message;
}

function setStatus(label, state) {
  elements.statusPill.textContent = label;
  elements.statusPill.dataset.state = state;
}

function clampPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return 0;
  }
  return Math.max(0, Math.min(number, 100));
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function escapeAttribute(value) {
  return escapeHtml(value);
}
