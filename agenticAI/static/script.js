let currentThreadId = localStorage.getItem("travel_thread_id") || null;
let latestAnswerMarkdown = "";
let currentTripData = null;
let waitingForApproval = false;
let isPaperMode = false;

const AGENT_LABELS = {
  flight_agent: "✈️ Flight Agent",
  hotel_agent: "🏨 Hotel Agent",
  weather_agent: "🌦️ Weather Agent",
  budget_agent: "💰 Budget Agent",
  itinerary_agent: "🗓️ Itinerary Agent"
};

function setPrompt(text) {
  const input = document.getElementById("userInput");
  input.value = text;
  input.focus();
  input.scrollIntoView({ behavior: "smooth", block: "center" });
}

function setLoading(isLoading, mode = "draft") {
  const sendBtn = document.getElementById("sendBtn");
  const btnText = document.getElementById("btnText");
  const btnLoader = document.getElementById("btnLoader");
  const approveBtn = document.getElementById("approveBtn");
  const reviseBtn = document.getElementById("reviseBtn");

  if (sendBtn) sendBtn.disabled = isLoading;
  if (approveBtn) approveBtn.disabled = isLoading;
  if (reviseBtn) reviseBtn.disabled = isLoading;

  if (isLoading && mode === "draft") {
    btnText.classList.add("hidden");
    btnLoader.classList.remove("hidden");
  } else {
    btnText.classList.remove("hidden");
    btnLoader.classList.add("hidden");
  }
}

function showError(message) {
  const errorBox = document.getElementById("errorBox");
  if (!errorBox) return;
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
  errorBox.scrollIntoView({ behavior: "smooth", block: "center" });
}

function hideError() {
  const errorBox = document.getElementById("errorBox");
  if (!errorBox) return;
  errorBox.classList.add("hidden");
  errorBox.textContent = "";
}

function renderMarkdown(element, markdown) {
  if (typeof marked !== "undefined") {
    element.innerHTML = marked.parse(markdown || "");
  } else {
    element.innerText = markdown || "";
  }

  // Wrap tables and align columns for a premium dashboard look
  const tables = element.querySelectorAll("table");
  tables.forEach((table) => {
    if (!table.parentElement.classList.contains("table-wrapper")) {
      const wrapper = document.createElement("div");
      wrapper.className = "table-wrapper";
      table.parentNode.insertBefore(wrapper, table);
      wrapper.appendChild(table);
    }

    const headers = table.querySelectorAll("th");
    headers.forEach((th, colIdx) => {
      const text = th.textContent.trim().toLowerCase();
      let alignmentClass = "";
      if (/cost|price|budget|fare|rate|amount|\$|€|£|₹|tk|lakh/i.test(text)) {
        alignmentClass = "col-num";
      } else if (/^(day|step|no\.|#|status|tier)$/i.test(text)) {
        alignmentClass = "col-center";
      }

      if (alignmentClass) {
        th.classList.add(alignmentClass);
        const rows = table.querySelectorAll("tbody tr");
        rows.forEach((tr) => {
          const cell = tr.children[colIdx];
          if (cell) {
            cell.classList.add(alignmentClass);
          }
        });
      }
    });
  });
}

function showWorkflow(data) {
  const section = document.getElementById("workflowSection");
  const reasoning = document.getElementById("supervisorReasoning");
  const chips = document.getElementById("agentChips");
  const guardrailBadge = document.getElementById("guardrailBadge");

  reasoning.textContent = data.supervisor_reasoning || "Supervisor routing completed.";
  chips.innerHTML = "";

  (data.selected_agents || []).forEach((agent) => {
    const chip = document.createElement("span");
    chip.className = "agent-chip";
    chip.textContent = AGENT_LABELS[agent] || agent;
    chips.appendChild(chip);
  });

  if (data.guardrail_allowed === false) {
    guardrailBadge.textContent = "Guardrail blocked";
    guardrailBadge.classList.add("blocked");
  } else {
    guardrailBadge.textContent = "Guardrail passed";
    guardrailBadge.classList.remove("blocked");
  }

  section.classList.remove("hidden");
}

function updateTripSnapshot(data) {
  const snapshotContainer = document.getElementById("tripSnapshot");
  if (!snapshotContainer) return;

  const constraints = (data && data.trip_constraints) || {};
  const destination = constraints.destination || "Curated Travel Destination";
  const duration = constraints.duration || "Multi-Day Tour";
  const budget = constraints.budget || "Target Budget";
  const origin = constraints.origin || "International Departure";

  document.getElementById("snapDestination").textContent = destination;
  document.getElementById("snapDuration").textContent = duration;
  document.getElementById("snapBudget").textContent = budget;
  document.getElementById("snapOrigin").textContent = origin;

  const agentNames = ((data && data.selected_agents) || []).map(a => AGENT_LABELS[a] || a);
  document.getElementById("snapAgents").textContent = agentNames.length > 0 
    ? agentNames.join("  •  ") 
    : "✈️ Multi-Agent Travel Planner";

  snapshotContainer.classList.remove("hidden");
}

function showResult(answer, threadId, isDraft = false, data = null) {
  latestAnswerMarkdown = answer || "";

  const resultSection = document.getElementById("resultSection");
  const resultBox = document.getElementById("resultBox");
  const threadInfo = document.getElementById("threadInfo");
  const resultTitle = document.getElementById("resultTitle");
  const planStatusBadge = document.getElementById("planStatusBadge");

  renderMarkdown(resultBox, latestAnswerMarkdown);
  threadInfo.textContent = `Thread ID: ${threadId || "-"}`;

  if (planStatusBadge) {
    if (isDraft) {
      planStatusBadge.textContent = "Draft Itinerary (Review Needed)";
      planStatusBadge.className = "status-badge draft";
    } else {
      planStatusBadge.textContent = "Final Curated Plan";
      planStatusBadge.className = "status-badge final";
    }
  }

  resultTitle.textContent = isDraft ? "Review Draft Travel Plan" : "Your Final AI Travel Plan";

  if (data) {
    updateTripSnapshot(data);
  }

  resultSection.classList.remove("hidden");
  resultSection.scrollIntoView({
    behavior: "smooth",
    block: "start"
  });
}

function showApproval(data) {
  waitingForApproval = true;
  const section = document.getElementById("approvalSection");
  const approvalRequest = document.getElementById("approvalRequest");
  approvalRequest.textContent = data.approval_request ||
    "Approve the draft or provide feedback before the final plan is generated.";
  section.classList.remove("hidden");
}

function hideApproval() {
  waitingForApproval = false;
  document.getElementById("approvalSection").classList.add("hidden");
  document.getElementById("approvalFeedback").value = "";
}

let isDarkView = false;

function toggleResultTheme() {
  const pdfContent = document.getElementById("pdfContent");
  const toggleBtnText = document.getElementById("toggleBtnText");
  const toggleBtn = document.getElementById("viewToggleBtn");

  isDarkView = !isDarkView;
  if (isDarkView) {
    pdfContent.classList.add("dark-view");
    toggleBtnText.textContent = "Document View";
    toggleBtn.querySelector(".btn-icon").textContent = "📄";
  } else {
    pdfContent.classList.remove("dark-view");
    toggleBtnText.textContent = "Dark View";
    toggleBtn.querySelector(".btn-icon").textContent = "🌙";
  }
}

async function sendMessage() {
  hideError();
  hideApproval();

  const input = document.getElementById("userInput");
  const message = input.value.trim();

  if (!message) {
    showError("Please enter your travel request first.");
    return;
  }

  setLoading(true, "draft");

  try {
    const response = await fetch("/api/travel", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        message: message,
        thread_id: null
      })
    });

    const data = await response.json();

    if (!response.ok || !data.success) {
      throw new Error(data.error || "Something went wrong.");
    }

    currentThreadId = data.thread_id;
    currentTripData = data;
    localStorage.setItem("travel_thread_id", currentThreadId);

    showWorkflow(data);

    if (data.requires_approval) {
      showResult(data.itinerary || data.answer, data.thread_id, true, data);
      showApproval(data);
    } else {
      hideApproval();
      showResult(data.answer, data.thread_id, false, data);
    }
  } catch (error) {
    showError(error.message);
  } finally {
    setLoading(false, "draft");
  }
}

async function submitApproval(approved) {
  hideError();

  if (!currentThreadId || !waitingForApproval) {
    showError("There is no draft waiting for approval.");
    return;
  }

  const feedbackInput = document.getElementById("approvalFeedback");
  const feedback = feedbackInput.value.trim();

  if (!approved && !feedback) {
    showError("Please enter revision feedback before requesting changes.");
    feedbackInput.focus();
    return;
  }

  setLoading(true, "approval");

  try {
    const response = await fetch("/api/travel/approve", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        thread_id: currentThreadId,
        approved: approved,
        feedback: feedback
      })
    });

    const data = await response.json();

    if (!response.ok || !data.success) {
      throw new Error(data.error || "Could not resume the travel workflow.");
    }

    currentTripData = data;
    showWorkflow(data);
    hideApproval();
    showResult(data.answer, data.thread_id, false, data);
  } catch (error) {
    showError(error.message);
  } finally {
    setLoading(false, "approval");
  }
}

function copyResult() {
  const resultBox = document.getElementById("resultBox");
  const text = latestAnswerMarkdown || (resultBox ? resultBox.innerText : "");

  if (!text) {
    showError("No content available to copy.");
    return;
  }

  navigator.clipboard.writeText(text)
    .then(() => {
      const copyBtn = document.querySelector(".copy-btn");
      const label = copyBtn ? copyBtn.querySelector("span:not(.btn-icon)") : null;
      const oldText = label ? label.textContent : "Copy";
      
      if (label) {
        label.textContent = "Copied!";
        setTimeout(() => {
          label.textContent = oldText;
        }, 1500);
      }
    })
    .catch(() => {
      showError("Could not copy result.");
    });
}

function downloadPDF() {
  const pdfContent = document.getElementById("pdfContent");

  if (!latestAnswerMarkdown || !pdfContent) {
    showError("No travel plan available to download.");
    return;
  }

  const downloadBtn = document.querySelector(".download-btn");
  const oldHtml = downloadBtn.innerHTML;
  downloadBtn.innerHTML = '<span class="loader"></span> <span>Preparing PDF...</span>';
  downloadBtn.disabled = true;

  // Revert dark view temporarily to export clean white document
  const hadDarkView = pdfContent.classList.contains("dark-view");
  if (hadDarkView) {
    pdfContent.classList.remove("dark-view");
  }

  const options = {
    margin: [0.35, 0.35, 0.35, 0.35],
    filename: "ai-travel-plan.pdf",
    image: {
      type: "jpeg",
      quality: 0.98
    },
    html2canvas: {
      scale: 2,
      useCORS: true,
      backgroundColor: "#ffffff",
      logging: false
    },
    jsPDF: {
      unit: "in",
      format: "a4",
      orientation: "portrait"
    },
    pagebreak: {
      mode: ["avoid-all", "css", "legacy"]
    }
  };

  html2pdf()
    .set(options)
    .from(pdfContent)
    .save()
    .then(() => {
      downloadBtn.innerHTML = oldHtml;
      downloadBtn.disabled = false;
      if (hadDarkView) {
        pdfContent.classList.add("dark-view");
      }
    })
    .catch((err) => {
      downloadBtn.innerHTML = oldHtml;
      downloadBtn.disabled = false;
      if (hadDarkView) {
        pdfContent.classList.add("dark-view");
      }
      showError("Could not download PDF: " + (err.message || err));
    });
}

document.addEventListener("keydown", function(event) {
  if (event.ctrlKey && event.key === "Enter") {
    sendMessage();
  }
});
