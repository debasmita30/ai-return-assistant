import { useState, useEffect, useRef } from "react";

const PALETTE = {
  bg: "#0A0C10",
  surface: "#10141C",
  surfaceHover: "#161B26",
  border: "#1E2535",
  borderBright: "#2A3448",
  accent: "#4F8EF7",
  accentGlow: "#3B72D4",
  accentDim: "#1C3B7A",
  success: "#2DD98F",
  successDim: "#0D3D27",
  warning: "#F5A623",
  warningDim: "#3A2600",
  danger: "#EF4444",
  dangerDim: "#3A0F0F",
  textPrimary: "#EEF0F6",
  textSecondary: "#7A8BA6",
  textMuted: "#3F4F6A",
  purple: "#A855F7",
  purpleDim: "#2D1A45",
  teal: "#14B8A6",
  tealDim: "#0D2E2A",
};

const COMPLAINTS = ["Wrong Colour", "Size Issue", "Defective", "Not as Described", "Other"];

const MOCK_PRODUCTS = {
  "1078": { name: "Floral Wrap Midi Dress", type: "Dresses", rating: 4.2, reviews: 342 },
  "2034": { name: "Cotton Ribbed Turtleneck", type: "Tops", rating: 3.8, reviews: 218 },
  "3091": { name: "High-Rise Skinny Jeans", type: "Jeans", rating: 4.5, reviews: 501 },
  "4205": { name: "Linen Blazer Set", type: "Blazers", rating: 4.1, reviews: 127 },
  "5512": { name: "Strappy Heeled Sandals", type: "Shoes", rating: 3.6, reviews: 88 },
};

const MOCK_SENTIMENT = { Positive: 1847, Neutral: 623, Negative: 412 };

const MOCK_NEG_REVIEWS = [
  "The fabric quality is nowhere near what was shown in the photos. Very disappointing.",
  "Sizing is way off — ordered my usual size but it was two sizes too small.",
  "Received a completely different color than what was pictured online.",
  "The stitching started unraveling after just one wash. Poor craftsmanship.",
  "Waited 3 weeks and the item arrived with a broken zipper.",
];

function useCountUp(target, duration = 1200, active = false) {
  const [val, setVal] = useState(0);
  useEffect(() => {
    if (!active) return;
    let start = null;
    const step = (ts) => {
      if (!start) start = ts;
      const p = Math.min((ts - start) / duration, 1);
      setVal(Math.floor(p * target));
      if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }, [target, active]);
  return val;
}

function RiskArc({ score }) {
  const r = 58;
  const cx = 70;
  const cy = 70;
  const circumference = Math.PI * r;
  const dash = (score / 100) * circumference;
  const gap = circumference - dash;
  const color = score >= 70 ? PALETTE.danger : score >= 40 ? PALETTE.warning : PALETTE.success;
  const glowColor = score >= 70 ? "#EF444440" : score >= 40 ? "#F5A62340" : "#2DD98F40";

  return (
    <svg width="140" height="80" viewBox="0 0 140 90" style={{ overflow: "visible" }}>
      <defs>
        <filter id="arcglow">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
      </defs>
      <path
        d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
        fill="none"
        stroke={PALETTE.border}
        strokeWidth="10"
        strokeLinecap="round"
      />
      <path
        d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
        fill="none"
        stroke={color}
        strokeWidth="10"
        strokeLinecap="round"
        strokeDasharray={`${dash} ${gap + 1}`}
        style={{ filter: `drop-shadow(0 0 6px ${glowColor})`, transition: "all 0.8s cubic-bezier(0.34,1.56,0.64,1)" }}
      />
      <text x={cx} y={cy - 10} textAnchor="middle" fill={color} fontSize="22" fontWeight="700" fontFamily="monospace">
        {score}
      </text>
      <text x={cx} y={cy + 6} textAnchor="middle" fill={PALETTE.textSecondary} fontSize="10">
        RISK SCORE
      </text>
    </svg>
  );
}

function SentimentDonut({ data, active }) {
  const total = data.Positive + data.Neutral + data.Negative;
  const slices = [
    { label: "Positive", val: data.Positive, color: PALETTE.success },
    { label: "Neutral", val: data.Neutral, color: PALETTE.warning },
    { label: "Negative", val: data.Negative, color: PALETTE.danger },
  ];
  let cumulAngle = -Math.PI / 2;
  const cx = 60, cy = 60, r = 44, inner = 28;

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
      <svg width="120" height="120" viewBox="0 0 120 120">
        {slices.map((s) => {
          const angle = (s.val / total) * 2 * Math.PI;
          const startAngle = cumulAngle;
          cumulAngle += angle;
          const endAngle = cumulAngle;
          const x1 = cx + r * Math.cos(startAngle);
          const y1 = cy + r * Math.sin(startAngle);
          const x2 = cx + r * Math.cos(endAngle);
          const y2 = cy + r * Math.sin(endAngle);
          const xi1 = cx + inner * Math.cos(startAngle);
          const yi1 = cy + inner * Math.sin(startAngle);
          const xi2 = cx + inner * Math.cos(endAngle);
          const yi2 = cy + inner * Math.sin(endAngle);
          const lg = angle > Math.PI ? 1 : 0;
          const d = `M ${xi1} ${yi1} L ${x1} ${y1} A ${r} ${r} 0 ${lg} 1 ${x2} ${y2} L ${xi2} ${yi2} A ${inner} ${inner} 0 ${lg} 0 ${xi1} ${yi1}`;
          return (
            <path
              key={s.label}
              d={d}
              fill={s.color}
              opacity={active ? 0.9 : 0.3}
              style={{ transition: "opacity 0.5s" }}
            />
          );
        })}
        <circle cx={cx} cy={cy} r={inner - 2} fill={PALETTE.surface} />
        <text x={cx} y={cy - 4} textAnchor="middle" fill={PALETTE.textPrimary} fontSize="12" fontWeight="600">
          {total.toLocaleString()}
        </text>
        <text x={cx} y={cy + 10} textAnchor="middle" fill={PALETTE.textSecondary} fontSize="8">
          REVIEWS
        </text>
      </svg>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {slices.map((s) => (
          <div key={s.label} style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{ width: 8, height: 8, borderRadius: "50%", background: s.color, flexShrink: 0 }} />
            <span style={{ fontSize: 11, color: PALETTE.textSecondary }}>{s.label}</span>
            <span style={{ fontSize: 11, color: PALETTE.textPrimary, fontWeight: 600, marginLeft: "auto", minWidth: 36, textAlign: "right" }}>
              {active ? s.val.toLocaleString() : "—"}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function Scanline({ active }) {
  const [pos, setPos] = useState(0);
  useEffect(() => {
    if (!active) return;
    let frame;
    let start = null;
    const animate = (ts) => {
      if (!start) start = ts;
      const elapsed = (ts - start) % 1800;
      setPos(elapsed / 1800);
      frame = requestAnimationFrame(animate);
    };
    frame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frame);
  }, [active]);
  if (!active) return null;
  return (
    <div style={{
      position: "absolute", top: 0, left: 0, right: 0, bottom: 0,
      pointerEvents: "none", overflow: "hidden", borderRadius: 8,
    }}>
      <div style={{
        position: "absolute", left: 0, right: 0, height: 2,
        background: `linear-gradient(to right, transparent, ${PALETTE.accent}80, transparent)`,
        top: `${pos * 100}%`,
        transition: "none",
        boxShadow: `0 0 12px ${PALETTE.accent}`,
      }} />
    </div>
  );
}

function Tag({ children, color = PALETTE.accent, bg }) {
  return (
    <span style={{
      display: "inline-flex", alignItems: "center",
      padding: "2px 8px", borderRadius: 4,
      fontSize: 10, fontWeight: 700, letterSpacing: "0.06em",
      color: color,
      background: bg || `${color}18`,
      border: `1px solid ${color}30`,
    }}>
      {children}
    </span>
  );
}

function MetricCard({ label, value, unit, color, icon, active, delta }) {
  return (
    <div style={{
      background: PALETTE.surface,
      border: `1px solid ${PALETTE.border}`,
      borderRadius: 8, padding: "14px 16px",
      display: "flex", flexDirection: "column", gap: 4,
      transition: "border-color 0.3s",
    }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <span style={{ fontSize: 11, color: PALETTE.textSecondary, letterSpacing: "0.04em" }}>{label}</span>
        <span style={{ fontSize: 16, color: color || PALETTE.accent }}>{icon}</span>
      </div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 4 }}>
        <span style={{ fontSize: 22, fontWeight: 700, color: PALETTE.textPrimary, fontFamily: "monospace" }}>
          {active ? value : "—"}
        </span>
        {unit && <span style={{ fontSize: 11, color: PALETTE.textSecondary }}>{unit}</span>}
      </div>
      {delta && active && (
        <span style={{ fontSize: 10, color: delta > 0 ? PALETTE.danger : PALETTE.success }}>
          {delta > 0 ? "▲" : "▼"} {Math.abs(delta)}% vs avg
        </span>
      )}
    </div>
  );
}

function StepIndicator({ step, total }) {
  return (
    <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
      {Array.from({ length: total }).map((_, i) => (
        <div key={i} style={{
          height: 3, borderRadius: 2,
          background: i <= step ? PALETTE.accent : PALETTE.border,
          width: i === step ? 20 : 10,
          transition: "all 0.3s",
        }} />
      ))}
    </div>
  );
}

export default function ReturnAssistant() {
  const [productId, setProductId] = useState("1078");
  const [complaint, setComplaint] = useState("Defective");
  const [customComplaint, setCustomComplaint] = useState("");
  const [severity, setSeverity] = useState(5);
  const [manualApprove, setManualApprove] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzed, setAnalyzed] = useState(false);
  const [showNegReviews, setShowNegReviews] = useState(false);
  const [step, setStep] = useState(0);
  const [animStep, setAnimStep] = useState(0);
  const inputRef = useRef();

  const product = MOCK_PRODUCTS[productId.trim()] || null;
  const actualComplaint = complaint === "Other" ? customComplaint : complaint;

  const imagePrediction = severity > 6 || complaint === "Defective" ? "Defective" : "Normal";
  const predictedClass = product?.type || "Tops";
  const expectedClass = product?.type || "N/A";
  const complaintMismatch = false;

  let riskScore = severity * 10;
  if (imagePrediction === "Defective") riskScore += 40;
  if (complaintMismatch) riskScore += 20;
  riskScore = Math.min(riskScore, 100);

  const riskLevel = riskScore >= 70 ? "high" : riskScore >= 40 ? "moderate" : "low";
  const riskColor = riskScore >= 70 ? PALETTE.danger : riskScore >= 40 ? PALETTE.warning : PALETTE.success;

  const handleAnalyze = () => {
    if (!productId || !actualComplaint) return;
    setAnalyzing(true);
    setAnalyzed(false);
    setStep(0);
    setAnimStep(0);
    let s = 0;
    const iv = setInterval(() => {
      s++;
      setStep(s);
      if (s >= 4) {
        clearInterval(iv);
        setAnalyzing(false);
        setAnalyzed(true);
        setAnimStep(1);
      }
    }, 480);
  };

  const stepLabels = ["Parsing request", "Text classification", "Image analysis", "Risk computation", "Complete"];

  return (
    <div style={{
      background: PALETTE.bg,
      minHeight: "100vh",
      fontFamily: "'IBM Plex Mono', 'Courier New', monospace",
      color: PALETTE.textPrimary,
      padding: "0 0 48px",
    }}>
      {/* Header */}
      <div style={{
        borderBottom: `1px solid ${PALETTE.border}`,
        padding: "14px 28px",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        background: PALETTE.surface,
        position: "sticky", top: 0, zIndex: 10,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{
            width: 28, height: 28, borderRadius: 6,
            background: `linear-gradient(135deg, ${PALETTE.accent}, ${PALETTE.purple})`,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 14,
          }}>⬡</div>
          <div>
            <div style={{ fontSize: 13, fontWeight: 700, letterSpacing: "0.06em" }}>RETURN.AI</div>
            <div style={{ fontSize: 9, color: PALETTE.textMuted, letterSpacing: "0.1em" }}>INTELLIGENT RETURNS PLATFORM</div>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ width: 6, height: 6, borderRadius: "50%", background: PALETTE.success, boxShadow: `0 0 8px ${PALETTE.success}` }} />
          <span style={{ fontSize: 10, color: PALETTE.textSecondary }}>SYSTEM ONLINE</span>
        </div>
      </div>

      <div style={{ padding: "24px 28px", maxWidth: 1160, margin: "0 auto" }}>
        {/* Top: 2-col layout */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginBottom: 20 }}>
          {/* Left Panel — Input */}
          <div style={{
            background: PALETTE.surface,
            border: `1px solid ${PALETTE.border}`,
            borderRadius: 10, overflow: "hidden",
          }}>
            <div style={{
              padding: "12px 18px",
              borderBottom: `1px solid ${PALETTE.border}`,
              display: "flex", alignItems: "center", gap: 8,
            }}>
              <span style={{ fontSize: 10, color: PALETTE.textSecondary, letterSpacing: "0.1em" }}>INPUT TERMINAL</span>
              <StepIndicator step={step} total={5} />
            </div>

            <div style={{ padding: 18, display: "flex", flexDirection: "column", gap: 16 }}>
              {/* Product ID */}
              <div>
                <label style={{ fontSize: 10, color: PALETTE.textSecondary, letterSpacing: "0.08em", display: "block", marginBottom: 6 }}>
                  PRODUCT ID
                </label>
                <div style={{ position: "relative" }}>
                  <input
                    ref={inputRef}
                    value={productId}
                    onChange={e => setProductId(e.target.value)}
                    placeholder="e.g. 1078"
                    style={{
                      width: "100%", padding: "9px 12px",
                      background: PALETTE.bg,
                      border: `1px solid ${product ? PALETTE.accent + "60" : PALETTE.border}`,
                      borderRadius: 6, color: PALETTE.textPrimary,
                      fontSize: 13, fontFamily: "inherit",
                      outline: "none", boxSizing: "border-box",
                      transition: "border-color 0.3s",
                    }}
                  />
                  {product && (
                    <div style={{
                      marginTop: 6, padding: "6px 10px",
                      background: `${PALETTE.accent}12`,
                      border: `1px solid ${PALETTE.accent}30`,
                      borderRadius: 5,
                      display: "flex", alignItems: "center", justifyContent: "space-between",
                    }}>
                      <span style={{ fontSize: 11, color: PALETTE.textPrimary }}>{product.name}</span>
                      <Tag color={PALETTE.teal}>{product.type}</Tag>
                    </div>
                  )}
                  {!product && productId && (
                    <div style={{ marginTop: 4, fontSize: 10, color: PALETTE.danger }}>
                      ✗ Product not found. Try: 1078, 2034, 3091, 4205, 5512
                    </div>
                  )}
                </div>
              </div>

              {/* Complaint */}
              <div>
                <label style={{ fontSize: 10, color: PALETTE.textSecondary, letterSpacing: "0.08em", display: "block", marginBottom: 6 }}>
                  COMPLAINT CATEGORY
                </label>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
                  {COMPLAINTS.map(c => (
                    <button
                      key={c}
                      onClick={() => setComplaint(c)}
                      style={{
                        padding: "7px 10px", borderRadius: 5,
                        border: `1px solid ${complaint === c ? PALETTE.accent : PALETTE.border}`,
                        background: complaint === c ? `${PALETTE.accent}15` : "transparent",
                        color: complaint === c ? PALETTE.accent : PALETTE.textSecondary,
                        fontSize: 10, cursor: "pointer",
                        fontFamily: "inherit", letterSpacing: "0.04em",
                        transition: "all 0.2s",
                        textAlign: "left",
                      }}
                    >
                      {complaint === c ? "◈ " : "◇ "}{c}
                    </button>
                  ))}
                </div>
                {complaint === "Other" && (
                  <input
                    value={customComplaint}
                    onChange={e => setCustomComplaint(e.target.value)}
                    placeholder="Describe the issue…"
                    style={{
                      marginTop: 8, width: "100%", padding: "8px 12px",
                      background: PALETTE.bg, border: `1px solid ${PALETTE.border}`,
                      borderRadius: 6, color: PALETTE.textPrimary,
                      fontSize: 12, fontFamily: "inherit", outline: "none", boxSizing: "border-box",
                    }}
                  />
                )}
              </div>

              {/* Severity */}
              <div>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                  <label style={{ fontSize: 10, color: PALETTE.textSecondary, letterSpacing: "0.08em" }}>
                    SEVERITY LEVEL
                  </label>
                  <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <span style={{
                      fontSize: 16, fontWeight: 700, fontFamily: "monospace",
                      color: severity >= 7 ? PALETTE.danger : severity >= 4 ? PALETTE.warning : PALETTE.success,
                    }}>{severity}</span>
                    <span style={{ fontSize: 9, color: PALETTE.textMuted }}>/10</span>
                  </div>
                </div>
                <div style={{ position: "relative" }}>
                  <div style={{
                    height: 4, borderRadius: 2, background: PALETTE.border,
                    marginBottom: 4, position: "relative", overflow: "hidden",
                  }}>
                    <div style={{
                      height: "100%", borderRadius: 2,
                      width: `${severity * 10}%`,
                      background: severity >= 7 ? PALETTE.danger : severity >= 4 ? PALETTE.warning : PALETTE.success,
                      transition: "all 0.3s",
                      boxShadow: `0 0 8px ${severity >= 7 ? PALETTE.danger : severity >= 4 ? PALETTE.warning : PALETTE.success}80`,
                    }} />
                  </div>
                  <input
                    type="range" min={1} max={10} value={severity}
                    onChange={e => setSeverity(Number(e.target.value))}
                    style={{
                      width: "100%", height: 4,
                      WebkitAppearance: "none", appearance: "none",
                      background: "transparent", outline: "none", cursor: "pointer",
                      position: "absolute", top: 0, left: 0,
                    }}
                  />
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
                  {[1,2,3,4,5,6,7,8,9,10].map(n => (
                    <span key={n} style={{
                      fontSize: 8, color: n <= severity ? PALETTE.textSecondary : PALETTE.textMuted,
                      fontFamily: "monospace",
                    }}>{n}</span>
                  ))}
                </div>
              </div>

              {/* Manual approve */}
              <div
                onClick={() => setManualApprove(v => !v)}
                style={{
                  display: "flex", alignItems: "center", gap: 10,
                  padding: "8px 12px",
                  background: manualApprove ? `${PALETTE.success}10` : PALETTE.bg,
                  border: `1px solid ${manualApprove ? PALETTE.success + "40" : PALETTE.border}`,
                  borderRadius: 6, cursor: "pointer",
                  transition: "all 0.2s",
                }}
              >
                <div style={{
                  width: 16, height: 16, borderRadius: 3,
                  border: `1px solid ${manualApprove ? PALETTE.success : PALETTE.textMuted}`,
                  background: manualApprove ? PALETTE.success : "transparent",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 10, color: PALETTE.bg, flexShrink: 0,
                  transition: "all 0.2s",
                }}>
                  {manualApprove ? "✓" : ""}
                </div>
                <span style={{ fontSize: 10, color: manualApprove ? PALETTE.success : PALETTE.textSecondary, letterSpacing: "0.04em" }}>
                  MANUAL OVERRIDE — FORCE APPROVE
                </span>
              </div>

              {/* Analyze button */}
              <button
                onClick={handleAnalyze}
                disabled={analyzing || !productId || !actualComplaint}
                style={{
                  padding: "11px 20px",
                  background: analyzing || !productId ? PALETTE.border : `linear-gradient(135deg, ${PALETTE.accent}, ${PALETTE.purple})`,
                  border: "none", borderRadius: 7,
                  color: PALETTE.textPrimary, fontSize: 11, fontWeight: 700,
                  cursor: analyzing || !productId ? "not-allowed" : "pointer",
                  fontFamily: "inherit", letterSpacing: "0.1em",
                  transition: "all 0.3s",
                  position: "relative", overflow: "hidden",
                }}
              >
                {analyzing ? (
                  <span>⟳ {stepLabels[step]}…</span>
                ) : (
                  <span>⬡ RUN ANALYSIS</span>
                )}
              </button>

              {/* Step progress during analysis */}
              {analyzing && (
                <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                  {stepLabels.slice(0, -1).map((l, i) => (
                    <div key={l} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <div style={{
                        width: 12, height: 12, borderRadius: "50%",
                        border: `1px solid ${i < step ? PALETTE.success : i === step ? PALETTE.accent : PALETTE.textMuted}`,
                        background: i < step ? PALETTE.success : "transparent",
                        display: "flex", alignItems: "center", justifyContent: "center",
                        fontSize: 7, color: PALETTE.bg, flexShrink: 0,
                        transition: "all 0.3s",
                      }}>
                        {i < step ? "✓" : ""}
                      </div>
                      <span style={{
                        fontSize: 10,
                        color: i < step ? PALETTE.success : i === step ? PALETTE.accent : PALETTE.textMuted,
                        transition: "color 0.3s",
                      }}>{l}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Right Panel — Analysis Output */}
          <div style={{
            background: PALETTE.surface,
            border: `1px solid ${analyzed ? riskColor + "40" : PALETTE.border}`,
            borderRadius: 10, overflow: "hidden",
            transition: "border-color 0.5s",
            position: "relative",
          }}>
            <Scanline active={analyzing} />
            <div style={{
              padding: "12px 18px",
              borderBottom: `1px solid ${PALETTE.border}`,
              display: "flex", alignItems: "center", justifyContent: "space-between",
            }}>
              <span style={{ fontSize: 10, color: PALETTE.textSecondary, letterSpacing: "0.1em" }}>ANALYSIS OUTPUT</span>
              {analyzed && (
                <Tag color={riskColor}>
                  {riskLevel.toUpperCase()} RISK
                </Tag>
              )}
            </div>

            {!analyzed && !analyzing && (
              <div style={{
                display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
                height: 320, gap: 12,
              }}>
                <div style={{
                  width: 56, height: 56, borderRadius: "50%",
                  border: `1px solid ${PALETTE.border}`,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 24, color: PALETTE.textMuted,
                }}>⬡</div>
                <span style={{ fontSize: 11, color: PALETTE.textMuted }}>Awaiting analysis input</span>
              </div>
            )}

            {analyzing && (
              <div style={{
                display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
                height: 320, gap: 16,
              }}>
                <div style={{
                  width: 48, height: 48,
                  border: `2px solid ${PALETTE.border}`,
                  borderTop: `2px solid ${PALETTE.accent}`,
                  borderRadius: "50%",
                  animation: "spin 0.8s linear infinite",
                }} />
                <span style={{ fontSize: 11, color: PALETTE.accent }}>{stepLabels[step]}…</span>
              </div>
            )}

            {analyzed && (
              <div style={{ padding: 18, display: "flex", flexDirection: "column", gap: 16 }}>
                {/* Product Info */}
                <div style={{
                  padding: "10px 14px",
                  background: PALETTE.bg,
                  border: `1px solid ${PALETTE.border}`,
                  borderRadius: 6,
                  display: "flex", alignItems: "center", justifyContent: "space-between",
                }}>
                  <div>
                    <div style={{ fontSize: 12, color: PALETTE.textPrimary, fontWeight: 600 }}>
                      {product ? product.name : "Unknown Product"}
                    </div>
                    <div style={{ fontSize: 10, color: PALETTE.textSecondary, marginTop: 2 }}>
                      ID: {productId} · {product ? `${product.reviews} reviews · ★ ${product.rating}` : "Not in catalog"}
                    </div>
                  </div>
                  {product && <Tag color={PALETTE.teal}>{product.type}</Tag>}
                </div>

                {/* Scores grid */}
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                  <MetricCard
                    label="COMPLAINT CLASS"
                    value={predictedClass}
                    icon="◈"
                    color={PALETTE.purple}
                    active={analyzed}
                  />
                  <MetricCard
                    label="IMAGE RESULT"
                    value={imagePrediction}
                    icon={imagePrediction === "Defective" ? "✗" : "✓"}
                    color={imagePrediction === "Defective" ? PALETTE.danger : PALETTE.success}
                    active={analyzed}
                  />
                </div>

                {/* Risk Arc + recommendation */}
                <div style={{
                  display: "flex", alignItems: "center", gap: 16,
                  padding: "12px 16px",
                  background: `${riskColor}08`,
                  border: `1px solid ${riskColor}25`,
                  borderRadius: 8,
                }}>
                  <RiskArc score={riskScore} />
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 10, color: PALETTE.textSecondary, marginBottom: 4 }}>RECOMMENDATION</div>
                    {manualApprove ? (
                      <div style={{ fontSize: 13, color: PALETTE.success, fontWeight: 600 }}>
                        ✓ Manually approved by operator
                      </div>
                    ) : riskScore >= 70 ? (
                      <div style={{ fontSize: 13, color: PALETTE.danger, fontWeight: 600 }}>
                        ✗ Reject — escalate for review
                      </div>
                    ) : riskScore >= 40 ? (
                      <div style={{ fontSize: 13, color: PALETTE.warning, fontWeight: 600 }}>
                        ⚠ Manual review required
                      </div>
                    ) : (
                      <div style={{ fontSize: 13, color: PALETTE.success, fontWeight: 600 }}>
                        ✓ Auto-approve return
                      </div>
                    )}
                    <div style={{ fontSize: 10, color: PALETTE.textSecondary, marginTop: 4 }}>
                      Severity {severity}/10 · {imagePrediction === "Defective" ? "+40 image penalty" : "image normal"}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Bottom: Global Intelligence Panel */}
        <div style={{
          background: PALETTE.surface,
          border: `1px solid ${PALETTE.border}`,
          borderRadius: 10, overflow: "hidden",
        }}>
          <div style={{
            padding: "12px 18px",
            borderBottom: `1px solid ${PALETTE.border}`,
            display: "flex", alignItems: "center", justifyContent: "space-between",
          }}>
            <span style={{ fontSize: 10, color: PALETTE.textSecondary, letterSpacing: "0.1em" }}>
              GLOBAL REVIEW INTELLIGENCE
            </span>
            <span style={{ fontSize: 10, color: PALETTE.textMuted }}>
              {(MOCK_SENTIMENT.Positive + MOCK_SENTIMENT.Neutral + MOCK_SENTIMENT.Negative).toLocaleString()} reviews analysed
            </span>
          </div>

          <div style={{ padding: 18, display: "grid", gridTemplateColumns: "auto 1fr 1fr 1fr", gap: 20, alignItems: "start" }}>
            {/* Donut */}
            <div>
              <div style={{ fontSize: 10, color: PALETTE.textSecondary, letterSpacing: "0.08em", marginBottom: 10 }}>
                SENTIMENT SPLIT
              </div>
              <SentimentDonut data={MOCK_SENTIMENT} active={analyzed} />
            </div>

            {/* Bar chart */}
            <div>
              <div style={{ fontSize: 10, color: PALETTE.textSecondary, letterSpacing: "0.08em", marginBottom: 10 }}>
                DISTRIBUTION
              </div>
              {[
                { label: "Positive", val: MOCK_SENTIMENT.Positive, total: MOCK_SENTIMENT.Positive + MOCK_SENTIMENT.Neutral + MOCK_SENTIMENT.Negative, color: PALETTE.success },
                { label: "Neutral", val: MOCK_SENTIMENT.Neutral, total: MOCK_SENTIMENT.Positive + MOCK_SENTIMENT.Neutral + MOCK_SENTIMENT.Negative, color: PALETTE.warning },
                { label: "Negative", val: MOCK_SENTIMENT.Negative, total: MOCK_SENTIMENT.Positive + MOCK_SENTIMENT.Neutral + MOCK_SENTIMENT.Negative, color: PALETTE.danger },
              ].map(s => (
                <div key={s.label} style={{ marginBottom: 10 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                    <span style={{ fontSize: 10, color: PALETTE.textSecondary }}>{s.label}</span>
                    <span style={{ fontSize: 10, color: PALETTE.textPrimary, fontFamily: "monospace" }}>
                      {analyzed ? `${((s.val / s.total) * 100).toFixed(1)}%` : "—"}
                    </span>
                  </div>
                  <div style={{ height: 5, borderRadius: 2, background: PALETTE.border }}>
                    <div style={{
                      height: "100%", borderRadius: 2,
                      width: analyzed ? `${(s.val / s.total) * 100}%` : "0%",
                      background: s.color,
                      transition: "width 1s cubic-bezier(0.34,1.56,0.64,1)",
                      boxShadow: analyzed ? `0 0 6px ${s.color}60` : "none",
                    }} />
                  </div>
                </div>
              ))}
            </div>

            {/* Top complaint categories */}
            <div>
              <div style={{ fontSize: 10, color: PALETTE.textSecondary, letterSpacing: "0.08em", marginBottom: 10 }}>
                TOP COMPLAINT TYPES
              </div>
              {[
                { label: "Size Issue", pct: 34, color: PALETTE.accent },
                { label: "Wrong Colour", pct: 24, color: PALETTE.purple },
                { label: "Defective", pct: 18, color: PALETTE.danger },
                { label: "Not as Described", pct: 15, color: PALETTE.warning },
                { label: "Other", pct: 9, color: PALETTE.textMuted },
              ].map(c => (
                <div key={c.label} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 7 }}>
                  <div style={{ width: 3, height: 14, borderRadius: 1, background: c.color, flexShrink: 0 }} />
                  <span style={{ fontSize: 10, color: PALETTE.textSecondary, flex: 1 }}>{c.label}</span>
                  <div style={{ width: 50, height: 3, borderRadius: 1, background: PALETTE.border }}>
                    <div style={{
                      height: "100%", borderRadius: 1, background: c.color,
                      width: analyzed ? `${c.pct}%` : "0%",
                      transition: "width 1.2s ease",
                    }} />
                  </div>
                  <span style={{ fontSize: 10, color: PALETTE.textPrimary, fontFamily: "monospace", minWidth: 26, textAlign: "right" }}>
                    {analyzed ? `${c.pct}%` : "—"}
                  </span>
                </div>
              ))}
            </div>

            {/* Negative reviews */}
            <div>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
                <span style={{ fontSize: 10, color: PALETTE.textSecondary, letterSpacing: "0.08em" }}>
                  RECENT NEGATIVE
                </span>
                <button
                  onClick={() => setShowNegReviews(v => !v)}
                  style={{
                    fontSize: 9, color: PALETTE.accent,
                    background: "none", border: "none", cursor: "pointer",
                    fontFamily: "inherit", letterSpacing: "0.04em",
                  }}
                >
                  {showNegReviews ? "▲ HIDE" : "▼ SHOW"}
                </button>
              </div>
              {showNegReviews && analyzed ? (
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  {MOCK_NEG_REVIEWS.slice(0, 3).map((r, i) => (
                    <div key={i} style={{
                      padding: "8px 10px",
                      background: `${PALETTE.danger}08`,
                      border: `1px solid ${PALETTE.danger}20`,
                      borderLeft: `2px solid ${PALETTE.danger}60`,
                      borderRadius: 5,
                      fontSize: 10, color: PALETTE.textSecondary,
                      lineHeight: 1.5,
                    }}>
                      {r}
                    </div>
                  ))}
                </div>
              ) : !showNegReviews ? (
                <div style={{
                  padding: "20px 10px", textAlign: "center",
                  border: `1px dashed ${PALETTE.border}`,
                  borderRadius: 6, fontSize: 10, color: PALETTE.textMuted,
                }}>
                  {analyzed ? "412 flagged reviews" : "Run analysis first"}
                </div>
              ) : (
                <div style={{ fontSize: 10, color: PALETTE.textMuted, padding: "8px 0" }}>
                  Run analysis to load reviews
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div style={{
          marginTop: 16,
          display: "flex", justifyContent: "space-between", alignItems: "center",
          padding: "8px 4px",
          borderTop: `1px solid ${PALETTE.border}`,
        }}>
          <span style={{ fontSize: 9, color: PALETTE.textMuted, letterSpacing: "0.06em" }}>
            RETURN.AI v2.4.1 · MODEL: TEXT_CLASSIFIER + IMAGE_CNN + VADER_SENTIMENT
          </span>
          <span style={{ fontSize: 9, color: PALETTE.textMuted, letterSpacing: "0.06em" }}>
            {analyzed ? `LAST ANALYSIS: ${new Date().toLocaleTimeString()}` : "NO ANALYSIS RUN"}
          </span>
        </div>
      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        input[type=range]::-webkit-slider-thumb {
          -webkit-appearance: none;
          width: 14px; height: 14px;
          border-radius: 50%;
          background: ${PALETTE.accent};
          cursor: pointer;
          box-shadow: 0 0 6px ${PALETTE.accent}80;
        }
        input[type=range]::-moz-range-thumb {
          width: 14px; height: 14px;
          border-radius: 50%;
          background: ${PALETTE.accent};
          cursor: pointer;
          border: none;
        }
        * { box-sizing: border-box; }
      `}</style>
    </div>
  );
}
