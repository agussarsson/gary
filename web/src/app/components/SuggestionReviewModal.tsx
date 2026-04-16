"use client";

import { useMemo, useState } from "react";
import { DeliverSuggestion, ProgramJson, suggestionDecision } from "@/lib/api";
import {
  applyAdjustmentDraftToProgram,
  buildSuggestionDraftRows,
  describeAdjustment,
  SuggestionDraftRow,
  toSuggestionCard,
} from "@/lib/suggestions";

type Props = {
  programJson: ProgramJson;
  suggestions: DeliverSuggestion[];
  onClose: () => void;
  onProgramJsonUpdated: (next: ProgramJson) => void;
};

export default function SuggestionReviewModal({
  programJson,
  suggestions,
  onClose,
  onProgramJsonUpdated,
}: Props) {
  const [queue, setQueue] = useState<DeliverSuggestion[]>(suggestions);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [drafts, setDrafts] = useState<Record<string, SuggestionDraftRow[]>>({});

  const current = queue[0] ?? null;
  const currentCard = useMemo(() => (current ? toSuggestionCard(current) : null), [current]);

  function currentDraft(id: string, fallback: SuggestionDraftRow[]) {
    return drafts[id] ?? fallback.map((a) => ({ ...a }));
  }

  function updateDraft(id: string, next: SuggestionDraftRow[]) {
    setDrafts((prev) => ({ ...prev, [id]: next }));
  }

  function dropCurrent() {
    setQueue((q) => q.slice(1));
    setEditingId(null);
  }

  async function handleDecision(decision: "accept" | "decline" | "modify") {
    if (!current || !currentCard) return;
    setError(null);
    setBusyId(current.id);
    try {
      if (decision === "modify") {
        const fallbackDraft = buildSuggestionDraftRows(programJson, currentCard.adjustments);
        const draft = currentDraft(current.id, fallbackDraft);
        const modifiedProgram = applyAdjustmentDraftToProgram(programJson, draft);
        const result = await suggestionDecision(current.id, {
          decision: "modify",
          modified_program_json: modifiedProgram,
        });
        if (result.program_json) onProgramJsonUpdated(result.program_json);
      } else {
        const result = await suggestionDecision(current.id, { decision });
        if (result.program_json) onProgramJsonUpdated(result.program_json);
      }
      dropCurrent();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not save suggestion decision");
    } finally {
      setBusyId(null);
    }
  }

  if (!current || !currentCard) {
    return (
      <div className="modal-overlay">
        <div className="modal-panel card">
          <h2>Suggestions reviewed</h2>
          <p className="muted">All suggestions from this workout are handled.</p>
          <button type="button" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    );
  }

  const fallbackDraftRows = buildSuggestionDraftRows(programJson, currentCard.adjustments);
  const draftRows = currentDraft(current.id, fallbackDraftRows);
  const phrases = currentCard.phrases;

  return (
    <div className="modal-overlay">
      <div className="modal-panel card stack">
        <div className="row-between">
          <h2>{currentCard.title}</h2>
          <span className="muted">
            {queue.length} remaining
          </span>
        </div>
        <p className="muted">{currentCard.subtitle}</p>
        {current.suggestion_type === "note_feedback" && phrases.length > 0 && (
          <p>{phrases[0]}</p>
        )}
        {current.rationale && <p className="muted">{current.rationale}</p>}

        <div className="stack">
          {draftRows.length === 0 && (
            <p className="muted">No exercise changes were generated for this note.</p>
          )}
          {draftRows.map((adj, idx) => (
            <div key={`${adj.exerciseName}-${adj.field}-${idx}`} className="card-nested">
              <p>
                <strong>{adj.exerciseName}</strong>:{" "}
                {describeAdjustment({
                  exercise_name: adj.exerciseName,
                  field: adj.field,
                  multiplier: adj.multiplier,
                  reason: adj.reason,
                })}
              </p>
              {phrases[idx] && <p className="muted">{phrases[idx]}</p>}
              <p className="muted">
                Current: {adj.currentSets} sets · {adj.currentReps}
                {adj.currentLoad != null ? ` · ${adj.currentLoad} kg` : ""}
              </p>
              <p className="muted">
                Suggested: {adj.proposedSets} sets · {adj.proposedReps}
                {adj.proposedLoad != null ? ` · ${adj.proposedLoad} kg` : ""}
              </p>
              {editingId === current.id && (
                <div className="row">
                  <label>
                    Load (kg)
                    <input
                      type="number"
                      step="0.5"
                      value={adj.editedLoad ?? ""}
                      onChange={(e) => {
                        const next = [...draftRows];
                        next[idx] = {
                          ...adj,
                          editedLoad:
                            e.target.value === "" ? null : Number(e.target.value),
                        };
                        updateDraft(current.id, next);
                      }}
                    />
                  </label>
                  <label>
                    Sets
                    <input
                      type="number"
                      min={1}
                      value={adj.editedSets}
                      onChange={(e) => {
                        const next = [...draftRows];
                        next[idx] = {
                          ...adj,
                          editedSets: Math.max(1, Number(e.target.value)),
                        };
                        updateDraft(current.id, next);
                      }}
                    />
                  </label>
                  <label>
                    Reps
                    <input
                      type="number"
                      min={1}
                      value={adj.editedReps}
                      onChange={(e) => {
                        const next = [...draftRows];
                        next[idx] = {
                          ...adj,
                          editedReps: Math.max(1, Number(e.target.value)),
                        };
                        updateDraft(current.id, next);
                      }}
                    />
                  </label>
                </div>
              )}
            </div>
          ))}
        </div>

        {error && <p className="error">{error}</p>}

        <div className="row">
          <button
            type="button"
            disabled={busyId === current.id}
            onClick={() => void handleDecision("accept")}
          >
            Accept
          </button>
          <button
            type="button"
            className="btn-muted"
            disabled={busyId === current.id}
            onClick={() => void handleDecision("decline")}
          >
            Decline
          </button>
          {editingId === current.id ? (
            <>
              <button
                type="button"
                disabled={busyId === current.id}
                onClick={() => void handleDecision("modify")}
              >
                Apply modified suggestion
              </button>
              <button
                type="button"
                className="btn-muted"
                onClick={() => setEditingId(null)}
              >
                Cancel modify
              </button>
            </>
          ) : (
            <button
              type="button"
              className="btn-muted"
              onClick={() => setEditingId(current.id)}
            >
              Prepare modify
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
