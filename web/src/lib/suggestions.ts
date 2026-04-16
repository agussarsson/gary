import { DeliverSuggestion, ProgramJson } from "@/lib/api";

export type SuggestionAdjustment = {
  exercise_name: string;
  field: string;
  multiplier?: number | null;
  reason?: string | null;
  body_part?: string | null;
};

export type SuggestionDraftRow = {
  exerciseName: string;
  field: string;
  multiplier?: number | null;
  reason?: string | null;
  currentLoad: number | null;
  currentSets: number;
  currentReps: number;
  proposedLoad: number | null;
  proposedSets: number;
  proposedReps: number;
  editedLoad: number | null;
  editedSets: number;
  editedReps: number;
};

export type SuggestionCard = {
  suggestion: DeliverSuggestion;
  adjustments: SuggestionAdjustment[];
  title: string;
  subtitle: string;
  phrases: string[];
};

function cleanReason(reason?: string | null): string {
  return (reason ?? "").trim().toLowerCase();
}

function phraseCandidates(adj: SuggestionAdjustment): string[] {
  const reason = cleanReason(adj.reason);
  if (reason.includes("too heavy")) {
    return [
      "Well done showing up. Should we reduce the weight a little next time?",
      "Nice work. Want to lower the load slightly for better control?",
    ];
  }
  if (reason.includes("too light")) {
    return [
      "Great effort. Should we increase the weight next session?",
      "You looked comfortable there. Want to bump the load a touch?",
    ];
  }
  if (reason.includes("pain")) {
    return [
      "Strong job completing the day. Should we dial this movement down to protect recovery?",
      "Great consistency. Want to make this exercise easier while the area settles?",
    ];
  }
  if (adj.multiplier != null && adj.multiplier < 1) {
    return [
      "Nice work today. Should we reduce this slightly for the next workout?",
      "You got it done. Want to make this movement a bit lighter next time?",
    ];
  }
  if (adj.multiplier != null && adj.multiplier > 1) {
    return [
      "Great session. Ready for a small progression next time?",
      "Solid effort today. Want to increase the challenge slightly?",
    ];
  }
  return [
    "Great work. Want to apply this adjustment for your next session?",
    "Nice completion. Should we use this tweak going forward?",
  ];
}

function pickPhrase(adj: SuggestionAdjustment): string {
  const options = phraseCandidates(adj);
  const seed = `${adj.exercise_name}:${adj.field}:${adj.reason ?? ""}`;
  let hash = 0;
  for (let i = 0; i < seed.length; i += 1) {
    hash = (hash * 31 + seed.charCodeAt(i)) >>> 0;
  }
  return options[hash % options.length];
}

function pctChange(multiplier?: number | null): string | null {
  if (multiplier == null) return null;
  const delta = (multiplier - 1) * 100;
  const rounded = Math.round(delta * 10) / 10;
  if (rounded === 0) return "keep the same load";
  if (rounded > 0) return `increase load by ${rounded}%`;
  return `decrease load by ${Math.abs(rounded)}%`;
}

export function roundLoadFloor(load: number): number {
  if (load < 12.5) return Math.floor(load);
  return Math.floor(load / 2.5) * 2.5;
}

export function describeAdjustment(adj: SuggestionAdjustment): string {
  if (adj.field === "load") {
    return pctChange(adj.multiplier) ?? "adjust load";
  }
  if (adj.field === "sets" && adj.multiplier != null) {
    const rounded = Math.round(adj.multiplier * 100) / 100;
    return `scale sets by ${rounded}x`;
  }
  if (adj.field === "reps" && adj.multiplier != null) {
    const rounded = Math.round(adj.multiplier * 100) / 100;
    return `scale reps by ${rounded}x`;
  }
  if (adj.field === "pain") {
    return "apply pain-aware conservative adjustment";
  }
  return `adjust ${adj.field}`;
}

function suggestionTitle(suggestion: DeliverSuggestion): string {
  if (suggestion.suggestion_type === "no_note_progression") return "Progression suggestion";
  if (suggestion.suggestion_type === "note_feedback") return "Feedback suggestion";
  return "Workout suggestion";
}

export function toSuggestionCard(suggestion: DeliverSuggestion): SuggestionCard {
  const adjustments = Array.isArray(suggestion.payload?.adjustments)
    ? (suggestion.payload.adjustments as SuggestionAdjustment[])
    : [];
  const subtitle = suggestion.day_name
    ? `Day: ${suggestion.day_name}`
    : "From your latest workout feedback";
  return {
    suggestion,
    adjustments,
    title: suggestionTitle(suggestion),
    subtitle,
    phrases: adjustments.map(pickPhrase),
  };
}

export function applyAdjustmentDraftToProgram(
  baseProgram: ProgramJson,
  rows: SuggestionDraftRow[],
): ProgramJson {
  const next = structuredClone(baseProgram);
  for (const row of rows) {
    for (const day of next.days) {
      for (const ex of day.exercises) {
        if (ex.name !== row.exerciseName) continue;
        ex.sets = Math.max(1, Math.round(row.editedSets));
        ex.reps = row.editedReps;
        ex.load =
          row.editedLoad == null ? null : roundLoadFloor(Number(row.editedLoad));
      }
    }
  }
  return next;
}

export function buildSuggestionDraftRows(
  program: ProgramJson,
  adjustments: SuggestionAdjustment[],
): SuggestionDraftRow[] {
  const rows: SuggestionDraftRow[] = [];
  for (const adj of adjustments) {
    let currentLoad: number | null = null;
    let currentSets = 3;
    let currentReps = 8;
    for (const day of program.days) {
      const ex = day.exercises.find((candidate) => candidate.name === adj.exercise_name);
      if (!ex) continue;
      currentLoad = ex.load ?? null;
      currentSets = ex.sets;
      currentReps = ex.reps;
      break;
    }

    let proposedLoad = currentLoad;
    let proposedSets = currentSets;
    let proposedReps = currentReps;

    if (adj.field === "load" && adj.multiplier != null && currentLoad != null) {
      proposedLoad = roundLoadFloor(currentLoad * adj.multiplier);
    } else if (adj.field === "sets" && adj.multiplier != null) {
      proposedSets = Math.max(1, Math.round(currentSets * adj.multiplier));
    } else if (adj.field === "reps" && adj.multiplier != null) {
      const factor = Math.round(adj.multiplier * 100) / 100;
      proposedReps = Math.max(1, Math.round(currentReps * factor));
    }

    rows.push({
      exerciseName: adj.exercise_name,
      field: adj.field,
      multiplier: adj.multiplier,
      reason: adj.reason,
      currentLoad,
      currentSets,
      currentReps,
      proposedLoad,
      proposedSets,
      proposedReps,
      editedLoad: proposedLoad,
      editedSets: proposedSets,
      editedReps: proposedReps,
    });
  }
  return rows;
}
