import { cn } from "../../lib/utils";

const STEPS = ["Microphone", "Test", "Voice Profile", "Position", "Ready"];

export default function SetupProgress({ current }: { current: number }) {
  return (
    <ol aria-label="Setup progress" className="flex items-center gap-2">
      {STEPS.map((label, i) => {
        const done = i < current;
        const active = i === current;
        return (
          <li key={label} className="flex flex-1 items-center gap-2 last:flex-none">
            <span
              aria-current={active ? "step" : undefined}
              className={cn(
                "grid h-7 w-7 shrink-0 place-items-center rounded-full font-mono text-[11px] font-semibold transition-colors",
                done || active ? "bg-[#111111] text-white" : "bg-white text-[#8A8A86] ring-1 ring-[#E4E4E0]"
              )}
            >
              {String(i + 1).padStart(2, "0")}
            </span>
            <span
              className={cn(
                "hidden text-[11px] font-semibold uppercase tracking-[0.1em] sm:inline",
                active ? "text-[#111111]" : "text-[#8A8A86]"
              )}
            >
              {label}
            </span>
            {i < STEPS.length - 1 && <span className={cn("h-px flex-1", done ? "bg-[#111111]" : "bg-[#E4E4E0]")} />}
          </li>
        );
      })}
    </ol>
  );
}
