import { useEffect, useMemo, useState } from "react";
import type { GridIndex, Source } from "../data/types";
import "./Nav.css";

const LINKS = [
  { id: "grids", label: "Grids" },
  { id: "replay", label: "Replay" },
  { id: "compare", label: "Compare" },
  { id: "robustness", label: "Robustness" },
  { id: "system", label: "System" },
];

interface Props {
  source: Source;
  onSource: (s: Source) => void;
  index: GridIndex | null;
  switching: boolean;
}

export default function Nav({ source, onSource, index, switching }: Props) {
  const [active, setActive] = useState("replay");
  const [solid, setSolid] = useState(false);
  const hasGrids = index !== null;
  const links = useMemo(() => LINKS.filter((l) => l.id !== "grids" || hasGrids), [hasGrids]);

  useEffect(() => {
    const onScroll = () => setSolid(window.scrollY > 40);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    const obs = new IntersectionObserver(
      (entries) => {
        const hit = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (hit) setActive(hit.target.id);
      },
      { rootMargin: "-20% 0px -70% 0px", threshold: [0.05, 0.25, 0.5] }
    );

    links.forEach((l) => {
      const el = document.getElementById(l.id);
      if (el) obs.observe(el);
    });

    return () => obs.disconnect();
  }, [links]);

  return (
    <nav className={`nav${solid ? " is-solid" : ""}`}>
      <div className="shell nav-inner">
        <a className="nav-brand" href="#top">
          <span className="nav-brand-mark" />
          <span className="nav-brand-text">
            Microgrid<span className="nav-brand-dim">/RL</span>
          </span>
        </a>

        <div className="nav-links">
          {links.map((l) => (
            <a
              key={l.id}
              href={`#${l.id}`}
              className={`nav-link${active === l.id ? " is-on" : ""}`}
            >
              {l.label}
            </a>
          ))}
        </div>

        {index && (
          <div
            className={`nav-grids${switching ? " is-busy" : ""}`}
            role="radiogroup"
            aria-label="Microgrid shown on this page"
          >
            <span className="nav-grids-label mono">Grid</span>
            <button
              role="radio"
              aria-checked={source === "base"}
              className={`nav-grid${source === "base" ? " is-on" : ""}`}
              onClick={() => onSource("base")}
              title="Original single-home study (Pecan Street day, SB3 agents)"
            >
              Home
            </button>
            {index.grids.map((g) => (
              <button
                key={g.id}
                role="radio"
                aria-checked={source === g.id}
                className={`nav-grid${source === g.id ? " is-on" : ""}`}
                onClick={() => onSource(g.id)}
                title={`${g.id} · ${g.name}`}
              >
                {g.id}
              </button>
            ))}
          </div>
        )}
      </div>
    </nav>
  );
}
