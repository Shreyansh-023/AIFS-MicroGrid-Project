import { useEffect, useState } from "react";
import "./Nav.css";

const LINKS = [
  { id: "replay", label: "Replay" },
  { id: "compare", label: "Compare" },
  { id: "robustness", label: "Robustness" },
  { id: "system", label: "System" },
];

export default function Nav() {
  const [active, setActive] = useState("replay");
  const [solid, setSolid] = useState(false);

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

    LINKS.forEach((l) => {
      const el = document.getElementById(l.id);
      if (el) obs.observe(el);
    });

    return () => obs.disconnect();
  }, []);

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
          {LINKS.map((l) => (
            <a
              key={l.id}
              href={`#${l.id}`}
              className={`nav-link${active === l.id ? " is-on" : ""}`}
            >
              {l.label}
            </a>
          ))}
        </div>

      </div>
    </nav>
  );
}
