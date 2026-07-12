import { Act } from "./Act";

// The model chain, told as the six steps the thesis actually computes, in order.
// Values quoted here are model inputs (data/input/parameters/), not chart data.
const STEPS: { title: string; body: string }[] = [
  {
    title: "Heat piles up",
    body: "Every building gains heat: sun through windows, warm outside air, people, appliances. The model runs an hourly heat balance per building; whatever pushes one past its comfort threshold — 25 °C today — becomes cooling demand, surplus heat that must be removed. The urban heat island makes it much worse: on a hot day central The Hague has measured 8.6 °C warmer than its rural surroundings, and without that effect the city's cooling demand would be roughly a third of what it is.",
  },
  {
    title: "There are many ways to shed it",
    body: "Some heat never becomes a machine's problem — shading, ventilation and building fabric can avoid or dump part of it passively. The rest takes active cooling. The model tracks six technologies, from portable ACs to split units, chillers, and air-, ground- and water-source heat pumps.",
  },
  {
    title: "Cooling costs electricity",
    body: "Each technology needs electricity to move heat out: demand divided by its seasonal efficiency (SEER — from ~2.5 for a portable AC to ~7.5 for a water-source heat pump today, improving in every future scenario).",
  },
  {
    title: "…but only where cooling is installed",
    body: "Only about 15% of Hague homes have cooling, against roughly three-quarters of offices. These market-penetration rates, per building type, scale the hypothetical electricity down to what is actually drawn — about 101 GWh a year today.",
  },
  {
    title: "Electricity carries emissions",
    body: "Generating that electricity emits greenhouse gases in step with the grid's carbon intensity — the biggest lever between the 2050 paths, and 88% of cooling's climate impact today. This page tracks climate only; the thesis also assessed resource depletion.",
  },
  {
    title: "So does the equipment itself",
    body: "Making, installing and scrapping cooling machines has its own footprint, plus refrigerant leaks along the way (another 10% of today's impact). Installed capacity scales with the peak: the model assumes systems are sized to cover cooling demand for 98% of the year, riding out the hottest 2% of hours.",
  },
];

/** The model, in plain words — the chain from surplus heat to climate impact.
 *  A real sequence, so the numbered markers carry information, not decoration. */
export function HowItWorks() {
  return (
    <Act id="model" variant="near" eyebrow="Behind the numbers · the model" labelledBy="model-h">
      <h2 id="model-h">From heat to impact, in six steps</h2>
      <p className="lede">
        Every number on this page comes out of one chain, computed building by building and hour by
        hour for ~59,000 real buildings.
      </p>

      <ol className="steps">
        {STEPS.map((s, i) => (
          <li key={s.title} className="step">
            <span className="step__n" aria-hidden="true">
              {i + 1}
            </span>
            <h3 className="step__t">{s.title}</h3>
            <p className="step__b">{s.body}</p>
          </li>
        ))}
      </ol>

      <p className="note steps__note">
        Full method, validation and sensitivity analyses: chapter 3 of the thesis (link in the
        footer).
      </p>
    </Act>
  );
}
