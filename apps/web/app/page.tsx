import JournalDemo from "../components/JournalDemo";

export default function Home() {
  return (
    <main>
      <section className="hero">
        <div className="brand">CONTINUUM</div>
        <h1>The first running vertical slice.</h1>
        <p>Journal → API → Postgres → queue → worker → mock Context-MoDE → persisted state → UI.</p>
      </section>
      <JournalDemo />
    </main>
  );
}
