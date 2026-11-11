import { useMemo, useState } from 'react'
import {
  ResponsiveContainer, LineChart, Line, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, Cell,
} from 'recharts'
import results from './data/results.json'
import { predictChurn } from './predictor'
import './App.css'

const COLORS = { 'Logistic Regression': '#2563eb', 'Random Forest': '#0d9488', 'Gradient Boosting': '#d97706' }
const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`
const TABS = ['Predict', 'Model performance', 'Churn insights']

/* ---------------- Predict tab ---------------- */
const DEFAULT_CUSTOMER = {
  tenure_months: 8, monthly_charges: 75, support_calls: 3, senior_citizen: 0,
  contract: 'Month-to-month', internet_service: 'Fibre', tech_support: 'No',
  payment_method: 'Manual Payment', paperless_billing: 'Yes',
}
const OPTIONS = {
  contract: ['Month-to-month', 'One year', 'Two year'],
  internet_service: ['Fibre', 'DSL', 'No internet'],
  tech_support: ['Yes', 'No'],
  payment_method: ['Direct Debit', 'Credit Card', 'Manual Payment'],
  paperless_billing: ['Yes', 'No'],
}
const LABELS = {
  tenure_months: 'Tenure (months)', monthly_charges: 'Monthly charges (£)', support_calls: 'Support calls (last year)',
  senior_citizen: 'Senior citizen', contract: 'Contract', internet_service: 'Internet service',
  tech_support: 'Tech support', payment_method: 'Payment method', paperless_billing: 'Paperless billing',
}

function riskBand(p) {
  if (p >= 0.6) return { label: 'High risk', cls: 'high', action: 'Contact this customer now: offer a discounted 12-month contract or a free tech support add-on.' }
  if (p >= 0.35) return { label: 'Medium risk', cls: 'medium', action: 'Add to a retention email campaign and monitor support tickets.' }
  return { label: 'Low risk', cls: 'low', action: 'No action needed. Consider an upsell or loyalty reward.' }
}

function PredictTab() {
  const [c, setC] = useState(DEFAULT_CUSTOMER)
  const set = (k, v) => setC((prev) => ({ ...prev, [k]: v }))
  const { probability, contributions } = useMemo(() => predictChurn(c, results.predictor), [c])
  const band = riskBand(probability)
  const top = contributions.slice(0, 6)

  const slider = (k, min, max, step = 1) => (
    <label className="field" key={k}>
      <span>{LABELS[k]} <b>{k === 'monthly_charges' ? `£${c[k]}` : c[k]}</b></span>
      <input type="range" min={min} max={max} step={step} value={c[k]} onChange={(e) => set(k, Number(e.target.value))} />
    </label>
  )
  const select = (k) => (
    <label className="field" key={k}>
      <span>{LABELS[k]}</span>
      <select value={c[k]} onChange={(e) => set(k, e.target.value)}>
        {OPTIONS[k].map((o) => <option key={o}>{o}</option>)}
      </select>
    </label>
  )

  return (
    <div className="grid-2">
      <div className="card">
        <h2>Customer details</h2>
        <div className="form">
          {slider('tenure_months', 1, 72)}
          {slider('monthly_charges', 15, 120)}
          {slider('support_calls', 0, 8)}
          {select('contract')}
          {select('internet_service')}
          {select('tech_support')}
          {select('payment_method')}
          {select('paperless_billing')}
          <label className="field check">
            <input type="checkbox" checked={c.senior_citizen === 1} onChange={(e) => set('senior_citizen', e.target.checked ? 1 : 0)} />
            <span>Senior citizen</span>
          </label>
        </div>
        <button className="ghost" onClick={() => setC(DEFAULT_CUSTOMER)}>Reset example</button>
      </div>

      <div className="card">
        <h2>Churn prediction</h2>
        <div className={`result ${band.cls}`}>
          <div className="prob">{pct(probability, 0)}</div>
          <div className="band">{band.label}</div>
          <div className="meter"><div style={{ width: pct(probability) }} /></div>
        </div>
        <p className="action"><b>Suggested action:</b> {band.action}</p>

        <h2 className="mt">What is driving this prediction?</h2>
        <p className="muted small">Each feature's push on the churn score (log-odds) compared with an average customer. Red increases risk, green reduces it.</p>
        <ResponsiveContainer width="100%" height={230}>
          <BarChart data={top} layout="vertical" margin={{ left: 20, right: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" horizontal={false} />
            <XAxis type="number" tick={{ fontSize: 12 }} />
            <YAxis type="category" dataKey="label" width={150} tick={{ fontSize: 12 }} />
            <Tooltip formatter={(v) => v.toFixed(2)} />
            <Bar dataKey="value" name="Effect on churn score" radius={4}>
              {top.map((d) => <Cell key={d.label} fill={d.value > 0 ? '#dc2626' : '#16a34a'} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <p className="muted small">Live predictions use the logistic regression model, running in your browser.</p>
      </div>
    </div>
  )
}

/* ---------------- Model performance tab ---------------- */
function ConfusionMatrix({ m }) {
  const { tn, fp, fn, tp } = m.confusion
  return (
    <table className="cm">
      <thead><tr><th></th><th>Predicted stay</th><th>Predicted churn</th></tr></thead>
      <tbody>
        <tr><th>Actually stayed</th><td className="good">{tn}</td><td className="bad">{fp}</td></tr>
        <tr><th>Actually churned</th><td className="bad">{fn}</td><td className="good">{tp}</td></tr>
      </tbody>
    </table>
  )
}

function PerformanceTab() {
  const [selected, setSelected] = useState(results.best_model)
  const model = results.models.find((m) => m.model === selected)
  const random = [{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }]

  return (
    <>
      <div className="card">
        <h2>Model comparison</h2>
        <table>
          <thead>
            <tr><th>Model</th><th>CV ROC-AUC (5-fold)</th><th>Test ROC-AUC</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
          </thead>
          <tbody>
            {results.models.map((m) => (
              <tr key={m.model} className={m.model === results.best_model ? 'best' : ''}>
                <td><span className="dot" style={{ background: COLORS[m.model] }} />{m.model}{m.model === results.best_model && <span className="tag">best</span>}</td>
                <td>{m.cv_auc_mean.toFixed(3)} ± {m.cv_auc_std.toFixed(3)}</td>
                <td>{m.test_auc.toFixed(3)}</td>
                <td>{pct(m.accuracy)}</td>
                <td>{pct(m.precision)}</td>
                <td>{pct(m.recall)}</td>
                <td>{m.f1.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="muted small">
          Trained on {results.dataset.train_size.toLocaleString('en-GB')} customers and tested on {results.dataset.test_size.toLocaleString('en-GB')} unseen customers.
          Recall matters most here: missing a customer who is about to leave costs more than contacting one who would have stayed.
        </p>
      </div>

      <div className="grid-2">
        <div className="card">
          <h2>ROC curves (test set)</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart margin={{ right: 20, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="fpr" type="number" domain={[0, 1]} allowDuplicatedCategory={false} tick={{ fontSize: 12 }} label={{ value: 'False positive rate', position: 'insideBottom', offset: -5, fontSize: 12 }} />
              <YAxis domain={[0, 1]} tick={{ fontSize: 12 }} label={{ value: 'True positive rate', angle: -90, position: 'insideLeft', fontSize: 12 }} />
              <Tooltip formatter={(v) => v.toFixed(2)} labelFormatter={(v) => `FPR ${Number(v).toFixed(2)}`} />
              <Legend verticalAlign="top" />
              {results.models.map((m) => (
                <Line key={m.model} data={m.roc} name={m.model} type="linear" dataKey="tpr" stroke={COLORS[m.model]} strokeWidth={2} dot={false} />
              ))}
              <Line data={random} name="Random guess" dataKey="tpr" stroke="#9ca3af" strokeDasharray="5 5" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="card">
          <div className="row">
            <h2>Confusion matrix</h2>
            <select value={selected} onChange={(e) => setSelected(e.target.value)}>
              {results.models.map((m) => <option key={m.model}>{m.model}</option>)}
            </select>
          </div>
          <ConfusionMatrix m={model} />
          <p className="muted small">Decision threshold: {model.threshold}. The model catches {pct(model.recall, 0)} of customers who actually churned.</p>
          <h2 className="mt">Feature importance ({results.best_model})</h2>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={results.feature_importance} layout="vertical" margin={{ left: 20, right: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" horizontal={false} />
              <XAxis type="number" tick={{ fontSize: 12 }} />
              <YAxis type="category" dataKey="feature" width={130} tick={{ fontSize: 12 }} />
              <Tooltip formatter={(v) => v.toFixed(3)} />
              <Bar dataKey="importance" name="Drop in ROC-AUC when shuffled" fill="#2563eb" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
          <p className="muted small">Permutation importance: how much test ROC-AUC drops when a feature is randomly shuffled.</p>
        </div>
      </div>
    </>
  )
}

/* ---------------- Insights tab ---------------- */
function ChurnChart({ title, data, color }) {
  return (
    <div className="card">
      <h2>{title}</h2>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} margin={{ right: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" vertical={false} />
          <XAxis dataKey="group" tick={{ fontSize: 12 }} />
          <YAxis tickFormatter={(v) => pct(v, 0)} tick={{ fontSize: 12 }} />
          <Tooltip formatter={(v, n, p) => [`${pct(v)} (${p.payload.customers.toLocaleString('en-GB')} customers)`, 'Churn rate']} />
          <Bar dataKey="churn_rate" fill={color} radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

function InsightsTab() {
  const { eda } = results
  return (
    <>
      <div className="card findings">
        <h2>Key findings</h2>
        <ul>
          <li><b>Contract type is the strongest driver.</b> Month-to-month customers churn at {pct(eda.contract[0].churn_rate, 0)}, compared with {pct(eda.contract[2].churn_rate, 0)} on two-year contracts.</li>
          <li><b>New customers are most at risk.</b> {pct(eda.tenure_band[0].churn_rate, 0)} of customers in their first 6 months leave, falling to {pct(eda.tenure_band[4].churn_rate, 0)} after 4 years.</li>
          <li><b>Support calls are an early warning sign.</b> Churn more than doubles from 0 calls to 4 or more.</li>
          <li><b>Recommendation:</b> target month-to-month customers in their first year with contract upgrade offers, and flag anyone making 3+ support calls for proactive contact.</li>
        </ul>
      </div>
      <div className="grid-2">
        <ChurnChart title="Churn rate by contract" data={eda.contract} color="#2563eb" />
        <ChurnChart title="Churn rate by tenure (months)" data={eda.tenure_band} color="#0d9488" />
        <ChurnChart title="Churn rate by support calls" data={eda.support_calls} color="#d97706" />
        <ChurnChart title="Churn rate by payment method" data={eda.payment_method} color="#7c3aed" />
      </div>
    </>
  )
}

/* ---------------- App ---------------- */
export default function App() {
  const [tab, setTab] = useState(TABS[0])
  const { dataset } = results
  return (
    <div className="page">
      <header>
        <h1>Customer Churn Prediction</h1>
        <p className="muted">
          Predicting which telecom customers will leave · {dataset.customers.toLocaleString('en-GB')} customers · {pct(dataset.churn_rate)} churn rate ·
          scikit-learn models served in a React front end
        </p>
      </header>
      <nav className="tabs">
        {TABS.map((t) => <button key={t} className={t === tab ? 'active' : ''} onClick={() => setTab(t)}>{t}</button>)}
      </nav>
      {tab === 'Predict' && <PredictTab />}
      {tab === 'Model performance' && <PerformanceTab />}
      {tab === 'Churn insights' && <InsightsTab />}
      <footer className="muted small">Synthetic dataset generated for portfolio purposes.</footer>
    </div>
  )
}
