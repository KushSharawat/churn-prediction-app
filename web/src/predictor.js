// Re-implements the trained scikit-learn logistic regression pipeline in JavaScript:
//   1. StandardScaler on numeric features:  z = (x - mean) / scale
//   2. One-hot encoding (first category dropped) on categorical features
//   3. Linear score (log-odds) = intercept + sum(coef * feature)
//   4. Probability = sigmoid(score)
// The parameters come from ml/train_models.py via results.json.

const LABELS = {
  tenure_months: 'Tenure', monthly_charges: 'Monthly charges', support_calls: 'Support calls',
  senior_citizen: 'Senior citizen', contract: 'Contract', internet_service: 'Internet',
  tech_support: 'Tech support', payment_method: 'Payment', paperless_billing: 'Paperless billing',
}

const sigmoid = (z) => 1 / (1 + Math.exp(-z))

export function predictChurn(customer, params) {
  let score = params.intercept
  const contributions = []

  for (const f of params.numeric) {
    const z = (customer[f.name] - f.mean) / f.scale
    const value = f.coef * z // 0 for an average customer
    score += value
    contributions.push({ label: LABELS[f.name], value })
  }

  for (const f of params.categorical) {
    const value = f.levels[customer[f.name]] ?? 0
    score += value
    // Compare with the average effect of this feature so the bar reads as "vs typical"
    const levels = Object.values(f.levels)
    const avg = levels.reduce((a, b) => a + b, 0) / levels.length
    contributions.push({ label: `${LABELS[f.name]}: ${customer[f.name]}`, value: value - avg })
  }

  contributions.sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
  return { probability: sigmoid(score), contributions }
}
