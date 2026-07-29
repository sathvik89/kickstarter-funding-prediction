/**
 * Verify the browser predictor against scikit-learn.
 *
 * This deliberately loads and executes the *shipped* <script> block out of
 * docs/index.html rather than a copy of it, so the thing under test is the code
 * visitors actually run. A copy could drift from the page and pass anyway.
 *
 * Usage:  node tests/js_parity.mjs cases.json
 *   where cases.json is [{inputs: {...}, expected: <probability>}, ...] produced
 *   by the Python side, whose own parity with sklearn is asserted separately.
 *
 * Exits non-zero and prints the worst mismatch if anything disagrees.
 */

import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, '..');
const TOLERANCE = 1e-9;   // same payload, same formula: should agree to float noise

const html = readFileSync(resolve(ROOT, 'docs/index.html'), 'utf8');
const model = JSON.parse(readFileSync(resolve(ROOT, 'docs/model.json'), 'utf8'));
const cases = JSON.parse(readFileSync(resolve(process.argv[2]), 'utf8'));

// --- extract the page's script block -------------------------------------
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map((m) => m[1]);
if (scripts.length !== 1) {
  console.error(`expected exactly one <script> block in index.html, found ${scripts.length}`);
  process.exit(2);
}
const source = scripts[0];

// --- minimal DOM good enough for the page's own code ----------------------
function makeElement(id) {
  return {
    id, value: '', textContent: '', innerHTML: '', className: '', hidden: false,
    style: {}, dataset: {},
    setAttribute() {}, getAttribute: () => null, addEventListener() {},
    // outerHTML assignment is used only on the error path
    set outerHTML(_) {},
  };
}

const elements = new Map();
const getElementById = (id) => {
  if (!elements.has(id)) elements.set(id, makeElement(id));
  return elements.get(id);
};

const sandbox = {
  console,
  Math, JSON, Set, Map, Object, Array, Number, String, Boolean, Error, isNaN,
  document: {
    getElementById,
    documentElement: { setAttribute() {}, getAttribute: () => null },
  },
  localStorage: { getItem: () => null, setItem() {} },
  window: { matchMedia: () => ({ matches: false }) },
  // the page fetches model.json; hand it the parsed payload directly
  fetch: () => Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(model) }),
};
sandbox.globalThis = sandbox;

const context = vm.createContext(sandbox);
// Expose the internals the page keeps in module scope so we can drive them.
vm.runInContext(`${source}\n;globalThis.__score = score;
globalThis.__setThreshold = (v) => { THRESHOLD = v; };`, context);

// Let the stubbed fetch().then() chain settle before scoring.
await new Promise((r) => setImmediate(r));

// --- run the cases -------------------------------------------------------
const FIELD_IDS = {
  title: 'fTitle', goal: 'fGoal', duration: 'fDuration', category: 'fCat',
  main_category: 'fMain', country: 'fCountry', currency: 'fCurrency',
  year: 'fYear', month: 'fMonth',
};

let worst = { diff: 0 };
for (const { inputs, expected } of cases) {
  for (const [key, id] of Object.entries(FIELD_IDS)) {
    getElementById(id).value = String(inputs[key]);
  }
  const { p } = sandbox.__score();
  const diff = Math.abs(p - expected);
  if (diff > worst.diff) worst = { diff, inputs, expected, actual: p };
}

if (worst.diff > TOLERANCE) {
  console.error(`JS PARITY FAILED: max |diff| = ${worst.diff.toExponential(3)} > ${TOLERANCE}`);
  console.error(`  inputs:   ${JSON.stringify(worst.inputs)}`);
  console.error(`  expected: ${worst.expected}\n  actual:   ${worst.actual}`);
  process.exit(1);
}

console.log(`JS parity OK across ${cases.length} cases (max |diff| = ${worst.diff.toExponential(3)})`);
