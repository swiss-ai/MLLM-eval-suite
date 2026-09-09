// Execute the actual generated browser code; only the DOM sink is replaced.
// No browser framework is needed to verify its calculations and emitted cells.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync(process.argv[2], 'utf8');
const payload = JSON.parse(html.match(/<script id="data" type="application\/json">(.*?)<\/script>/s)[1]);
const script = html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
function page(data) {
  const elements = new Map();
  const get = id => {
    if (!elements.has(id)) elements.set(id, {
      textContent: id === 'data' ? JSON.stringify(data) : '', innerHTML: '',
      classList: { toggle() {} }, style: { setProperty() {} },
    });
    return elements.get(id);
  };
  const context = vm.createContext({document: {
    getElementById: get, querySelectorAll: () => [], querySelector: () => null,
  }});
  vm.runInContext(script, context);
  return {get, run: code => vm.runInContext(code, context)};
}
function cards(p) {
  return [...p.get('cards').innerHTML.matchAll(/<div class="card ([^"]*)"[^>]*title="([^"]*)">(.*?)(?=<div class="card |$)/gs)]
    .map(m => ({best: m[1].includes('best'), model: m[2], body: m[3]}));
}
function synthetic(rows, defaults = ['a', 'b']) {
  return {groups: {}, models: ['a', 'b'], labels: {a: 'A', b: 'B'}, defaultSelected: defaults,
    table: rows.map(([task, metric, dir, a, b, cat = 'ASR']) => ({task, metric, dir, cat,
      framework: 'lmms-eval', cells: {
        ...(a == null ? {} : {a: {v: a, raw: a, run: 'a'}}),
        ...(b == null ? {} : {b: {v: b, raw: b, run: 'b'}}),
      }})), categories: ['ASR', 'Audio QA'], modality: {ASR: 'audio', 'Audio QA': 'audio'}};
}
let failures = 0;
function test(name, body) {
  try { body(); console.log('PASS', name); }
  catch (error) { failures++; console.error('FAIL', name, error.stack); }
}
test('all nine audio models rank the 13-row common error cohort by its minimum', () => {
  const p = page(payload);
  p.get('selall').onclick(); p.get('tab-audio').onclick();
  assert.equal(cards(p).length, 9);
  const pretrain = cards(p).find(c => /pretrain/i.test(c.model));
  assert.equal(pretrain.best, true);
  assert.match(pretrain.body, /23\.8/);
  assert.equal(cards(p).find(c => /qwen2[ -]audio/i.test(c.model)).best, false);
  assert.match(p.get('cards-note').textContent, /13/);
  assert.match(p.get('cards-note').textContent, /lower/i);
  p.get('avg-macro').onclick();
  assert.equal(cards(p).find(c => /pretrain/i.test(c.model)).best, true);
});
test('mixed error and accuracy cohorts do not emit a combined mean or winner', () => {
  const p = page(synthetic([['wer', 'wer', -1, 20, 40], ['qa', 'accuracy', 1, 80, 60]]));
  p.get('tab-audio').onclick();
  for (const card of cards(p)) {
    assert.equal(card.best, false);
    assert.match(card.body, /—/);
  }
  assert.match(p.get('cards-note').textContent, /mix|different directions/i);
  assert.doesNotMatch(p.get('matrix').innerHTML, /class="cmean mono best"/);
  assert.match(p.get('matrix').innerHTML, /mix|different directions/i);
  p.get('avg-macro').onclick();
  assert.equal(cards(p).some(c => c.best), false);
});
test('accuracy cards maximize while empty and single-model states have no winner', () => {
  const p = page(synthetic([['qa', 'accuracy', 1, 80, 60]]));
  p.get('tab-audio').onclick();
  assert.equal(cards(p).find(c => c.model === 'a').best, true);
  p.run("state.sel = new Set(['a']); renderAll()");
  assert.equal(cards(p)[0].best, false);
  p.get('selnone').onclick();
  assert.equal(cards(p).length, 0);
  assert.match(p.get('cards-note').textContent, /select models/i);
});
test('macro error mean weights categories equally and keeps sparse rows outside common cohort', () => {
  const p = page(synthetic([
    ['wer1', 'wer', -1, 10, 20], ['wer2', 'wer', -1, 10, 20],
    ['cer', 'cer', -1, 50, 30, 'Audio QA'], ['extra', 'wer', -1, 0, null],
  ]));
  p.get('tab-audio').onclick();
  assert.equal(cards(p).find(c => c.model === 'a').best, true); // 23.3 vs 23.3 (tie)
  assert.equal(cards(p).find(c => c.model === 'b').best, true);
  p.get('avg-macro').onclick(); // a: (10 + 50) / 2 = 30; b: (20 + 30) / 2 = 25
  assert.equal(cards(p).find(c => c.model === 'b').best, true);
  assert.equal(cards(p).find(c => c.model === 'a').best, false);
  assert.match(cards(p).find(c => c.model === 'b').body, /25\.0/);
  assert.match(p.get('cards-note').textContent, /3/);
});
test('error rows, category means, and baseline deltas reward lower raw error', () => {
  const p = page(synthetic([['wer', 'wer', -1, 120, 140], ['cer', 'cer', -1, 10, 20]]));
  p.get('tab-audio').onclick();
  p.get('base').onchange({target: {value: 'b'}});
  const matrix = p.get('matrix').innerHTML;
  assert.match(matrix, /class="cmean mono best"[^>]*>65\.0/);
  assert.match(matrix, /class="cell mono best"[^>]*>120\.0/);
  assert.match(matrix, /class="delta up">-20\.0/);
  assert.doesNotMatch(matrix, /-20\.0.*accuracy/);
});
test('models with disjoint coverage do not invent a common score', () => {
  const p = page(synthetic([['wer1', 'wer', -1, 10, null], ['wer2', 'wer', -1, null, 20]]));
  p.get('tab-audio').onclick();
  assert.equal(cards(p).some(c => c.best), false);
  assert.equal(cards(p).length, 2);
  assert.ok(cards(p).every(c => c.body.includes('—')));
});
if (failures) process.exitCode = 1;
